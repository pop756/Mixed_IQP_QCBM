from __future__ import annotations

import numpy as np
import torch

from randomized_iqp.ancilla import branch_bitstrings, embed_visible_branches_into_ancilla, reduce_to_visible_branch, visible_generator_coefficients
from randomized_iqp.gates import build_visible_gates, expected_gate_count
from randomized_iqp.trainer import RandomizedIqpTrainer, branch_bits_for_index, label_branch_mapping


def test_gate_counts() -> None:
    layouts = {
        "fully_connected_order2": {"type": "fully_connected", "max_order": 2},
        "fully_connected_order4": {"type": "fully_connected", "max_order": 4},
        "fully_connected_order6": {"type": "fully_connected", "max_order": 6},
        "vertical_8x2": {"type": "vertical", "rows": 8, "cols": 2},
        "vertical_4x4": {"type": "vertical", "rows": 4, "cols": 4},
        "vertical_4x4_row_order2": {"type": "vertical", "rows": 4, "cols": 4, "max_row_order": 2},
    }
    expected = {
        "fully_connected_order2": 136,
        "fully_connected_order4": 2516,
        "fully_connected_order6": 14892,
        "vertical_8x2": 80,
        "vertical_4x4": 84,
        "vertical_4x4_row_order2": 64,
    }
    for name, layout in layouts.items():
        assert expected_gate_count(layout, 16) == expected[name]
        assert len(build_visible_gates(name, layout, 16)) == expected[name]


def test_branch_bitstrings() -> None:
    assert branch_bitstrings(3) == ["000", "001", "010", "011", "100", "101", "110", "111"]


def test_reduce_to_visible_branch_applies_signs() -> None:
    gates = [[[0]], [[16, 0]], [[17, 0]], [[16, 17, 0]], [[16, 17]]]
    params = np.array([1.0, 2.0, 3.0, 4.0, 99.0])
    visible_gates, visible_params = reduce_to_visible_branch(
        gates,
        params,
        visible_bits=16,
        ancilla_wires=[16, 17],
        branch_bits="01",
    )
    assert visible_gates == [[[0]]]
    assert np.allclose(visible_params, np.array([-4.0]))


def test_embed_visible_branches_into_ancilla_two_bit_theta_table() -> None:
    branch_circuits = {
        "00": ([[[0]]], np.array([0.6])),
        "01": ([[[0]]], np.array([0.3])),
        "10": ([[[0]]], np.array([-0.3])),
        "11": ([[[0]]], np.array([-0.6])),
    }

    gates, params, metadata = embed_visible_branches_into_ancilla(
        branch_circuits,
        visible_bits=1,
        ancilla_bits=2,
        ancilla_wires=[1, 2],
    )

    assert metadata["transform"] == "inverse_walsh_hadamard_over_ancilla_branches"
    coeffs = visible_generator_coefficients(gates, params, visible_bits=3)
    assert np.isclose(coeffs[(1, 0)], 0.45)
    assert np.isclose(coeffs[(2, 0)], 0.15)
    assert (0,) not in coeffs
    assert (1, 2, 0) not in coeffs

    for branch, (expected_gates, expected_params) in branch_circuits.items():
        visible_gates, visible_params = reduce_to_visible_branch(
            gates,
            params,
            visible_bits=1,
            ancilla_wires=[1, 2],
            branch_bits=branch,
        )
        assert visible_gates == expected_gates
        assert np.allclose(visible_params, expected_params)


def test_embed_visible_branches_into_ancilla_recovers_arbitrary_branch_circuits() -> None:
    branch_circuits = {
        "00": ([[[0]], [[1]], [[0, 1]]], np.array([0.2, -0.4, 0.7])),
        "01": ([[[0]], [[1]], [[0, 1]]], np.array([1.1, 0.5, -0.2])),
        "10": ([[[0]], [[1]], [[0, 1]]], np.array([-0.8, 0.3, 0.0])),
        "11": ([[[0]], [[1]], [[0, 1]]], np.array([0.9, -1.0, 1.3])),
    }

    gates, params, _ = embed_visible_branches_into_ancilla(
        branch_circuits,
        visible_bits=2,
        ancilla_bits=2,
    )

    for branch, (expected_gates, expected_params) in branch_circuits.items():
        visible_gates, visible_params = reduce_to_visible_branch(
            gates,
            params,
            visible_bits=2,
            ancilla_wires=[2, 3],
            branch_bits=branch,
        )
        expected = visible_generator_coefficients(expected_gates, expected_params, visible_bits=2)
        actual = visible_generator_coefficients(visible_gates, visible_params, visible_bits=2)
        assert actual.keys() == expected.keys()
        for key in expected:
            assert np.isclose(actual[key], expected[key])



def test_direct_max_train_layout_skips_previous_stage(tmp_path) -> None:
    trainer = RandomizedIqpTrainer.__new__(RandomizedIqpTrainer)
    trainer.qcbm = {"training_mode": "direct_max_ancilla"}
    calls: list[int] = []

    def fake_checkpoint_dir(layout_name: str, ancilla_bits: int):
        return tmp_path / layout_name / f"a{ancilla_bits}"

    def fail_load_stage(layout_name: str, ancilla_bits: int):
        raise AssertionError("direct max mode must not load previous ancilla stages")

    def fake_train_stage(
        layout_name: str,
        ancilla_bits: int,
        previous_gates=None,
        previous_params=None,
        stage_steps=None,
    ):
        assert layout_name == "layout"
        assert ancilla_bits == 4
        assert previous_gates is None
        assert previous_params is None
        assert stage_steps == 12
        calls.append(ancilla_bits)
        return {"losses": torch.as_tensor([0.4, 0.2], dtype=torch.float64)}

    trainer.checkpoint_dir = fake_checkpoint_dir
    trainer._load_stage_artifact = fail_load_stage
    trainer.train_stage = fake_train_stage

    result = RandomizedIqpTrainer.train_layout(
        trainer,
        "layout",
        max_ancilla_bits=4,
        start_ancilla_bits=4,
        stage_steps=12,
    )

    assert calls == [4]
    assert set(result) == {"a4"}
    assert result["a4"]["training_mode"] == "direct_max_ancilla"
    assert result["a4"]["final_loss"] == 0.2



def test_label_branch_mapping_uses_sorted_binary_branches() -> None:
    labels = np.array([3, 1, 3, 7, 1])
    assert branch_bits_for_index(2, 3) == "010"
    assert label_branch_mapping(labels, 2) == [
        {"branch_index": 0, "branch_bits": "00", "label": 1},
        {"branch_index": 1, "branch_bits": "01", "label": 3},
        {"branch_index": 2, "branch_bits": "10", "label": 7},
    ]


def test_label_conditioned_train_layout_uses_source_visible_artifact(tmp_path) -> None:
    trainer = RandomizedIqpTrainer.__new__(RandomizedIqpTrainer)
    trainer.qcbm = {"training_mode": "label_conditioned_ancilla"}
    source_path = tmp_path / "source_a0.pt"
    calls: list[dict[str, object]] = []

    def fake_checkpoint_dir(layout_name: str, ancilla_bits: int):
        return tmp_path / layout_name / f"a{ancilla_bits}"

    def fake_source_artifact(layout_name: str):
        assert layout_name == "layout"
        return source_path, {"gates": [[[0]]], "params": torch.as_tensor([0.25], dtype=torch.float64)}

    def fake_train_stage(
        layout_name: str,
        ancilla_bits: int,
        previous_gates=None,
        previous_params=None,
        stage_steps=None,
        label_condition_source=None,
    ):
        calls.append(
            {
                "layout_name": layout_name,
                "ancilla_bits": ancilla_bits,
                "previous_gates": previous_gates,
                "previous_params": previous_params,
                "stage_steps": stage_steps,
                "label_condition_source": label_condition_source,
            }
        )
        return {"losses": torch.as_tensor([0.3, 0.1], dtype=torch.float64)}

    trainer.checkpoint_dir = fake_checkpoint_dir
    trainer._load_label_condition_source_artifact = fake_source_artifact
    trainer.train_stage = fake_train_stage

    result = RandomizedIqpTrainer.train_layout(
        trainer,
        "layout",
        max_ancilla_bits=2,
        start_ancilla_bits=0,
        stage_steps=9,
    )

    assert set(result) == {"a2"}
    assert result["a2"]["training_mode"] == "label_conditioned_ancilla"
    assert result["a2"]["source_artifact"] == str(source_path)
    assert len(calls) == 1
    assert calls[0]["ancilla_bits"] == 2
    assert calls[0]["previous_gates"] == [[[0]]]
    assert np.allclose(calls[0]["previous_params"], np.array([0.25]))
    assert calls[0]["stage_steps"] == 9
    assert calls[0]["label_condition_source"] == str(source_path)
