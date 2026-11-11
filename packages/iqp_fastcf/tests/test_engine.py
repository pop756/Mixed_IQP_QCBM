"""GPU check: the fast engine matches the vendored iqp_hip_cf reference (constant and scheduled lr, callbacks), the
offload levels match the all-on-device mode, inputs are validated, and the evaluator and cluster labels do what they
say. Needs an AMD GPU, hipcc and the `reference` extras (jax, optax).

    python tests/test_engine.py        (or pytest tests/)
"""
import math, warnings
import numpy as np
from iqp_fastcf import train, cosine_lr, make_evaluator, make_inits, sigma_for, all_pairs, walsh
from iqp_fastcf.train import _train_reference, _lr_table, _plan, _subset_plan


def _case(a, n=20, N=500, seeds=(3, 10), steps=100):
    data = (np.random.default_rng(0).random((N, n)) < 0.3).astype(np.int8)
    c0, pi = make_inits(data, a, seeds)
    return (n, a, all_pairs(n), data, [sigma_for(n, 2), sigma_for(n, 6)], 1e-3, steps, c0, pi, list(seeds))


def _ref(args, lr=None, eval_steps=(), callback=None):
    n, a, edges, data, sig, lr0, steps, c0, pi, seeds = args
    lrs, const = _lr_table(lr0 if lr is None else lr, steps)
    return _train_reference(n, a, edges, data, sig, lrs, const, steps, c0, pi, seeds, 2000, 10, set(eval_steps), callback)


def _rel(x, y):
    return float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-300)))


def _quiet(f, *x, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return f(*x, **k)


def test_matches_reference():
    for a in (0, 2, 4):
        args = _case(a)
        got, ref = train(*args), _ref(args)
        assert got["plan"] == (0, 2 if a == 4 else 6, 1) and ref["plan"] is None, got["plan"]   # the fast path ran
        assert got["losses"].shape == ref["losses"].shape and got["params"].shape == ref["params"].shape
        assert _rel(got["losses"], ref["losses"]) < 1e-8, a
        assert np.max(np.abs(got["params"] - ref["params"])) < 1e-8, a


def test_schedule_and_callback():
    args = _case(2, steps=120)
    sched, ev = cosine_lr(1e-3, 120), (0, 37, 120)
    snaps = {"fast": {}, "ref": {}, "seq": {}}
    got = train(*args[:5], sched, *args[6:], eval_steps=ev, callback=lambda s, p: snaps["fast"].__setitem__(s, p))
    ref = _ref(args, sched, ev, lambda s, p: snaps["ref"].__setitem__(s, p))
    seq = train(*args[:5], [sched(s) for s in range(120)], *args[6:], eval_steps=ev,
                callback=lambda s, p: snaps["seq"].__setitem__(s, p))
    assert _rel(got["losses"], ref["losses"]) < 1e-8
    assert sorted(snaps["fast"]) == sorted(snaps["ref"]) == list(ev)
    for s in ev:
        assert np.max(np.abs(snaps["fast"][s] - snaps["ref"][s])) < 1e-8, s
        assert np.max(np.abs(snaps["seq"][s] - snaps["fast"][s])) < 1e-10, s
    assert np.array_equal(snaps["fast"][0], args[7]) and np.array_equal(snaps["fast"][120], got["params"])
    # the schedule really is used: a constant lr gives other parameters
    assert np.max(np.abs(train(*args)["params"] - got["params"])) > 1e-6
    assert abs(sched(0) - 1e-3) < 1e-18 and sched(120) == 0.0 and abs(sched(60) - 5e-4) < 1e-15


def test_low_memory_mode():
    args = _case(3)
    fast = train(*args)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        lean = train(*args, gpu_budget=1)      # a tiny budget forces host streaming (and cannot be met: a warning)
    assert any("cannot be met" in str(x.message) for x in w) and lean["plan"] == (2, 1, 8), lean["plan"]
    assert _rel(lean["losses"], fast["losses"]) < 1e-10
    assert np.max(np.abs(lean["params"] - fast["params"])) < 1e-10


def test_small_run_offload():
    # M = 2 < 3 chunks: level 2 cannot run, the budget is met with level 1 (moments on the host) instead of being ignored
    n, a, edges, data, sig, lr, steps, c0, pi, _ = _case(3)
    args = (n, a, edges, data, sig, lr, steps, c0[:2], pi[:2], [3])
    _, _, s_tot = _subset_plan(n, sig, 2000, 10)
    assert _plan(n, 8, c0.shape[2], 2, s_tot, 1, 1, 1)[0] == 1
    fast, lean = train(*args), _quiet(train, *args, gpu_budget=1)
    assert fast["plan"] == (0, 2, 1) and lean["plan"] == (1, 1, 8), lean["plan"]   # 0.1.0 dropped this run to level 0
    assert _rel(lean["losses"], fast["losses"]) < 1e-10
    assert np.max(np.abs(lean["params"] - fast["params"])) < 1e-10


def test_input_checks():
    n, a, edges, data, sig, lr, steps, c0, pi, seeds = _case(2, steps=5)
    bad = {
        "pi rows": dict(pi=pi[:-1]),
        "c0 for other edges": dict(c0=c0[:, :, :-1]),
        "+-1 data": dict(data=1 - 2 * data),
        "duplicate edge": dict(edges=np.vstack([edges, edges[:1, ::-1]]), c0=np.pad(c0, ((0, 0), (0, 0), (0, 1)))),
        "self edge": dict(edges=np.vstack([edges[:-1], [[4, 4]]])),
        "models vs seeds": dict(c0=c0[:-1], pi=pi[:-1]),
        "nan angle": dict(c0=np.where(np.arange(c0.size).reshape(c0.shape) == 7, np.nan, c0)),
        "lr length": dict(lr=[1e-3] * (steps - 1)),
        "eval without callback": dict(eval_steps=(2,)),
        "eval out of range": dict(eval_steps=(steps + 1,), callback=lambda s, p: None),
    }
    for name, kw in bad.items():
        k = dict(n=n, a=a, edges=edges, data=data, sigmas=sig, lr=lr, steps=steps, c0=c0, pi=pi, seeds=seeds)
        k.update(kw)
        try:
            train(**k)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for {name}")


def test_evaluator():
    # product branches (zero couplings): z_A = sum_l pi_l prod_{i in A} cos(2 theta_il), theta = W c; exact in numpy
    n, a, keval = 12, 1, 3000
    rng = np.random.default_rng(5)
    x_test = (rng.random((700, n)) < 0.4).astype(np.int8)
    edges = all_pairs(n)
    th = np.zeros((2, 2, n + len(edges)))
    th[:, :, :n] = rng.uniform(0, 1.5, (2, 2, n))
    params = np.einsum("kl,mlg->mkg", walsh(a).T, th) / 2           # c = W^T theta / L
    pi = np.array([[0.3, 0.7], [0.5, 0.5]])
    mean, per = make_evaluator(n, a, edges, x_test, keval=keval)(params, pi)
    s = 1.0 - 2.0 * x_test
    want = np.zeros((2, 6))
    for j, mb in enumerate(range(1, 7)):
        mask = np.random.default_rng(1000 + mb).random((keval, n)) < mb / n
        for A in (np.nonzero(r)[0] for r in mask if 1 <= r.sum() <= 10):
            t = s[:, A].prod(1).mean()
            z = (pi * np.cos(2 * th[:, :, A]).prod(-1)).sum(-1)
            want[:, j] += (z - t) ** 2 / keval
    assert _rel(per, want) < 1e-10 and _rel(mean, want.mean(1)) < 1e-10


def test_cluster_labels():
    n, a, seed = 20, 2, 3
    data = (np.random.default_rng(0).random((500, n)) < 0.3).astype(np.int8)
    y = np.random.default_rng(1).integers(0, 4, len(data))
    c0, pi = make_inits(data, a, [seed], labels=y)
    d0, q0 = make_inits(data, a, [seed])
    assert np.array_equal(c0[:2], d0[:2]) and np.array_equal(pi[:2], q0[:2])      # global, agnostic unchanged
    assert np.allclose(pi[2], np.bincount(y, minlength=4) / len(y), atol=1e-15)
    G = c0.shape[2]
    noise = np.random.default_rng(seed).uniform(-(math.pi / 2) / math.sqrt(4 * G), (math.pi / 2) / math.sqrt(4 * G), (4, G))
    th = walsh(a) @ (c0[2] - noise)
    mu = np.stack([1 - 2 * data[y == l].mean(0) for l in range(4)])
    assert np.max(np.abs(np.cos(2 * th[:, :n]) - mu)) < 1e-12 and np.max(np.abs(th[:, n:])) < 1e-12
    for bad in (y[:-1], y + 4):
        try:
            make_inits(data, a, [seed], labels=bad)
        except ValueError:
            continue
        raise AssertionError("bad labels accepted")


def test_trainable_weights():
    # a two-component product mixture with masses 0.8 / 0.2, started from its cluster centre with uniform weights:
    # the trained weights recover the masses and the loss ends below the fixed-weight run
    n, rng = 16, np.random.default_rng(0)
    lab = rng.random(4000) < 0.2
    data = (rng.random((4000, n)) < np.where(lab[:, None], 0.9, 0.1)).astype(np.int8)
    c0, _ = make_inits(data, 1, [3], labels=lab.astype(np.int64))
    args = (n, 1, all_pairs(n), data, [sigma_for(n, 2), sigma_for(n, 6)], 1e-2, 800, c0[2:3], np.full((1, 2), 0.5), [3])
    fixed, tr = train(*args), train(*args, train_pi=True)
    assert abs(np.sort(tr["pi"][0])[1] - (1 - lab.mean())) < 0.01, tr["pi"]
    assert tr["losses"][-50:].mean() < 0.2 * fixed["losses"][-50:].mean()
    # a zero weight learning rate is the fixed-weight run
    z = train(*args, train_pi=True, pi_lr_scale=0.0)
    assert np.max(np.abs(z["pi"] - 0.5)) < 1e-15 and _rel(z["losses"], fixed["losses"]) < 1e-12
    # offload levels and branch parts give the same weights and angles; the callback gets the current weights
    args = _case(4)
    snaps = {}
    fast = train(*args, train_pi=True, eval_steps=(0, 100), callback=lambda s, p, w: snaps.__setitem__(s, w))
    lean = _quiet(train, *args, train_pi=True, gpu_budget=1)
    assert lean["plan"] == (2, 1, 16), lean["plan"]
    assert np.max(np.abs(lean["pi"] - fast["pi"])) < 1e-12 and np.max(np.abs(lean["params"] - fast["params"])) < 1e-10
    assert np.array_equal(snaps[0], args[8]) and np.array_equal(snaps[100], fast["pi"])
    assert np.allclose(fast["pi"].sum(1), 1.0) and np.ptp(fast["pi"], axis=1).min() > 1e-3   # the weights moved
    try:
        train(*args[:8], np.where(np.arange(16) == 0, 1.0, 0.0)[None].repeat(len(args[7]), 0), args[9], train_pi=True)
        raise AssertionError("a zero initial weight must be rejected")
    except ValueError:
        pass


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name, flush=True)
    print("iqp_fastcf: all checks passed")
