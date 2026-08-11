"""Tests for the ridgeline QA tool. Fast synthetic checks that hold the load-bearing
contracts: the witness may not reuse the snapper's signal, the snapper recovers a
planted offset and beats every baseline, the controls behave, and the Layer-3
geometry re-derivation flags out-of-radius voxels.
"""
import numpy as np
import pytest

from ridgeline import synth, harness, witness, lift


def test_witness_rejects_snap_and_producer_signals():
    """Anti-circularity contract, enforced in code: the witness cannot be sato (the
    snapper's own signal) or frangi (the label producer's own operator)."""
    vol = np.zeros((8, 8, 8), np.float32)
    for bad in ("sato", "frangi"):
        with pytest.raises(ValueError):
            witness.witness_field(vol, witness_head=bad)
    # an allowed witness runs
    assert witness.witness_field(vol, witness_head="raw_ct").shape == vol.shape


def test_lift_features_are_sato_free():
    """The probe features must not include the snap signal."""
    assert "sato" in lift.FORBIDDEN_FEATURES
    feats = lift._feature_volumes(np.random.rand(6, 6, 6).astype(np.float32))
    assert feats.shape[-1] == 5                      # intensity, 2 smooths, std, grad
    import inspect
    src = inspect.getsource(lift._feature_volumes)
    # check for CALLS to a ridge operator, not the bare word, so an honest
    # "sato-free" comment does not trip the guard. Using any of these means
    # invoking it, which is the thing the contract forbids.
    for op in ("sato(", "meijering(", "frangi(", "hessian"):
        assert op not in src.lower(), f"lift feature extractor invokes {op}"



def test_planted_k2_recovery_beats_all_baselines():
    """A 2-voxel planted shift is recovered by the ridge snapper below half a voxel,
    beating snap-to-brightest, the random control and the frangi arm."""
    row = harness.run_cell(synth.plant_tube_shift(2, size=64, contrast=8.0, noise=2.0))
    assert row["pre_snap_error"] > 1.5
    assert row["ridge_recovery"] < 0.5
    assert row["ridge_recovery"] < row["brightest_recovery"]
    assert row["ridge_recovery"] < row["random_recovery"]
    assert row["ridge_recovery"] < row["frangi_recovery"]


def test_null_controls_idempotent_and_random_breaks():
    """Snapping a clean label barely moves it; a random move of comparable size
    damages it. Both are first-class outputs."""
    nc = harness.null_controls(size=64)
    assert nc["idempotence_median_move"] < 1.0
    assert nc["idempotence_post_error"] < 0.5
    assert nc["random_control_post_error"] > 1.0     # random move breaks a good label


def test_layer3_sandwich_flags_out_of_radius():
    """seed subset of published subset of dilate(seed, 3). A voxel planted beyond the
    dilation radius is counted as a provable defect; the seed is fully contained."""
    seed = np.zeros((40, 40, 40), bool)
    seed[20, 10:30, 20] = True                        # a short segment
    published = witness.dilate_by_inverse_edt(seed, 3)
    published[5, 5, 5] = True                          # stray voxel far outside radius 3
    r = witness.layer3_selfconsistency(seed, published, dilation=3)
    assert r["seed_containment_in_published"] == pytest.approx(1.0, abs=1e-6)
    assert r["defect_voxels_outside_dilation"] >= 1


@pytest.mark.slow
def test_lift_probe_recovers_and_controls_hold():
    """Regime A: the snap recovers most of the planted probe damage, snapping a clean
    label barely changes the probe, and a random move does not recover."""
    res = lift.lift_probe(k=2.0, n_phantoms=4, radius=2.0, size=64)
    assert res["dice_clean"] > res["dice_corrupted"]
    assert res["recovery_fraction"] > 0.3
    assert abs(res["null_control_delta"]) < 0.05
    assert res["random_control_recovery_fraction"] < res["recovery_fraction"]
