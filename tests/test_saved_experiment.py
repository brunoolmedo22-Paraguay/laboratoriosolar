"""Acceptance checks against the delivered full six-month 11×4 engineering run."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from cim26.schema import ROOT,MODELS,supsi_spec
from cim26.metrics import energy_metrics
from cim26.report import load_run,run_zip,import_run_zip
from cim26.data import sha256


@pytest.fixture(scope="module")
def run():
    paths=sorted(p for p in (ROOT/"runs").iterdir() if p.is_dir() and (p/"manifest_sha256.json").exists())
    assert paths,"Nenhuma execução completa incluída na entrega."
    return load_run(paths[-1])


def test_44_serial_records_shared_inputs_and_repeats(run):
    d=run["benchmark_runs"]
    assert len(d)==44 and set(d.model)==set(MODELS)
    assert d.groupby("model").size().eq(11).all()
    assert d.exit_code.eq(0).all() and d.deterministic.all() and d.nonconverged.eq(0).all()
    assert d.n_samples.nunique()==1 and d.input_sha256.nunique()==1
    assert d.groupby("model").inner_repeats.nunique().eq(1).all()
    assert (d.start_ns.to_numpy()[1:]>=d.end_ns.to_numpy()[:-1]).all()
    assert d.compute_s.gt(0).all() and (d.rss_peak_MiB>=d.rss_baseline_MiB).all()
    np.testing.assert_allclose(d.compute_s,d.compute_batch_s/d.inner_repeats)
    np.testing.assert_allclose(d.us_per_sample,d.compute_s/d.n_samples*1e6)


def test_round_outputs_deterministic(run):
    for model in MODELS:
        files=sorted((run["path"]/"round_outputs").glob("*"+model+".npz"))
        assert len(files)==11
        first=np.load(files[0],allow_pickle=False)["power"]
        for f in files[1:]:np.testing.assert_array_equal(first,np.load(f,allow_pickle=False)["power"])


def test_recalculate_from_saved_csv_and_hashes(run):
    actual=energy_metrics(run["predictions"],supsi_spec().p_stc_W,run["config"]["max_gap_minutes"])
    cols=["RMSE_W","MAE_W","MBE_W","E_observed_kWh","E_predicted_kWh","covered_hours"]
    np.testing.assert_allclose(actual[cols],run["metrics_energy"][cols],rtol=1e-10,atol=1e-10)
    manifest=json.loads((run["path"]/"manifest_sha256.json").read_text())
    assert all(sha256(run["path"]/p)==h for p,h in manifest.items())
    assert run["metadata"]["N_total"]==18705
    assert run["metadata"]["N_common"]==15457


def test_reopen_export_without_calculation(run,tmp_path,monkeypatch):
    import cim26.physics as physics
    def forbidden(*a,**k):raise AssertionError("Reopening must not predict")
    monkeypatch.setattr(physics,"predict",forbidden)
    path=import_run_zip(run_zip(run["path"]),tmp_path)
    reopened=load_run(path)
    pd.testing.assert_frame_equal(run["predictions"],reopened["predictions"])
    pd.testing.assert_frame_equal(run["metrics_energy"],reopened["metrics_energy"])
