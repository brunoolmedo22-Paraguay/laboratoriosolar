"""Scientific failures must remain visible; invalid ZIPs may not escape the history."""
from io import BytesIO
from pathlib import Path
import json
import zipfile
import numpy as np
import pandas as pd
import pytest
from cim26.schema import ExperimentConfig,supsi_spec,ROOT
from cim26.metrics import energy_metrics
from cim26.report import import_run_zip,load_run,plot_arrays
from cim26.experiment import acquire_experiment_lock,release_experiment_lock


def test_empty_support_is_unavailable_not_zero_energy():
    p=pd.DataFrame({"timestamp":pd.date_range("2019-01-01",periods=3,freq="5min"),
        "Pm":[100.,100.,100.],"eligible":True,"common_valid":False,
        **{m:np.nan for m in ["M1","M2","M3","M4"]}})
    r=energy_metrics(p,supsi_spec().p_stc_W)
    assert r.N_common.eq(0).all()
    assert r.RMSE_W.isna().all() and r.E_observed_kWh.isna().all()


def test_export_import_rejects_path_traversal(tmp_path):
    out=BytesIO()
    with zipfile.ZipFile(out,"w") as z:
        for n in ["metadata.json","config.json","manifest_sha256.json"]:z.writestr(n,"{}")
        z.writestr("predictions.csv","timestamp\n")
        z.writestr("../escape.txt","bad")
    with pytest.raises(ValueError,match="Caminho inválido"):
        import_run_zip(out.getvalue(),tmp_path)
    assert not (tmp_path.parent/"escape.txt").exists()


def test_failed_workers_preserve_every_attempt(tmp_path,monkeypatch):
    import cim26.experiment as engine
    source=ROOT/"reference/handoff/input/SUPSI/raw/field_2019/2019_01.csv"
    subset=tmp_path/"2019_01.csv"
    pd.read_csv(source,sep=";").head(24).to_csv(subset,sep=";",index=False)
    def failure(model,arrays,*args,**kwargs):
        return dict(model=model,exit_code=17,error="Injected worker failure",compute_s=np.nan,
            us_per_sample=np.nan,nonconverged=len(arrays["g"]),deterministic=False),None
    monkeypatch.setattr(engine,"isolated_evaluation",failure)
    # Rendering of empty diagnostic plots isn't relevant to failure persistence.
    monkeypatch.setattr(engine,"generate_figures",lambda path:None)
    path=engine.run_experiment(ExperimentConfig(months=("2019_01",),rounds=3,scalability_sizes=(),
        thermal_sensitivity=()),root=tmp_path/"runs",sources=[subset])
    data=load_run(path)
    assert len(data["benchmark_runs"])==12
    assert data["benchmark_runs"].exit_code.eq(17).all()
    assert data["metadata"]["status"]=="complete_with_failures"
    assert data["metadata"]["N_common"]==0
    assert data["metrics_energy"].E_predicted_kWh.isna().all()


def test_frozen_config_normalizes_lists():
    config=ExperimentConfig(months=["2019_01"],scalability_sizes=[128])
    assert isinstance(config.months,tuple) and isinstance(config.scalability_sizes,tuple)


def test_lock_excludes_concurrency_and_releases(tmp_path):
    lock=tmp_path/"test.lock"
    first=acquire_experiment_lock(lock)
    try:
        with pytest.raises(RuntimeError,match="ativo"):
            acquire_experiment_lock(lock)
    finally:release_experiment_lock(first)
    second=acquire_experiment_lock(lock)
    release_experiment_lock(second)


def test_power_figure_breaks_missing_night_without_modifying_data():
    p=pd.DataFrame({"timestamp":pd.to_datetime(["2019-01-01 16:00","2019-01-01 16:05","2019-01-02 08:00"]),"P":[30.,20.,25.]})
    x,y=plot_arrays(p,"timestamp","P",7.5)
    assert len(x)==4 and np.isnat(x[2]) and np.isnan(y[2])
    assert len(p)==3 and p.P.notna().all()
