from dataclasses import replace
from pathlib import Path
import json
import hashlib
import importlib.util
import numpy as np
import pandas as pd
import pytest
from cim26.schema import ROOT,INPUT,MODELS,ModuleSpec,EffectiveSDM,ExperimentConfig,catalogue,supsi_spec
from cim26.data import laboratory,laboratory_from_xlsx,read_field,field_paths,quality_flags,custom_profile
from cim26.calibration import classic,multipoint
from cim26.physics import predict,sdm_points,scalar_reference,operating,current,thermal_conditions,iv_curve,availability
from cim26.metrics import energy_integral,energy_metrics,pareto_mask,grouped_statistics,day_bootstrap
from cim26.experiment import model_order


@pytest.fixture(scope="module")
def supsi_fit():
    return multipoint(supsi_spec(),laboratory())


def test_original_package_hashes():
    root=ROOT/"reference/handoff"
    manifest=json.loads((root/"MANIFIESTO_SHA256.json").read_text())
    for name,data in manifest["files"].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==data["sha256"]


def test_excel_rebuild_and_stc():
    a=laboratory_from_xlsx();b=laboratory()
    pd.testing.assert_frame_equal(a,b,check_dtype=False,rtol=1e-12)
    assert a.groupby("variable").size().to_dict()=={"Isc":22,"Pmp":22,"Voc":22}
    spec=supsi_spec()
    ref=a[(a.irradiance_poa_Wm2==1000)&(a.temperature_module_C==25)].set_index("variable").measured_value
    np.testing.assert_allclose(ref.loc[["Pmp","Isc","Voc"]],[spec.p_stc_W,spec.isc_A,spec.voc_V])
    assert spec.n_cells is None and spec.vmp_V is None and spec.imp_A is None
    assert spec.manufacturer is None and spec.commercial_model is None


def test_six_csv_dates_poa_and_counts():
    d=read_field(field_paths(ExperimentConfig().months))
    assert len(d)==18705
    assert d.timestamp.min()==pd.Timestamp("2019-01-01 08:40:00")
    assert d.timestamp.max()==pd.Timestamp("2019-06-30 17:00:00")
    assert d.iloc[0].Gpoa==pytest.approx(297.2125303)
    assert d.iloc[0].Gglob_hor==pytest.approx(184.0133333)
    q=quality_flags(d,ExperimentConfig())
    assert len(q)==len(d) and q.eligible.sum()==15457
    assert (q.loc[~q.eligible,"exclusion_reasons"]!="").all()


@pytest.mark.parametrize("model",["M1","M2","M3"])
def test_stc_and_zero(model):
    spec=supsi_spec()
    with np.errstate(all="raise"):
        result=predict(model,[0,1000],[25,25],[25,25],spec)
    np.testing.assert_allclose(result.power_W,[0,spec.p_stc_W])


def test_thermal_hand_calculation_and_percent_units():
    s=supsi_spec()
    r=predict("M2",[800],[40],[45],s).power_W[0]
    assert r==pytest.approx(s.p_stc_W*.8*(1-.00405*20))
    assert r<predict("M2",[800],[25],[25],s).power_W[0]
    tm,tc=thermal_conditions(np.array([0,800,1000]),t_back=25,delta_T_C=3)
    np.testing.assert_allclose(tc,[25,27.4,28])
    tm,tc=thermal_conditions(np.array([1000]),t_ambient=20,wind=2,method="faiman",spec=s,delta_T_C=3)
    assert tm[0]==pytest.approx(20+1000/(29.84+3.44*2))
    assert tc[0]-tm[0]==pytest.approx(3)


def test_supplied_reference_formulas():
    path=ROOT/"reference/handoff/reference_implementations/m1_m2_m3_reference.py"
    spec=importlib.util.spec_from_file_location("formula_reference",path);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    s=supsi_spec();g=np.array([0,50,100,600,1000,1500]);t=np.array([25,-10,15,40,75,60])
    expected=[ref.m1_irradiance(g,s.p_stc_W),ref.m2_pvwatts_dc(g,t,s.p_stc_W,s.gamma_pct_C/100),ref.m3_huld(g,t,s.p_stc_W)]
    for m,yy in zip(MODELS,expected):np.testing.assert_allclose(predict(m,g,t,t,s).power_W,yy,atol=1e-10)


@pytest.mark.parametrize("key",list(catalogue())[1:])
def test_classic_all_16_catalogue_modules(key):
    s=catalogue()[key];params,diag,_=classic(s)
    assert diag["success"] and diag["cost"]<1e-6
    r=sdm_points([1000],[25],s,params)
    assert r["converged"].all()
    assert r["Pmp"][0]==pytest.approx(s.vmp_V*s.imp_A,abs=1e-5)
    assert r["Voc"][0]==pytest.approx(s.voc_V,abs=1e-6)
    assert r["Isc"][0]==pytest.approx(s.isc_A,abs=1e-6)


def test_supsi_multipoint_diagnostics(supsi_fit):
    params,diag,res=supsi_fit
    assert diag["success"] and diag["n_observations"]==66
    assert diag["jacobian_rank"]==5 and len(diag["starts"])==3
    assert all(s["success"] for s in diag["starts"])
    assert max(s["cost"] for s in diag["starts"])-min(s["cost"] for s in diag["starts"])<1e-6
    assert diag["residual_summary"]["Pmp"]["RMSE"]<2
    assert diag["spec"]["n_cells"] is None
    assert len(res)==66


@pytest.mark.parametrize("g,t",[(.5,25),(100,-10),(100,75),(600,40),(1000,25),(1100,75),(1500,90)])
def test_sdm_against_independent_implicit_root(g,t,supsi_fit):
    p,_,_=supsi_fit;s=supsi_spec()
    out=predict("M4",[g],[t],[t],s,p)
    assert out.converged[0]
    assert out.power_W[0]==pytest.approx(scalar_reference(g,t,s,p),rel=1e-8,abs=1e-6)


def test_solver_failure_and_array_electric_scaling(supsi_fit):
    p,_,_=supsi_fit;s=supsi_spec()
    fail=predict("M4",[1000],[25],[25],s,p,maxiter=1)
    assert not fail.converged[0] and np.isnan(fail.power_W[0])
    assert fail.status[0]=="solver_not_converged"
    v,i,power=iv_curve(1000,25,s,p)
    vv,ii,pp=iv_curve(1000,25,s,p,3,2)
    np.testing.assert_allclose(vv,v*3);np.testing.assert_allclose(ii,i*2);np.testing.assert_allclose(pp,power*6)


def test_unavailable_and_invalid_physics():
    s=supsi_spec()
    assert availability("M3",replace(s,technology="CdTe"))
    assert availability("M4",s)
    assert availability("M2",replace(s,gamma_pct_C=None))
    with pytest.raises(ValueError):predict("M1",[-1],[25],[25],s)
    with pytest.raises(ValueError):classic(s)
    with pytest.raises(ValueError):replace(s,p_stc_W=1000).validate()


def test_flags_preserve_duplicates_and_bad_data():
    raw=pd.DataFrame({"timestamp":pd.to_datetime(["2019-01-01 10:00","2019-01-01 10:00","2019-01-01 10:05","2019-01-01 10:10"]),
        "Pm":[20,20,-1,np.nan],"Gpoa":[200,200,50,-1],"Tbom":[25,25,25,110]})
    q=quality_flags(raw,ExperimentConfig())
    assert len(q)==4 and not q.eligible.any()
    assert "duplicate" in q.exclusion_reasons.iloc[0]
    assert "negative_Pm" in q.exclusion_reasons.iloc[2]
    assert "nonfinite_Pm" in q.exclusion_reasons.iloc[3]


def test_energy_no_gap_or_invalid_bridge():
    frame=pd.DataFrame({"timestamp":pd.to_datetime(["2019-01-01 10:00","2019-01-01 10:05","2019-01-01 10:10","2019-01-01 10:15","2019-01-02 10:00"]),"P":[100,100,100,100,100]})
    e,h,cum=energy_integral(frame,"P",[True,True,False,True,True])
    assert h==pytest.approx(5/60) and e==pytest.approx(100*5/60/1000)
    assert cum[-1]==e


def test_common_support_and_day_bootstrap():
    frames=[]
    for day in range(1,8):
        times=pd.date_range(f"2019-01-{day:02} 10:00",periods=4,freq="5min")
        f=pd.DataFrame({"timestamp":times,"Pm":100.,"Gpoa":400.,"Tmod_C":25.,"eligible":True,"common_valid":True})
        for i,m in enumerate(MODELS):f[m]=100+(i+1)*day
        frames.append(f)
    p=pd.concat(frames,ignore_index=True)
    metrics=energy_metrics(p,285.0)
    assert metrics.N_common.eq(28).all()
    np.testing.assert_allclose(metrics.covered_hours,7*15/60,atol=1e-12)
    groups=grouped_statistics(p);a=day_bootstrap(groups,reps=200);b=day_bootstrap(groups,reps=200)
    pd.testing.assert_frame_equal(a,b)
    assert a.N_blocks.eq(7).all()
    assert (a[a.model=="M1-M2"].CI95_high<0).all()


def test_orders_pareto_and_config_guards():
    orders=[model_order(r,"rotate",2026) for r in range(11)]
    assert sum(len(o) for o in orders)==44
    assert orders[0]!=orders[1] and all(set(o)==set(MODELS) for o in orders)
    assert model_order(1,"random",7)==model_order(1,"random",7)
    assert pareto_mask([2,1,3,1],[1,2,3,2]).tolist()==[True,True,False,True]
    with pytest.raises(ValueError):ExperimentConfig(rounds=2)
    with pytest.raises(ValueError):ExperimentConfig(g_min_Wm2=-1)


def test_ghi_not_renamed_and_arbitrary_horizon():
    raw=pd.DataFrame({"t":pd.date_range("2019-01-01",periods=241,freq="min").strftime("%Y-%m-%d %H:%M:%S"),"g":800.,"tmod":25.})
    kwargs=dict(timestamp_col="t",irradiance_col="g",temperature_col="tmod",timestamp_format="%Y-%m-%d %H:%M:%S",temperature_kind="Tmod",spec=supsi_spec())
    with pytest.raises(ValueError):custom_profile(raw,irradiance_kind="GHI",**kwargs)
    p=custom_profile(raw,irradiance_kind="POA",**kwargs)
    assert len(p)==241 and p.timestamp.iloc[-1]-p.timestamp.iloc[0]==pd.Timedelta(hours=4)
