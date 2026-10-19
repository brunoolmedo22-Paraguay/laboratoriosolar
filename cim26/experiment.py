"""Frozen, sequential and append-only experiments with replayable CSV outputs."""
from dataclasses import asdict,replace
from datetime import datetime,timezone
from pathlib import Path
from time import perf_counter,perf_counter_ns
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
import hashlib
import numpy as np
import pandas as pd
import psutil
from .schema import ROOT,INPUT,MODELS,HULD,ExperimentConfig,supsi_spec,dump_json
from . import __version__
from .data import read_field,field_paths,quality_flags,prepared_conditions,laboratory,sha256
from .calibration import multipoint
from .physics import predict,scalar_reference
from .metrics import energy_metrics,grouped_statistics,day_bootstrap,benchmark_summary,time_pair_bootstrap,integrated_metrics
from .report import generate_figures,manifest


def environment():
    cpu=platform.processor()
    if not cpu and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):cpu=line.partition(":")[2].strip();break
    try:freq=psutil.cpu_freq();frequency=freq.current if freq else None
    except Exception:frequency=None
    deps=["numpy","scipy","pandas","streamlit","plotly","matplotlib","psutil","openpyxl"]
    return dict(cpu=cpu or "not measured",logical_cpus=psutil.cpu_count(),
        physical_cpus=psutil.cpu_count(logical=False),ram_total_GiB=psutil.virtual_memory().total/2**30,
        os=platform.platform(),machine=platform.machine(),python=sys.version,
        versions={x:importlib.metadata.version(x) for x in deps},
        cpu_frequency_snapshot_MHz=frequency,cpu_temperature="not measured",
        cpu_energy="not measured",app_version=__version__,
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/"cim26").rglob("*.py"))})


def model_order(round_index,order,seed):
    if order=="rotate":
        offset=round_index%4
        return MODELS[offset:]+MODELS[:offset]
    return tuple(np.random.default_rng(seed+round_index).permutation(MODELS).tolist())


def acquire_experiment_lock(path):
    """OS advisory lock: automatically released on process exit, including crashes."""
    handle=open(path,"a+b")
    if handle.tell()==0:handle.write(b"0");handle.flush()
    handle.seek(0)
    try:
        if os.name=="nt":
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("Já existe um experimento ativo. Aguarde sua conclusão.")
    return handle


def release_experiment_lock(handle):
    handle.seek(0)
    if os.name=="nt":
        import msvcrt
        msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
    handle.close()


def isolated_evaluation(model,arrays,spec,params,config,workdir,tag,inner_repeats=None):
    workdir=Path(workdir)
    input_path=workdir/(tag+"_input.npz")
    np.savez(input_path,**arrays)
    output=workdir/(tag+"_output.npz");result=workdir/(tag+"_result.json")
    job=dict(model=model,spec=asdict(spec),parameters=asdict(params),inputs=str(input_path),
        output=str(output),result=str(result),tolerance=config.mpp_tolerance_V,
        max_iterations=config.max_iterations,rss_interval_s=config.rss_interval_s,
        target_batch_s=config.target_batch_s,inner_repeats=inner_repeats)
    job_path=workdir/(tag+"_job.json");dump_json(job_path,job)
    env=os.environ.copy()
    for name in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
        env[name]=str(config.threads)
    env["PYTHONHASHSEED"]=str(config.seed)
    env["PYTHONPATH"]=str(ROOT)
    start=perf_counter()
    try:
        completed=subprocess.run([sys.executable,"-m","cim26.worker",str(job_path)],cwd=ROOT,
            env=env,capture_output=True,text=True,timeout=600)
        exitcode=completed.returncode
        error=completed.stderr[-3000:] if exitcode else ""
    except subprocess.TimeoutExpired:
        exitcode=-1;error="Timeout de 600 s; subprocesso encerrado."
    row=dict(compute_s=np.nan,us_per_sample=np.nan,start_ns=np.nan,end_ns=np.nan,
        rss_baseline_MiB=np.nan,rss_peak_MiB=np.nan,rss_delta_MiB=np.nan,nonconverged=len(arrays["g"]),
        valid_predictions=0,deterministic=False,n_samples=len(arrays["g"]),error=error,exit_code=exitcode)
    if result.exists():row.update(json.loads(result.read_text()))
    row["exit_code"]=exitcode
    if exitcode and not row.get("error"):row["error"]=error or "Worker falhou sem diagnóstico."
    row["process_wall_s"]=perf_counter()-start
    row["model"]=model
    return row,output if output.exists() else None


def run_experiment(config=ExperimentConfig(),root=None,sources=None,progress=None):
    root=Path(root or ROOT/"runs");root.mkdir(parents=True,exist_ok=True)
    # Global to this project, including runs in custom directories or multiple UI sessions.
    lock=ROOT/".experiment.lock"
    lock_handle=acquire_experiment_lock(lock)
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:8]
    path=root/run_id;path.mkdir()
    meta=dict(run_id=run_id,status="running",created_utc=datetime.now(timezone.utc).isoformat(),
        diagnostic="Engineering diagnostic, not final scientific conclusions")
    dump_json(path/"metadata.json",meta);dump_json(path/"config.json",asdict(config))
    start=perf_counter()
    try:
        spec=supsi_spec()
        source_paths=[Path(p) for p in (sources or field_paths(config.months))]
        if len({p.name for p in source_paths}) != len(source_paths):
            raise ValueError("Nomes de arquivos duplicados.")
        folder=path/"inputs";folder.mkdir()
        for source in source_paths:shutil.copy2(source,folder/source.name)
        lab=laboratory()
        lab.to_csv(folder/"laboratory.csv",index=False)
        shutil.copy2(INPUT/"SUPSI/raw/laboratory/IEC61853_1-2.xlsx",folder/"IEC61853_1-2.xlsx")
        dump_json(folder/"module_spec.json",asdict(spec))
        q=quality_flags(read_field([folder/p.name for p in source_paths]),config)
        q.to_csv(path/"input_quality.csv",index=False)
        selected,arrays=prepared_conditions(q,config,spec)
        payload=b"".join(arrays[k].astype("<f8").tobytes() for k in ["g","tm","tc"])
        payload+=selected.timestamp.astype("int64").to_numpy().astype("<i8").tobytes()
        prepared_hash=hashlib.sha256(payload).hexdigest()
        prepare_s=perf_counter()-start
        if progress:progress(0,"Identificando SDM somente com as 66 observações de laboratório…")
        params,calibration,residuals=multipoint(spec,lab)
        dump_json(path/"calibration.json",calibration)
        residuals.to_csv(path/"calibration_residuals.csv",index=False)
        if not calibration["success"]:raise RuntimeError("Calibração SDM não convergiu; veja calibration.json.")
        metadata_env=environment()
        freeze=subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True)
        (path/"environment_freeze.txt").write_text(freeze.stdout,encoding="utf-8")
        if freeze.returncode:meta["pip_freeze_error"]=freeze.stderr[-1000:]
        # Check a fixed G/T grid before measuring, with independent implicit root equations.
        oracle=[]
        for g,t in [(100,15),(200,75),(600,25),(1000,25),(1100,75)]:
            actual=predict("M4",[g],[t],[t],spec,params,tol=config.mpp_tolerance_V,maxiter=config.max_iterations).power_W[0]
            ref=scalar_reference(g,t,spec,params)
            oracle.append(dict(G=g,T=t,delta_W=float(actual-ref)))
        meta.update(environment=metadata_env,prepare_s=prepare_s,calibration_s=calibration["calibration_s"],
            input_sha256={p.name:sha256(p) for p in folder.iterdir()},input_files=[p.name for p in source_paths],
            laboratory_temperature_assumption=calibration["assumptions"][0],algorithmic_reference=oracle,
            huld=dict(family="CSi CURRENT",coefficients=HULD,source="JRC PVGIS data-sources-calculation-methods; checked 2026-10-19"),
            timestamp_timezone="Not specified by source. Original clock retained; no UTC/DST conversion.",
            implementation="All models batch-vectorized. M4: legacy Lambert equation, Wright Omega evaluation, bounded golden search.",
            memory_policy="Fresh process per model/round; sampled RSS baseline after warmup; absolute and delta; 2 ms default sampling can miss peaks.",
            optical_policy="Measured Gpoa for all four; optical/spectral corrections not calibrated.",
            statistical_limits="11 software rounds are not physical replications. Day bootstrap assumes representative exchangeable days; serial/seasonal dependence remains.")
        dump_json(path/"metadata.json",meta)
        predictions=q.copy()
        predictions["Tmod_C"]=np.nan;predictions["Tcell_C"]=np.nan
        predictions.loc[selected.index,"Tmod_C"]=arrays["tm"]
        predictions.loc[selected.index,"Tcell_C"]=arrays["tc"]
        for m in MODELS:predictions[m]=np.nan;predictions[m+"_status"]="ineligible"
        first={};stable={m:np.ones(len(selected),bool) for m in MODELS};records=[];repeats={}
        (path/"round_outputs").mkdir()
        with tempfile.TemporaryDirectory(prefix="cim26_workers_") as temp:
            for round_index in range(config.rounds):
                for position,model in enumerate(model_order(round_index,config.order,config.seed)):
                    if progress:progress(len(records)/(config.rounds*4),f"Rodada {round_index+1}/{config.rounds} · {model} · posição {position+1}/4")
                    tag=f"r{round_index+1:03}_{position+1}_{model}"
                    row,output=isolated_evaluation(model,arrays,spec,params,config,temp,tag,repeats.get(model))
                    row.update(round=round_index+1,position=position+1,input_sha256=prepared_hash)
                    if output and row["exit_code"]==0:
                        r=np.load(output,allow_pickle=False)
                        if model not in first:
                            first[model]=r["power"].copy()
                            predictions.loc[selected.index,model]=r["power"]
                            predictions.loc[selected.index,model+"_status"]=r["status"]
                        consistent=np.isclose(first[model],r["power"],rtol=1e-12,atol=1e-9,equal_nan=True)
                        row["deterministic"]=bool(row["deterministic"] and consistent.all())
                        stable[model] &= consistent & r["converged"]
                        repeats.setdefault(model,int(row["inner_repeats"]))
                        shutil.copy2(output,path/"round_outputs"/(tag+".npz"))
                    else:stable[model][:]=False
                    records.append(row)
                    pd.DataFrame(records).to_csv(path/"benchmark_runs.csv",index=False)
                    predictions.to_csv(path/"predictions_partial.csv",index=False)
            post_start=perf_counter()
            common=np.logical_and.reduce([stable[m] for m in MODELS])
            for m in MODELS:
                common &= np.isfinite(predictions.loc[selected.index,m].to_numpy())
                predictions[m+"_residual_W"]=predictions[m]-predictions.Pm
            predictions["common_valid"]=False
            predictions.loc[selected.index,"common_valid"]=common
            predictions.to_csv(path/"predictions.csv",index=False)
            energy=energy_metrics(predictions,spec.p_stc_W,config.max_gap_minutes)
            energy.to_csv(path/"metrics_energy.csv",index=False)
            rawbench=pd.DataFrame(records)
            bench=benchmark_summary(rawbench,config.seed,config.bootstrap_reps)
            bench.to_csv(path/"benchmark_summary.csv",index=False)
            groups=grouped_statistics(predictions,config.max_gap_minutes)
            stats=pd.concat([groups,day_bootstrap(groups,config.seed,config.bootstrap_reps),
                time_pair_bootstrap(rawbench,config.seed,config.bootstrap_reps)],ignore_index=True)
            stats.to_csv(path/"statistics.csv",index=False)
            integrated_metrics(energy,bench).to_csv(path/"integrated.csv",index=False)
            sens=[]
            for dt in config.thermal_sensitivity:
                _,aa=prepared_conditions(q,replace(config,delta_T_C=dt),spec)
                pp=predictions.copy()
                for m in MODELS:
                    pp.loc[selected.index,m]=predict(m,aa["g"],aa["tm"],aa["tc"],spec,params,
                        tol=config.mpp_tolerance_V,maxiter=config.max_iterations).power_W
                metric=energy_metrics(pp,spec.p_stc_W,config.max_gap_minutes)
                metric["delta_T_C"]=dt;sens.append(metric)
            (pd.concat(sens,ignore_index=True) if sens else pd.DataFrame(columns=["model","delta_T_C","RMSE_W"])).to_csv(path/"thermal_sensitivity.csv",index=False)
            scale=[]
            for n in sorted(set(min(n,len(selected)) for n in config.scalability_sizes)):
                if progress:progress(.98,f"Escalabilidade adicional · N={n} · separada das rodadas principais")
                aa={k:v[:n] for k,v in arrays.items()}
                for rr in range(3):
                    for pos,m in enumerate(model_order(rr,config.order,config.seed)):
                        row,_=isolated_evaluation(m,aa,spec,params,config,temp,f"scale{n}_{rr}_{m}")
                        row.update(round=rr+1,position=pos+1,batch_N=n,scope="scalability_prefix_3_rounds")
                        scale.append(row)
            pd.DataFrame(scale,columns=None if scale else ["model","batch_N","compute_s","us_per_sample"]).to_csv(path/"scalability.csv",index=False)
        meta.update(status="complete" if all(r["exit_code"]==0 and r["deterministic"] and r["nonconverged"]==0 for r in records) else "complete_with_failures",
            N_total=len(q),N_eligible=len(selected),N_common=int(common.sum()),
            outside_lab_rectangle_eligible=int(q.loc[q.eligible,"outside_lab_rectangle"].sum()),
            benchmark_records=len(records),max_active_model_workers=1,
            postprocess_s=perf_counter()-post_start,finished_utc=datetime.now(timezone.utc).isoformat())
        dump_json(path/"metadata.json",meta)
        figure_start=perf_counter();generate_figures(path);meta["figures_s"]=perf_counter()-figure_start
        meta["total_wall_s"]=perf_counter()-start
        dump_json(path/"metadata.json",meta)
        (path/"predictions_partial.csv").unlink(missing_ok=True)
        (path/"README_RUN.md").write_text(f"""# CIM26 execution {run_id}

Engineering diagnostic, not a scientific article or a universal model ranking.
Open the directory in Artigo > Histórico, or import its exported ZIP. No recalculation occurs.
To reproduce in the supplied application: `python -m cim26 run --config runs/{run_id}/config.json`.
This creates a NEW directory. It reads the snapshot CSV files in this run's inputs directory.
All four models use the same eligible timestamps. `common_valid` also requires convergence and
deterministic output across all rounds; metrics use that fixed intersection.
Regenerate figures: `python -m cim26 figures runs/{run_id}` (writes only figures).
Recompute metrics from CSV: `python -m cim26 verify runs/{run_id}` (read only).

Main benchmark: {config.rounds} × 4 = {len(records)} records; scalability is separate.
Each worker performs one untimed warmup. The first round freezes inner repetitions per model
to target {config.target_batch_s} s. `compute_s` is batch time / inner repetitions;
`us_per_sample` also divides by the common N. Loop and Python API costs are included.
RSS is sampled at {config.rss_interval_s} s in a sampler thread, with common overhead;
very brief peaks may be missed and tiny memory deltas may be zero. No CPU energy was measured.
The raw source clock is retained without assuming a timezone. Calendar days follow that clock.
Energy is partial, on adjacent valid pairs with gap <= {config.max_gap_minutes} min, never annualized.
Huld CSi CURRENT coefficients are frozen. DeltaT={config.delta_T_C} °C is an assumption.
The laboratory reported temperature is treated as cell temperature at thermal equilibrium.
No field measurement was used to calibrate parameters. Multistart does not prove global uniqueness.
Day/round percentile bootstrap CIs are exploratory: dependence, seasonality and 11 rounds limit inference.
Input hashes, code hashes, environment, parameter diagnostics and every raw round are preserved.
""",encoding="utf-8")
        manifest(path)
        if progress:progress(1,"Experimento salvo. Resultados prontos para inspeção.")
        return path
    except BaseException as exc:
        meta.update(status="failed",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
        dump_json(path/"metadata.json",meta)
        raise RuntimeError(f"Experimento interrompido; saídas parciais em {path}. {exc}") from exc
    finally:
        release_experiment_lock(lock_handle)
