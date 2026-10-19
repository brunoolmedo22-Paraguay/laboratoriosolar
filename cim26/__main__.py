"""CLI entry point; keep reproducibility independent of Streamlit."""
import os
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key,"1")
import argparse
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .schema import ROOT,ExperimentConfig,supsi_spec


def main():
    parser=argparse.ArgumentParser(description="CIM26 photovoltaic laboratory")
    sub=parser.add_subparsers(dest="command",required=True)
    run=sub.add_parser("run",help="Create a new immutable experiment")
    run.add_argument("--config",type=Path)
    run.add_argument("--months",nargs="+")
    run.add_argument("--rounds",type=int)
    run.add_argument("--no-scalability",action="store_true")
    run.add_argument("--output",type=Path,default=ROOT/"runs")
    fig=sub.add_parser("figures",help="Rebuild SVG/PNG from saved predictions")
    fig.add_argument("path",type=Path)
    verify=sub.add_parser("verify",help="Verify manifest and recompute physical metrics from CSV")
    verify.add_argument("path",type=Path)
    lab=sub.add_parser("extract-lab",help="Re-extract laboratory matrix without changing sources")
    lab.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="run":
        from .experiment import run_experiment
        cfg=ExperimentConfig(**json.loads(args.config.read_text())) if args.config else ExperimentConfig()
        if args.months:cfg=replace(cfg,months=tuple(args.months))
        if args.rounds is not None:cfg=replace(cfg,rounds=args.rounds)
        if args.no_scalability:cfg=replace(cfg,scalability_sizes=())
        sources=None
        if args.config and (args.config.parent/"metadata.json").exists():
            meta=json.loads((args.config.parent/"metadata.json").read_text())
            sources=[args.config.parent/"inputs"/f for f in meta["input_files"]]
        path=run_experiment(cfg,root=args.output,sources=sources,
            progress=lambda f,msg:print(f"{100*f:5.1f}% {msg}",flush=True))
        print(path)
    elif args.command=="figures":
        from .report import generate_figures,manifest
        generate_figures(args.path);manifest(args.path)
        print("Figuras regeneradas dos CSV; manifesto atualizado.")
    elif args.command=="verify":
        from .report import load_run
        from .data import sha256
        from .metrics import energy_metrics
        data=load_run(args.path)
        expected=json.loads((args.path/"manifest_sha256.json").read_text())
        for f,h in expected.items():
            if sha256(args.path/f)!=h:raise ValueError("Hash divergente: "+f)
        actual=energy_metrics(data["predictions"],supsi_spec().p_stc_W,data["config"]["max_gap_minutes"])
        cols=["RMSE_W","MAE_W","MBE_W","E_observed_kWh","E_predicted_kWh","covered_hours"]
        np.testing.assert_allclose(actual[cols],data["metrics_energy"][cols],atol=1e-10,rtol=1e-10,equal_nan=True)
        print(f"OK: {len(expected)} hashes e métricas físicas recalculadas dos CSV.")
    elif args.command=="extract-lab":
        from .data import laboratory_from_xlsx
        if args.output.exists():raise ValueError("Escolha um arquivo novo; não sobrescrevemos fontes.")
        laboratory_from_xlsx().to_csv(args.output,index=False)
        print(args.output)


if __name__=="__main__":main()
