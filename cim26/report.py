"""Portable exports. Figures are regenerated from saved CSVs only."""
from io import BytesIO
from pathlib import Path, PurePosixPath
import json
import zipfile
import hashlib
import numpy as np
import pandas as pd
from .schema import MODELS, LABELS, dump_json

COLORS={"Pm":"#12283b","M1":"#d49223","M2":"#168b85","M3":"#735bc1","M4":"#2373cc"}


def plot_arrays(frame,x_column,y_column,max_gap_minutes=None):
    """Visual separators only: never interpolate power across missing timestamps."""
    times=pd.to_datetime(frame[x_column])
    x=times.to_numpy(dtype="datetime64[ns]")
    y=frame[y_column].to_numpy(float)
    if max_gap_minutes is not None:
        gaps=np.flatnonzero(times.diff().dt.total_seconds().to_numpy()>max_gap_minutes*60)
        x=np.insert(x,gaps,np.datetime64("NaT"))
        y=np.insert(y,gaps,np.nan)
    return x,y


def load_run(path):
    path=Path(path)
    return {"path":path,
        "config":json.loads((path/"config.json").read_text()),
        "metadata":json.loads((path/"metadata.json").read_text()),
        **{name:pd.read_csv(path/(name+".csv")) for name in ["predictions","metrics_energy",
            "benchmark_runs","benchmark_summary","statistics","integrated","thermal_sensitivity","scalability"]}}


def run_zip(path):
    path=Path(path);buf=BytesIO()
    frozen=path/"manifest_sha256.json"
    names=sorted(json.loads(frozen.read_text()))+["manifest_sha256.json"] if frozen.exists() else [
        f.relative_to(path).as_posix() for f in sorted(path.rglob("*")) if f.is_file()]
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(path/name,arcname=name)
    return buf.getvalue()


def import_run_zip(data, parent):
    """Never execute uploaded code or unpickle content. Extract a bounded data archive."""
    parent=Path(parent)
    with zipfile.ZipFile(BytesIO(data)) as z:
        entries=z.infolist()
        if sum(x.file_size for x in entries)>500*1024**2 or len(entries)>2000:
            raise ValueError("ZIP excede o limite de 500 MiB / 2000 arquivos.")
        names={x.filename for x in entries}
        required={"config.json","metadata.json","predictions.csv","manifest_sha256.json"}
        if not required.issubset(names):
            raise ValueError("Selecione o ZIP de uma execução exportada pelo aplicativo.")
        for info in entries:
            p=PurePosixPath(info.filename)
            if p.is_absolute() or ".." in p.parts or "\\" in info.filename or ":" in info.filename:
                raise ValueError("Caminho inválido no ZIP.")
            if (info.external_attr>>16)&0o170000 == 0o120000:
                raise ValueError("Links simbólicos não permitidos.")
        manifest=json.loads(z.read("manifest_sha256.json"))
        for name,digest in manifest.items():
            if name not in names or hashlib.sha256(z.read(name)).hexdigest()!=digest:
                raise ValueError("Integridade inválida no arquivo "+name)
        if any(n not in manifest and n!="manifest_sha256.json" and not n.endswith("/") for n in names):
            raise ValueError("Há arquivos não declarados no manifesto.")
        rid="import_"+hashlib.sha256(data).hexdigest()[:12]
        target=parent/rid
        if target.exists():
            return target
        target.mkdir(parents=True)
        z.extractall(target)
    load_run(target)
    return target


def manifest(path):
    path=Path(path)
    dump_json(path/"manifest_sha256.json",{f.relative_to(path).as_posix():hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(path.rglob("*")) if f.is_file() and f.name!="manifest_sha256.json"})


def generate_figures(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .metrics import energy_integral
    p=Path(path)
    data=load_run(p)
    pred=data["predictions"];bench=data["benchmark_runs"];integrated=data["integrated"]
    folder=p/"figures";folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
        "axes.spines.right":False,"axes.grid":True,"grid.alpha":.16,"svg.fonttype":"none"})
    times=pd.to_datetime(pred.timestamp)
    def save(fig,name):
        fig.tight_layout()
        for ext in ["svg","png"]:fig.savefig(folder/f"{name}.{ext}",dpi=180,bbox_inches="tight",facecolor="white")
        plt.close(fig)
    day=times.loc[pred.common_valid].dt.date.iloc[0] if pred.common_valid.any() else times.dt.date.iloc[0]
    sel=times.dt.date==day
    fig,ax=plt.subplots(figsize=(9,4.2))
    for m in ("Pm",)+MODELS:
        x,y=plot_arrays(pred[sel],"timestamp",m,data["config"]["max_gap_minutes"])
        ax.plot(x,y,label="Medido" if m=="Pm" else m,color=COLORS[m],lw=1.6)
    ax.set(ylabel="Potência DC por módulo (W)",title=f"SUPSI · {day} · diagnóstico de engenharia")
    ax.legend(ncol=5,frameon=False);fig.autofmt_xdate();save(fig,"power_first_day")
    fig,ax=plt.subplots(figsize=(9,4.2))
    for m in ("Pm",)+MODELS:
        _,_,cum=energy_integral(pred,m,pred.common_valid,data["config"]["max_gap_minutes"])
        ax.plot(times,cum,label="Medido" if m=="Pm" else m,color=COLORS[m])
    ax.set(ylabel="Energia parcial acumulada (kWh)",title="Somente intervalos adjacentes válidos")
    ax.legend(ncol=5,frameon=False);save(fig,"energy_partial")
    fig,ax=plt.subplots(figsize=(7,4.2))
    arrays=[bench.loc[(bench.model==m)&bench.exit_code.eq(0),"us_per_sample"].dropna() for m in MODELS]
    ax.boxplot(arrays,tick_labels=MODELS,patch_artist=False)
    ax.set(yscale="log",ylabel="Tempo por amostra (µs, log)",title="Rodadas independentes de software")
    save(fig,"benchmark")
    for cost,flag,name,xlabel in [("median_us_per_sample","pareto_time","pareto_time","Mediana por amostra (µs)"),
        ("median_RSS_peak_MiB","pareto_memory","pareto_memory","Pico RSS amostrado (MiB)")]:
        fig,ax=plt.subplots(figsize=(7,4.2))
        for _,r in integrated.iterrows():
            if np.isfinite(r.get(cost,np.nan)) and np.isfinite(r.RMSE_W):
                ax.scatter(r[cost],r.RMSE_W,s=100,color=COLORS[r.model],marker="D" if r[flag] else "o")
                ax.annotate(r.model,(r[cost],r.RMSE_W),xytext=(7,5),textcoords="offset points")
        if cost=="median_us_per_sample":ax.set_xscale("log")
        ax.set(xlabel=xlabel,ylabel="RMSE (W)",title="Exatidão × custo · losangos: não dominados")
        save(fig,name)


def simulation_zip(frame, metadata, iv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    buf=BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("simulation.csv",frame.to_csv(index=False))
        z.writestr("config.json",json.dumps(metadata,ensure_ascii=False,indent=2))
        fig,ax=plt.subplots(figsize=(9,4))
        x,y=plot_arrays(frame,"timestamp","P_array_W",metadata["max_gap_minutes"])
        ax.plot(x,y,color="#168b85")
        ax.set(ylabel="Potência DC do arranjo (W)",xlabel="Tempo",title="Simulação fotovoltaica")
        fig.autofmt_xdate();fig.tight_layout()
        for ext in ["svg","png"]:
            out=BytesIO();fig.savefig(out,format=ext,dpi=180);z.writestr("power."+ext,out.getvalue())
        plt.close(fig)
        if iv is not None:z.writestr("iv_pv.csv",iv.to_csv(index=False))
    return buf.getvalue()
