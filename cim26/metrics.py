"""Common-support physical metrics and explicitly paired bootstrap summaries."""
from itertools import combinations
import numpy as np
import pandas as pd
from .schema import MODELS


def interval_support(frame, valid, max_gap_minutes):
    t=pd.to_datetime(frame.timestamp)
    dt=t.diff().dt.total_seconds().to_numpy(float)/3600
    v=np.asarray(valid,bool)
    previous=np.r_[False,v[:-1]]
    good=v & previous & np.isfinite(dt) & (dt>0) & (dt<=max_gap_minutes/60)
    return good,np.where(good,dt,0.0)


def energy_integral(frame, column, valid, max_gap_minutes=7.5):
    y=frame[column].to_numpy(float)
    good,dt=interval_support(frame,np.asarray(valid)&np.isfinite(y),max_gap_minutes)
    prev=np.r_[np.nan,y[:-1]]
    increments=np.zeros(len(y))
    increments[good]=(prev[good]+y[good])*0.5*dt[good]/1000
    return float(increments.sum()),float(dt.sum()),np.cumsum(increments)


def energy_metrics(predictions, p_stc, max_gap_minutes=7.5):
    p=predictions
    mask=p.common_valid.to_numpy(bool)
    obs=p.loc[mask,"Pm"].to_numpy(float)
    eobs,hours,_=energy_integral(p,"Pm",mask,max_gap_minutes)
    rows=[]
    for model in MODELS:
        y=p.loc[mask,model].to_numpy(float)
        r=y-obs
        finite=p.eligible & np.isfinite(p[model])
        est,_,_=energy_integral(p,model,mask,max_gap_minutes)
        n=len(r)
        rows.append(dict(model=model,N_common=n,N_eligible=int(p.eligible.sum()),N_total=len(p),
            N_model_valid=int(finite.sum()),coverage_of_eligible=float(finite.sum()/max(1,p.eligible.sum())),
            common_coverage_of_total=float(n/max(1,len(p))),
            RMSE_W=float(np.sqrt(np.mean(r*r))) if n else np.nan,
            MAE_W=float(np.mean(np.abs(r))) if n else np.nan,
            MBE_W=float(np.mean(r)) if n else np.nan,
            NRMSE_pct_Pstc=float(100*np.sqrt(np.mean(r*r))/p_stc) if n else np.nan,
            E_observed_kWh=eobs if hours>0 else np.nan,E_predicted_kWh=est if hours>0 else np.nan,E_error_kWh=est-eobs if hours>0 else np.nan,
            E_error_pct=float((est-eobs)/eobs*100) if eobs>0 else np.nan,
            covered_hours=hours,energy_kind="partial_common_adjacent_intervals",unit="W; kWh; h"))
    return pd.DataFrame(rows)


def grouped_statistics(predictions,max_gap_minutes=7.5):
    p=predictions.copy()
    t=pd.to_datetime(p.timestamp)
    dimensions={"day":t.dt.strftime("%Y-%m-%d"),"month":t.dt.strftime("%Y-%m"),
        "G_bin_Wm2":pd.cut(p.Gpoa,[0,100,200,400,600,800,1000,1200,1600],include_lowest=True).astype(str),
        "Tmod_bin_C":pd.cut(p.Tmod_C,[-100,0,15,25,40,55,75,120],include_lowest=True).astype(str)}
    rows=[]
    for dimension,labels in dimensions.items():
        for label in sorted(labels.dropna().unique()):
            mask=p.common_valid.to_numpy(bool)&(labels==label).to_numpy()
            if not mask.any():
                continue
            for model in MODELS:
                r=(p.loc[mask,model]-p.loc[mask,"Pm"]).to_numpy(float)
                row=dict(kind="group",dimension=dimension,group=str(label),model=model,
                    N=len(r),SSE=float(r@r),SAE=float(np.abs(r).sum()),SE=float(r.sum()),
                    RMSE_W=float(np.sqrt(np.mean(r*r))),MAE_W=float(np.mean(abs(r))),MBE_W=float(np.mean(r)))
                if dimension in ("day","month"):
                    eo,h,_=energy_integral(p,"Pm",mask,max_gap_minutes)
                    ep,_,_=energy_integral(p,model,mask,max_gap_minutes)
                    row.update(E_observed_kWh=eo,E_predicted_kWh=ep,covered_hours=h)
                rows.append(row)
    return pd.DataFrame(rows,columns=list(rows[0]) if rows else ["kind","dimension","group","model","N","SSE","SAE","SE"])


def day_bootstrap(groups,seed=2026,reps=1000):
    daily=groups[groups.dimension=="day"]
    days=sorted(daily.group.unique())
    if len(days)<5:
        return pd.DataFrame([dict(kind="bootstrap_unavailable",dimension="day",group="fewer_than_5_days")])
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(days),(reps,len(days)))
    draws={};rows=[]
    for model in MODELS:
        d=daily[daily.model==model].set_index("group").loc[days]
        n=d.N.to_numpy()[idx].sum(axis=1)
        rmse=np.sqrt(d.SSE.to_numpy()[idx].sum(axis=1)/n)
        mae=d.SAE.to_numpy()[idx].sum(axis=1)/n
        mbe=d.SE.to_numpy()[idx].sum(axis=1)/n
        eo=d.E_observed_kWh.to_numpy()[idx].sum(axis=1)
        ep=d.E_predicted_kWh.to_numpy()[idx].sum(axis=1)
        draws[model]=rmse
        energy_pct=np.full(len(eo),np.nan)
        np.divide(100*(ep-eo),eo,out=energy_pct,where=eo>0)
        for name,values in [("RMSE_W",rmse),("MAE_W",mae),("MBE_W",mbe),("E_error_pct",energy_pct)]:
            finite=values[np.isfinite(values)]
            low,high=np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
            rows.append(dict(kind="bootstrap",dimension="whole_day",group=name,model=model,
                CI95_low=float(low),CI95_high=float(high),N_blocks=len(days),bootstrap_reps=reps,
                method="paired_day_percentile",seed=seed))
    for a,b in combinations(MODELS,2):
        low,high=np.quantile(draws[a]-draws[b],[.025,.975])
        rows.append(dict(kind="paired_difference",dimension="whole_day",group="RMSE_A_minus_B_W",
            model=f"{a}-{b}",CI95_low=float(low),CI95_high=float(high),N_blocks=len(days),
            bootstrap_reps=reps,method="paired_day_percentile",seed=seed))
    return pd.DataFrame(rows)


def benchmark_summary(records,seed=2026,reps=1000):
    rows=[]
    for model,d in records.groupby("model",sort=False):
        good=d[(d.exit_code==0)&d.error.fillna("").eq("")]
        x=good.compute_s.dropna().to_numpy(float)
        row=dict(model=model,attempted=len(d),successful=len(x),failed=len(d)-len(x),
            median_s=np.nan,median_us_per_sample=np.nan,median_RSS_peak_MiB=np.nan,
            median_RSS_delta_MiB=np.nan,deterministic=False)
        if len(x):
            rng=np.random.default_rng(seed)
            ci=np.quantile(np.median(x[rng.integers(0,len(x),(reps,len(x)))],axis=1),[.025,.975])
            row.update(mean_s=float(x.mean()),median_s=float(np.median(x)),sd_s=float(x.std(ddof=1)) if len(x)>1 else np.nan,
                CV=float(x.std(ddof=1)/x.mean()) if len(x)>1 and x.mean()>0 else np.nan,
                min_s=float(x.min()),max_s=float(x.max()),
                median_us_per_sample=float(good.us_per_sample.median()),
                median_RSS_peak_MiB=float(good.rss_peak_MiB.median()),
                median_RSS_delta_MiB=float(good.rss_delta_MiB.median()),
                CI95_median_low_s=float(ci[0]),CI95_median_high_s=float(ci[1]),
                method="round_percentile_bootstrap",seed=seed,
                nonconverged=int(good.nonconverged.sum()),
                deterministic=bool(good.deterministic.all()))
            row.update({f"p{q}_s":float(np.percentile(x,q)) for q in [5,25,75,95]})
        rows.append(row)
    return pd.DataFrame(rows)


def time_pair_bootstrap(records,seed=2026,reps=1000):
    good=records[(records.exit_code==0)&records.error.fillna("").eq("")]
    wide=good.pivot(index="round",columns="model",values="compute_s").reindex(columns=MODELS).dropna()
    if len(wide)<3:
        return pd.DataFrame()
    rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(wide),(reps,len(wide)))
    med={m:np.median(wide[m].to_numpy()[ix],axis=1) for m in MODELS}
    rows=[]
    for a,b in combinations(MODELS,2):
        lo,hi=np.quantile(med[a]/med[b],[.025,.975])
        rows.append(dict(kind="paired_time_ratio",dimension="round",group="median_time_A_div_B",
            model=f"{a}/{b}",CI95_low=lo,CI95_high=hi,N_blocks=len(wide),bootstrap_reps=reps,
            method="paired_round_percentile",seed=seed))
    return pd.DataFrame(rows)


def pareto_mask(error,cost):
    e=np.asarray(error,float);c=np.asarray(cost,float)
    valid=np.isfinite(e)&np.isfinite(c)
    result=np.zeros(len(e),bool)
    for i in np.flatnonzero(valid):
        dominated=valid&(e<=e[i])&(c<=c[i])&((e<e[i])|(c<c[i]))
        result[i]=not dominated.any()
    return result


def integrated_metrics(energy,benchmark):
    out=energy.merge(benchmark,on="model",how="left")
    # Failed/incomplete benchmarks cannot claim a Pareto position.
    complete=out.successful.eq(out.attempted) & out.deterministic.fillna(False)
    out["pareto_time"]=pareto_mask(out.RMSE_W.where(complete),out.median_us_per_sample.where(complete))
    out["pareto_memory"]=pareto_mask(out.RMSE_W.where(complete),out.median_RSS_peak_MiB.where(complete))
    base=out[out.model=="M1"].iloc[0]
    out["RMSE_improvement_vs_M1_W"]=base.RMSE_W-out.RMSE_W
    out["time_ratio_vs_M1"]=out.median_us_per_sample/base.median_us_per_sample
    extra=out.median_us_per_sample-base.median_us_per_sample
    out["RMSE_gain_W_per_extra_us"]=out.RMSE_improvement_vs_M1_W/extra.where(extra>0)
    return out
