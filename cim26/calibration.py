"""Identification uses laboratory or manufacturer data only, never field CSVs."""
from dataclasses import asdict
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from .schema import EffectiveSDM
from .physics import sdm_points


def classic(spec):
    from .legacy.models.pv_module import ModuleSTC
    from .legacy.simulation.solver import extract_sdm_params
    if spec.technology != "c-Si":
        raise ValueError("O calibrador herdado usa propriedades do silício; esta versão suporta apenas c-Si no SDM.")
    required = ("isc_A","voc_V","vmp_V","imp_A","n_cells","alpha_pct_C","beta_pct_C")
    missing = [k for k in required if getattr(spec,k) is None]
    if missing:
        raise ValueError("Datasheet incompleto para SDM clássico: "+", ".join(missing))
    spec.validate()
    if spec.beta_pct_C == 0:
        raise ValueError("O calibrador clássico exige beta Voc diferente de zero.")
    stc = ModuleSTC(manufacturer=spec.manufacturer or "Não informado",model=spec.id,
        technology=spec.technology,p_nom=spec.p_stc_W,v_oc=spec.voc_V,i_sc=spec.isc_A,
        v_mp=spec.vmp_V,i_mp=spec.imp_A,n_cells=spec.n_cells,
        alpha_isc_pct=spec.alpha_pct_C,beta_voc_pct=spec.beta_pct_C,
        gamma_pmax_pct=spec.gamma_pct_C if spec.gamma_pct_C is not None else 0,
        eg_ref=spec.eg_ref_eV,degdt=spec.degdt_per_K)
    start=perf_counter()
    params,report=extract_sdm_params(stc)
    effective=EffectiveSDM(params.IL_ref,params.I0_ref,params.Rs,params.Rsh_ref,params.a_ref)
    diag=asdict(report)
    diag.update(method="classic_legacy_5_equations",calibration_s=perf_counter()-start,
        parameters=asdict(effective),spec=asdict(spec),n=params.n,n_cells=params.n_cells,
        source=spec.source,normalization="Isc, Imp, abs(beta Voc)",
        assumptions=["Beta Voc secante de 25 a 35 °C, conforme calibrador original.",
                     "P nominal pode diferir de Vmp × Imp devido ao arredondamento do datasheet."])
    return effective,diag,pd.DataFrame()


def multipoint(spec, lab):
    spec.validate()
    counts=lab.groupby("variable").size().to_dict()
    if counts != {"Isc":22,"Pmp":22,"Voc":22}:
        raise ValueError("SUPSI requer 22 medições de Pmp, Isc e Voc.")
    if lab.duplicated(["variable","irradiance_poa_Wm2","temperature_module_C"]).any():
        raise ValueError("Observações de laboratório duplicadas.")
    wide=lab.pivot(index=["irradiance_poa_Wm2","temperature_module_C"],columns="variable",values="measured_value")
    if len(wide)!=22 or wide.isna().any().any():
        raise ValueError("As 3 grandezas devem compartilhar os mesmos 22 pares G–T.")
    g=wide.index.get_level_values(0).to_numpy(float)
    tc=wide.index.get_level_values(1).to_numpy(float)
    scales={"Pmp":spec.p_stc_W,"Isc":spec.isc_A,"Voc":spec.voc_V}
    names=["Pmp","Isc","Voc"]
    def unpack(x):
        return EffectiveSDM(float(x[0]),float(10**x[1]),float(x[2]),float(10**x[3]),float(x[4]))
    def residual(x):
        r=sdm_points(g,tc,spec,unpack(x),tol=1e-8)
        if not r["converged"].all():
            return np.full(66,1e3)
        return np.concatenate([(r[k]-wide[k].to_numpy())/scales[k] for k in names])
    lo=np.array([8,-15,1e-4,1.5,0.8]); hi=np.array([12,-3,3,5.5,3.4])
    starts=[[9.5,-9,.4,2.7,1.8],[9.6,-10,.6,3.3,2.1],[9.7,-8,.2,2,1.2]]
    start=perf_counter()
    solutions=[least_squares(residual,x,bounds=(lo,hi),max_nfev=500,method="trf",
        xtol=1e-10,ftol=1e-10,gtol=1e-10,diff_step=1e-5) for x in starts]
    successful=[s for s in solutions if s.success and np.all(np.isfinite(s.fun))]
    best=min(successful or solutions,key=lambda s:float(s.fun@s.fun))
    parameters=unpack(best.x)
    r=sdm_points(g,tc,spec,parameters,tol=1e-8)
    rows=[]
    for k in names:
        for j in range(22):
            obs=float(wide[k].iloc[j]); pred=float(r[k][j])
            rows.append(dict(variable=k,Gpoa_Wm2=g[j],temperature_reported_C=tc[j],observed=obs,
                predicted=pred,residual=pred-obs,normalized_residual=(pred-obs)/scales[k]))
    residuals=pd.DataFrame(rows)
    singular=np.linalg.svd(best.jac,compute_uv=False)
    # Local sensitivity in optimization coordinates, NOT a physical confidence interval.
    diag=dict(method="SUPSI_22x3_multistart",success=bool(best.success and r["converged"].all()),
        message=str(best.message),calibration_s=perf_counter()-start,parameters=asdict(parameters),
        spec=asdict(spec),n_observations=66,n_conditions=22,n_parameters=5,
        cost=float(best.cost),n_iter=int(best.nfev),
        bounds={"coordinates":["IL_A","log10_I0_A","Rs_ohm","log10_Rsh_ohm","a_ref_V"],
                "lower":lo.tolist(),"upper":hi.tolist()},
        active_bounds=best.active_mask.tolist(),jacobian_rank=int(np.linalg.matrix_rank(best.jac)),
        jacobian_singular_values=singular.tolist(),
        jacobian_condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None,
        jacobian_coordinate_note="Condição local depende da escala dos parâmetros; não prova identificabilidade global.",
        starts=[dict(initial=x,solution=s.x.tolist(),cost=float(s.cost),success=bool(s.success),
                     nfev=int(s.nfev),message=str(s.message)) for x,s in zip(starts,solutions)],
        residual_summary={k:dict(RMSE=float(np.sqrt(np.mean(group.residual**2))),
            MAE=float(np.mean(np.abs(group.residual))),max_abs=float(np.max(np.abs(group.residual))))
            for k,group in residuals.groupby("variable")},
        assumptions=["Temperatura reportada em laboratório usada como Tcell (equilíbrio do ensaio flash); sem correção solar de campo.",
            "Ns, n individual, Vmp/Imp medidos e identidade comercial desconhecidos.",
            "Eg=1.121 eV e dEg/dT relativo=-0.0002677/K herdados para silício.",
            "Resíduos normalizados por STC, sem ponderação por incertezas instrumentais.",
            "Convergência multiarrranque e Jacobiano local não demonstram unicidade global."])
    return parameters,diag,residuals
