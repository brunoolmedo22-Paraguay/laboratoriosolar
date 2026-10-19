"""DC module physics. Batch vectorization for all four models; no Streamlit."""
from dataclasses import dataclass
import numpy as np
from scipy.special import wrightomega
from scipy.optimize import brentq, minimize_scalar
from .schema import ModuleSpec, EffectiveSDM, HULD

K_EV = 1.380649e-23 / 1.602176634e-19
TREF = 298.15


@dataclass
class Prediction:
    power_W: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    status: np.ndarray


def thermal_conditions(g, t_back=None, t_ambient=None, wind=None, *, method="measured_back",
                       delta_T_C=3.0, spec=None, t_cell=None):
    g = np.asarray(g, dtype=float)
    if method == "measured_back":
        if t_back is None:
            raise ValueError("Temperatura posterior medida não fornecida.")
        tm = np.broadcast_to(np.asarray(t_back, float), g.shape).copy()
        tc = tm + g / 1000 * delta_T_C
    elif method == "measured_cell":
        if t_cell is None:
            raise ValueError("Temperatura de célula não fornecida.")
        tc = np.broadcast_to(np.asarray(t_cell, float), g.shape).copy()
        tm = tc - g / 1000 * delta_T_C
    elif method == "faiman":
        if spec is None or spec.u0 is None or spec.u1 is None or t_ambient is None or wind is None:
            raise ValueError("Faiman requer U0, U1, Tamb e vento.")
        if np.any(np.asarray(wind) < 0):
            raise ValueError("Velocidade de vento negativa.")
        tm = np.asarray(t_ambient) + g / (spec.u0 + spec.u1 * np.asarray(wind))
        tc = tm + g / 1000 * delta_T_C
    elif method == "noct":
        if spec is None or spec.noct_C is None or t_ambient is None:
            raise ValueError("NOCT requer Tamb e NOCT/NMOT declarado.")
        tc = np.asarray(t_ambient) + (spec.noct_C - 20) / 800 * g
        tm = tc - g / 1000 * delta_T_C
    else:
        raise ValueError("Método térmico desconhecido.")
    return np.asarray(tm, float), np.asarray(tc, float)


def operating(params, spec, g, tc):
    """De Soto/CEC translation retained from the supplied legacy solver."""
    tk = np.asarray(tc, float) + 273.15
    g = np.asarray(g, float)
    a = params.a_ref * tk / TREF
    il = g / 1000 * (params.IL_ref + spec.alpha_A_C * (tk - TREF))
    eg = spec.eg_ref_eV * (1 + spec.degdt_per_K * (tk - TREF))
    i0 = params.I0_ref * (tk / TREF)**3 * np.exp(spec.eg_ref_eV / (K_EV*TREF) - eg / (K_EV*tk))
    return il, i0, params.Rs, params.Rsh_ref * 1000 / g, a


def current(v, op):
    """Legacy Lambert W equation; Omega(log z) = W(exp(log z)), overflow safe."""
    il, i0, rs, rsh, a = op
    denom = a * (rs + rsh)
    lz = np.log(rs * i0 * rsh / denom) + rsh * (rs * (il+i0) + v) / denom
    return (rsh*(il+i0)-v)/(rs+rsh) - a/rs * wrightomega(lz)


def sdm_points(g, tc, spec, params, tol=1e-6, maxiter=100):
    """Vector bounded golden-section P(V) maximum; bisection Voc.

    Uniform algorithm for every positive-G sample. No fallback is hidden.
    Brackets, finiteness, optimizer termination and implicit residual are checked.
    """
    g, tc = np.broadcast_arrays(np.atleast_1d(g).astype(float), np.atleast_1d(tc).astype(float))
    op = operating(params, spec, g, tc)
    il, i0, rs, rsh, a = op
    lo = np.zeros_like(g)
    hi = a * np.log1p(il / i0) * 1.05
    valid = (il > 0) & np.isfinite(hi) & (current(lo, op) > 0) & (current(hi, op) <= 0)
    for _ in range(60):
        mid = (lo + hi)/2
        positive = current(mid, op) > 0
        lo, hi = np.where(positive, mid, lo), np.where(positive, hi, mid)
        if np.all(hi-lo <= 1e-9):
            break
    voc = (lo+hi)/2
    valid &= hi-lo <= 1e-8
    left, right = np.zeros_like(g), voc.copy()
    ratio = (np.sqrt(5)-1)/2
    x1, x2 = right-ratio*(right-left), left+ratio*(right-left)
    f1, f2 = x1*current(x1, op), x2*current(x2, op)
    iterations = np.zeros(len(g), dtype=np.int32)
    for _ in range(maxiter):
        active = right-left > tol
        if not np.any(active):
            break
        iterations += active
        choose = f1 < f2
        left = np.where(active & choose, x1, left)
        right = np.where(active & ~choose, x2, right)
        x1 = right-ratio*(right-left)
        x2 = left+ratio*(right-left)
        f1, f2 = x1*current(x1, op), x2*current(x2, op)
    vmp = (left+right)/2
    imp = current(vmp, op)
    pmp = vmp*imp
    residual = il-i0*np.expm1(np.minimum((vmp+imp*rs)/a, 700))-(vmp+imp*rs)/rsh-imp
    valid &= (right-left <= tol) & np.isfinite(pmp) & (pmp >= 0) & (np.abs(residual) < 1e-7*np.maximum(il,1))
    return dict(Pmp=pmp, Vmp=vmp, Imp=imp, Voc=voc, Isc=current(np.zeros_like(g), op),
                converged=valid, iterations=iterations, residual_A=residual)


def scalar_reference(g, tc, spec, params):
    """Independent algorithmic check: implicit I root + scalar bounded MPP."""
    op = operating(params, spec, np.float64(g), np.float64(tc))
    il, i0, rs, rsh, a = op
    def iv(v):
        return brentq(lambda i: il-i0*np.expm1(min((v+i*rs)/a, 700))-(v+i*rs)/rsh-i,
                      -il-5, il+5, xtol=1e-11)
    voc = brentq(iv, 0, a*np.log1p(il/i0)*1.05, xtol=1e-10)
    result = minimize_scalar(lambda v: -v*iv(v), bounds=(0,voc), method="bounded",
                             options={"xatol":1e-8})
    if not result.success:
        raise RuntimeError(result.message)
    return float(-result.fun)


def availability(model, spec, params=None):
    if model == "M2" and spec.gamma_pct_C is None:
        return "Falta gamma de potência (%/°C)."
    if model == "M3" and spec.technology != "c-Si":
        return "Coeficientes Huld disponíveis apenas para c-Si CURRENT."
    if model == "M4" and spec.technology != "c-Si":
        return "SDM desta versão usa propriedades do silício e suporta apenas c-Si."
    if model == "M4" and params is None:
        return "Estime os parâmetros SDM primeiro."
    return ""


def predict(model, g_poa, t_module, t_cell, spec: ModuleSpec, params: EffectiveSDM | None = None,
            *, tol=1e-6, maxiter=100):
    reason = availability(model, spec, params)
    if reason:
        raise ValueError(reason)
    g, tm, tc = np.broadcast_arrays(np.atleast_1d(g_poa).astype(float),
        np.atleast_1d(t_module).astype(float), np.atleast_1d(t_cell).astype(float))
    if g.ndim != 1 or np.any(~np.isfinite(g)) or np.any(g < 0):
        raise ValueError("Gpoa deve ser vetor finito e não negativo; GHI não é POA.")
    needed_t = tm if model == "M3" else tc
    if model != "M1" and (np.any(~np.isfinite(needed_t)) or np.any(needed_t <= -273.15)):
        raise ValueError("Temperatura inválida.")
    u = g/1000
    ok = np.ones(len(g), bool)
    it = np.zeros(len(g), np.int32)
    status = np.full(len(g), "ok", dtype="U32")
    if model == "M1":
        power = spec.p_stc_W*u
    elif model == "M2":
        power = spec.p_stc_W*u*(1+spec.gamma_pct_C/100*(tc-25))
    elif model == "M3":
        logu = np.zeros_like(u)
        np.log(u, out=logu, where=u > 0)
        dt = tm-25
        k1,k2,k3,k4,k5,k6 = HULD
        eta = 1+k1*logu+k2*logu**2+k3*dt+k4*dt*logu+k5*dt*logu**2+k6*dt**2
        power = spec.p_stc_W*u*eta
        status[(g > 0) & (g < 100)] = "low_G_extrapolation"
    elif model == "M4":
        power = np.zeros_like(g)
        positive = g > 0
        if np.any(positive):
            r = sdm_points(g[positive], tc[positive], spec, params, tol, maxiter)
            power[positive], ok[positive], it[positive] = r["Pmp"], r["converged"], r["iterations"]
        status[~ok] = "solver_not_converged"
        power[~ok] = np.nan
    else:
        raise ValueError("Modelo desconhecido.")
    status[power < 0] = "clipped_zero"
    power = np.maximum(power, 0)
    ok &= np.isfinite(power)
    return Prediction(power, ok, it, status)


def iv_curve(g, tc, spec, params, n_series=1, n_parallel=1):
    if min(n_series, n_parallel) < 1 or int(n_series) != n_series or int(n_parallel) != n_parallel:
        raise ValueError("Arranjo requer inteiros positivos.")
    if g <= 0:
        return np.zeros(200), np.zeros(200), np.zeros(200)
    r = sdm_points([g], [tc], spec, params)
    if not r["converged"][0]:
        raise ValueError("Solver SDM não convergiu.")
    v = np.linspace(0,r["Voc"][0],300)
    i = np.maximum(current(v, operating(params,spec,g,tc)),0)
    return v*n_series, i*n_parallel, v*i*n_series*n_parallel
