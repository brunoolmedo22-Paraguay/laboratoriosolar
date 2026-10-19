from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reference/handoff/input"
MODELS = ("M1", "M2", "M3", "M4")
LABELS = {"M1": "Proporcional", "M2": "PVWatts DC", "M3": "Huld / PVGIS", "M4": "Diodo único"}
HULD = (-0.017237, -0.040465, -0.004702, 0.000149, 0.000170, 0.000005)


@dataclass(frozen=True)
class ModuleSpec:
    id: str
    p_stc_W: float
    technology: str = "c-Si"
    manufacturer: str | None = None
    commercial_model: str | None = None
    isc_A: float | None = None
    voc_V: float | None = None
    vmp_V: float | None = None
    imp_A: float | None = None
    n_cells: int | None = None
    alpha_pct_C: float | None = None
    beta_pct_C: float | None = None
    gamma_pct_C: float | None = None
    noct_C: float | None = None
    u0: float | None = None
    u1: float | None = None
    eg_ref_eV: float = 1.121
    degdt_per_K: float = -0.0002677
    source: str = ""

    def validate(self):
        if not self.id.strip() or not self.source.strip():
            raise ValueError("Identificação e fonte são obrigatórias.")
        for k, v in asdict(self).items():
            if isinstance(v, (float, int)) and not math.isfinite(v):
                raise ValueError(f"{k}: valor não finito.")
        for k in ("p_stc_W", "isc_A", "voc_V", "vmp_V", "imp_A", "n_cells", "eg_ref_eV"):
            v = getattr(self, k)
            if v is not None and v <= 0:
                raise ValueError(f"{k} deve ser positivo.")
        if self.n_cells is not None and int(self.n_cells) != self.n_cells:
            raise ValueError("Número de células deve ser inteiro.")
        if self.isc_A and self.voc_V and self.p_stc_W > self.isc_A * self.voc_V:
            raise ValueError("Pstc deve ser menor que Voc × Isc.")
        if self.vmp_V and self.voc_V and self.vmp_V >= self.voc_V:
            raise ValueError("Vmp deve ser menor que Voc.")
        if self.imp_A and self.isc_A and self.imp_A >= self.isc_A:
            raise ValueError("Imp deve ser menor que Isc.")
        if self.u0 is not None and self.u0 <= 0 or self.u1 is not None and self.u1 < 0:
            raise ValueError("Coeficientes térmicos inválidos.")
        return self

    @property
    def alpha_A_C(self):
        if self.alpha_pct_C is None or self.isc_A is None:
            raise ValueError("SDM requer alpha e Isc.")
        return self.alpha_pct_C / 100 * self.isc_A


@dataclass(frozen=True)
class EffectiveSDM:
    IL_ref: float
    I0_ref: float
    Rs: float
    Rsh_ref: float
    a_ref: float

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Parâmetros efetivos SDM devem ser positivos e finitos.")


@dataclass(frozen=True)
class ExperimentConfig:
    months: tuple[str, ...] = ("2019_01", "2019_02", "2019_03", "2019_04", "2019_05", "2019_06")
    rounds: int = 11
    seed: int = 2026
    order: str = "rotate"
    thermal: str = "measured_back"
    delta_T_C: float = 3.0
    g_min_Wm2: float = 100.0
    g_max_Wm2: float = 1600.0
    temperature_min_C: float = -50.0
    temperature_max_C: float = 100.0
    max_gap_minutes: float = 7.5
    bootstrap_reps: int = 1000
    threads: int = 1
    rss_interval_s: float = 0.002
    target_batch_s: float = 0.1
    mpp_tolerance_V: float = 1e-6
    max_iterations: int = 100
    scalability_sizes: tuple[int, ...] = (128, 1024)
    thermal_sensitivity: tuple[float, ...] = (0.0, 2.0, 3.0, 5.0)

    def __post_init__(self):
        for field in ("months","scalability_sizes","thermal_sensitivity"):
            object.__setattr__(self,field,tuple(getattr(self,field)))
        if not isinstance(self.rounds, int) or not 3 <= self.rounds <= 101:
            raise ValueError("Use de 3 a 101 rodadas inteiras.")
        if not self.months or any(m not in {f"2019_{i:02}" for i in range(1, 7)} for m in self.months):
            raise ValueError("Selecione meses SUPSI de janeiro a junho de 2019.")
        if len(set(self.months)) != len(self.months):
            raise ValueError("Meses duplicados.")
        if self.order not in ("rotate", "random") or self.thermal not in ("measured_back", "faiman"):
            raise ValueError("Protocolo desconhecido.")
        if not 0 <= self.g_min_Wm2 < self.g_max_Wm2 or not 0 < self.max_gap_minutes <= 60:
            raise ValueError("Limites de irradiância/intervalo inválidos.")
        if not 0 <= self.delta_T_C <= 20 or self.threads < 1 or self.bootstrap_reps < 100 or self.seed<0:
            raise ValueError("Configuração térmica/estatística inválida.")
        if self.mpp_tolerance_V <= 0 or self.max_iterations < 1 or self.target_batch_s <= 0:
            raise ValueError("Tolerância/tempo inválido.")
        if self.rss_interval_s <= 0 or any(n < 1 for n in self.scalability_sizes):
            raise ValueError("Amostragem de memória/escala inválida.")


def dump_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def supsi_spec():
    d = json.loads((INPUT / "SUPSI/derived/SUPSI_module_experimental.json").read_text())
    return ModuleSpec(id=d["id"], p_stc_W=d["power_stc_measured_W"],
        isc_A=d["isc_stc_A"], voc_V=d["voc_stc_V"], alpha_pct_C=d["alpha_isc_pct_per_C"],
        beta_pct_C=d["beta_voc_pct_per_C"], gamma_pct_C=d["gamma_pmax_pct_per_C"],
        noct_C=d["nmot_C"], u0=d["u0_W_m2K"], u1=d["u1_W_s_m3K"], source=d["source_original"]).validate()


def catalogue():
    from .legacy.config.pv_database import MODULE_DB
    out = {"SUPSI experimental": supsi_spec()}
    for key, entry in MODULE_DB.items():
        s = entry["stc"]
        out[key] = ModuleSpec(id=key, p_stc_W=s["p_nom"], manufacturer=s["manufacturer"],
            commercial_model=s["model"], isc_A=s["i_sc"], voc_V=s["v_oc"],
            vmp_V=s["v_mp"], imp_A=s["i_mp"], n_cells=s["n_cells"],
            alpha_pct_C=s["alpha_isc_pct"], beta_pct_C=s["beta_voc_pct"],
            gamma_pct_C=s["gamma_pmax_pct"], noct_C=s["noct"],
            source="Catálogo Canadian Solar do handoff v1.0; " + s["notes"]).validate()
    return out
