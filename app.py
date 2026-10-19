"""Presentation only: every physical and experimental calculation lives in cim26."""
import os
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key,"1")
from dataclasses import asdict,replace
from pathlib import Path
import hashlib
import json
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from cim26 import __version__
from cim26.schema import ROOT,MODELS,LABELS,ModuleSpec,EffectiveSDM,ExperimentConfig,catalogue,supsi_spec,dump_json
from cim26.data import read_field,field_paths,quality_flags,prepared_conditions,laboratory,custom_profile
from cim26.calibration import classic,multipoint
from cim26.physics import predict,availability,thermal_conditions,iv_curve
from cim26.metrics import energy_integral
from cim26.experiment import run_experiment
from cim26.report import COLORS,load_run,run_zip,import_run_zip,simulation_zip,plot_arrays

st.set_page_config(page_title="CIM26 · Laboratório fotovoltaico",page_icon="☀",layout="wide")
st.markdown("""<style>
.block-container{padding-top:2.2rem;max-width:1500px}
h1,h2,h3{letter-spacing:-.025em} h1{font-size:2.25rem!important}
[data-testid="stSidebar"]{border-right:1px solid #dbe4ee}
[data-testid="stMetric"]{background:white;border:1px solid #e0e7ef;border-radius:14px;padding:17px}
.eyebrow{font-size:11px;letter-spacing:.18em;font-weight:750;color:#168b85;margin-bottom:9px}
.intro{color:#60758a;max-width:850px;margin-bottom:26px;line-height:1.65}
.model-note{border-left:3px solid #168b85;padding:10px 16px;background:#e9f5f3;border-radius:0 8px 8px 0}
.stButton>button{border-radius:9px}.stTabs [data-baseweb="tab"]{font-weight:600}
</style>""",unsafe_allow_html=True)
px.defaults.template="plotly_white"
px.defaults.color_discrete_map=COLORS


def chart(fig,key=None):
    fig.update_layout(margin=dict(l=10,r=10,t=35,b=15),font=dict(family="Arial",color="#17324d"),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="white",legend_title_text="",
        hovermode="x unified",legend=dict(orientation="h",y=1.12),height=380)
    st.plotly_chart(fig,width="stretch",key=key,config={"displaylogo":False})


def line(frame,x,ys,y_title,key=None,gap_minutes=None):
    fig=go.Figure()
    for col in ys:
        xx,yy=plot_arrays(frame,x,col,gap_minutes)
        fig.add_trace(go.Scatter(x=xx,y=yy,name="Medido" if col=="Pm" else col,
            line=dict(color=COLORS.get(col),width=2),connectgaps=False))
    fig.update_yaxes(title=y_title)
    fig.update_xaxes(rangeslider_visible=True)
    chart(fig,key)


@st.cache_data(show_spinner=False)
def fit_spec(payload):
    spec=ModuleSpec(**json.loads(payload))
    return multipoint(spec,laboratory()) if spec.id==supsi_spec().id else classic(spec)


def calibration_panel(params,diag,residuals):
    with st.expander("Diagnóstico da identificação",expanded=False):
        st.write("Método: **"+diag["method"]+"**")
        st.json({"convergiu":diag["success"],"tempo_s":diag["calibration_s"],
                 "custo_normalizado":diag["cost"],"parametros":asdict(params)})
        if not residuals.empty:st.dataframe(residuals,width="stretch",hide_index=True)
        for note in diag.get("assumptions",[]):st.caption(note)
        st.download_button("Baixar identificação e metadados",json.dumps(diag,ensure_ascii=False,indent=2),
            file_name="identificacao_SDM.json",mime="application/json")


def edit_spec(spec,new=False):
    with st.expander("Nova ficha comercial" if new else "Editar ficha e registrar a fonte",expanded=new):
        st.caption("Campos sem dados podem permanecer vazios. A disponibilidade dos modelos será indicada.")
        key="new" if new else spec.id
        a,b,c=st.columns(3)
        with a:
            name=st.text_input("Identificação",value="" if new else spec.id,key=key+"name")
            man=st.text_input("Fabricante",value="" if new else spec.manufacturer or "",key=key+"man")
            techs=["c-Si","CdTe","CIGS","Outra"]
            tech=st.selectbox("Tecnologia",techs,index=techs.index(spec.technology) if spec.technology in techs else 3,key=key+"tech")
            p=st.number_input("Pstc (W)",min_value=0.001,value=None if new else float(spec.p_stc_W),key=key+"p")
        with b:
            voc=st.number_input("Voc (V)",min_value=0.001,value=None if new else spec.voc_V,key=key+"voc")
            isc=st.number_input("Isc (A)",min_value=0.001,value=None if new else spec.isc_A,key=key+"isc")
            vmp=st.number_input("Vmp (V)",min_value=0.001,value=None if new else spec.vmp_V,key=key+"vmp")
            imp=st.number_input("Imp (A)",min_value=0.001,value=None if new else spec.imp_A,key=key+"imp")
        with c:
            cells=st.number_input("Células em série (Ns elétrico)",min_value=1,value=None if new else spec.n_cells,key=key+"cells")
            alpha=st.number_input("α Isc (%/°C)",value=None if new else spec.alpha_pct_C,format="%.5f",key=key+"a")
            beta=st.number_input("β Voc (%/°C)",value=None if new else spec.beta_pct_C,format="%.5f",key=key+"b")
            gamma=st.number_input("γ Pmax (%/°C)",value=None if new else spec.gamma_pct_C,format="%.5f",key=key+"gam")
        noct=st.number_input("NOCT/NMOT utilizado na aproximação térmica (°C)",min_value=20.0,value=None if new else spec.noct_C,key=key+"noct")
        source=st.text_area("Fonte / revisão do datasheet / justificativa de alterações",value="" if new else spec.source,key=key+"source")
        if p is None:
            st.info("Informe ao menos identificação, Pstc e fonte.");return None
        try:
            edited=ModuleSpec(id=name,p_stc_W=p,manufacturer=man or None,commercial_model=name,technology=tech,
                isc_A=isc,voc_V=voc,vmp_V=vmp,imp_A=imp,n_cells=cells,alpha_pct_C=alpha,beta_pct_C=beta,
                gamma_pct_C=gamma,noct_C=noct,source=source).validate()
        except ValueError as e:st.warning(str(e));return None
        if gamma is not None and gamma>=0:st.warning("Gamma não negativo: confirme e justifique na fonte da ficha.")
        if st.button("Salvar ficha local",key=key+"save"):
            folder=ROOT/"user_modules";folder.mkdir(exist_ok=True)
            digest=hashlib.sha256(json.dumps(asdict(edited),sort_keys=True).encode()).hexdigest()[:12]
            dump_json(folder/(digest+".json"),asdict(edited));st.success("Ficha salva para as próximas sessões.")
        st.download_button("Baixar ficha JSON",json.dumps(asdict(edited),ensure_ascii=False,indent=2),
            file_name="ficha_modulo.json",key=key+"download")
        return edited


def models_mode():
    st.markdown('<div class="eyebrow">SIMULAÇÃO DC</div>',unsafe_allow_html=True)
    st.title("Do módulo à curva de potência")
    st.markdown('<div class="intro">Explore condições de operação, identifique o circuito equivalente e dimensione um arranjo uniforme. Todos os resultados usam o mesmo motor do laboratório experimental.</div>',unsafe_allow_html=True)
    modules=catalogue()
    for f in sorted((ROOT/"user_modules").glob("*.json")):
        try:s=ModuleSpec(**json.loads(f.read_text())).validate();modules["Salvo · "+s.id]=s
        except (ValueError,TypeError,json.JSONDecodeError):st.warning("Ficha inválida: "+f.name)
    chosen=st.selectbox("Módulo",list(modules)+["Novo módulo comercial"],index=1)
    new=chosen=="Novo módulo comercial"
    spec=modules.get(chosen,supsi_spec())
    if new or st.checkbox("Editar parâmetros desta simulação",value=False):
        spec=edit_spec(spec,new)
        if spec is None:return
    spec.validate()
    cols=st.columns(4)
    cols[0].metric("Potência STC",f"{spec.p_stc_W:.2f} W")
    cols[1].metric("Voc",f"{spec.voc_V:.2f} V" if spec.voc_V else "Não informado")
    cols[2].metric("Isc",f"{spec.isc_A:.3f} A" if spec.isc_A else "Não informado")
    cols[3].metric("Tecnologia",spec.technology)
    st.caption(spec.source)
    fingerprint=json.dumps(asdict(spec),sort_keys=True)
    fitkey="fit_"+hashlib.sha256(fingerprint.encode()).hexdigest()[:12]
    if st.button("Estimar parâmetros SDM",type="secondary"):
        try:
            with st.spinner("Identificando o circuito equivalente…"):
                st.session_state[fitkey]=fit_spec(fingerprint)
        except (ValueError,RuntimeError) as e:st.error(str(e))
    params=None
    if fitkey in st.session_state:
        pp,diag,residuals=st.session_state[fitkey]
        calibration_panel(pp,diag,residuals)
        if diag["success"]:params=pp
        else:st.error("A identificação não satisfez o critério de convergência. M4 indisponível.")
    a,b,c=st.columns([2,1,1])
    model=a.selectbox("Modelo",MODELS,format_func=lambda m:f"{m} · {LABELS[m]}")
    ns=b.number_input("Módulos em série",min_value=1,value=1,step=1)
    np_=c.number_input("Strings em paralelo",min_value=1,value=1,step=1)
    reason=availability(model,spec,params)
    if reason:st.info(reason)
    st.caption(f"Potência instalada: {spec.p_stc_W*ns*np_/1000:.3f} kWp · módulos idênticos, sem sombreamento ou perdas de conexão.")
    entry=st.radio("Condições meteorológicas",["Condição constante","Perfil sintético","CSV local"],horizontal=True)
    frame=None;provenance={}
    delta=st.number_input("ΔT célula–módulo a 1000 W/m² (°C, hipótese)",min_value=0.0,max_value=20.0,value=3.0)
    if entry=="CSV local":
        upload=st.file_uploader("CSV com timestamp, irradiância e temperatura",type="csv",key="profile_csv")
        if upload:
            try:
                raw=pd.read_csv(io.BytesIO(upload.getvalue()),sep=None,engine="python")
                a,b,c=st.columns(3)
                tcol=a.selectbox("Coluna de data/hora",raw.columns)
                gcol=b.selectbox("Coluna de irradiância",raw.columns,index=min(1,len(raw.columns)-1))
                tempcol=c.selectbox("Coluna de temperatura",raw.columns,index=min(2,len(raw.columns)-1))
                kind=st.radio("Grandeza de irradiância",["POA","GHI"],horizontal=True)
                tempkind=st.radio("Grandeza de temperatura",["Tmod","Tcell","Tamb"],horizontal=True)
                fmt=st.text_input("Formato explícito de data/hora",value="%Y-%m-%d %H:%M:%S")
                frame=custom_profile(raw,tcol,gcol,tempcol,irradiance_kind=kind,timestamp_format=fmt,
                    temperature_kind=tempkind,spec=spec,delta_T_C=delta)
                provenance=dict(kind="user_supplied",filename=upload.name,sha256=hashlib.sha256(upload.getvalue()).hexdigest(),
                    mapping=dict(timestamp=tcol,Gpoa=gcol,temperature=tempcol,temperature_kind=tempkind),timestamp_format=fmt)
                st.dataframe(frame.head(30),hide_index=True,width="stretch")
            except Exception as e:st.error(str(e))
    else:
        a,b,c=st.columns(3)
        duration=a.number_input("Horizonte (min)",min_value=1,value=120,step=1)
        step=b.number_input("Passo (min)",min_value=1,value=1,step=1)
        peak=c.number_input("POA constante / pico (W/m²)",min_value=0.0,max_value=2000.0,value=1000.0)
        a,b=st.columns(2)
        tempkind=a.selectbox("Temperatura fornecida",["Tmod","Tcell","Tamb"])
        temperature=b.number_input("Temperatura (°C)",min_value=-80.0,max_value=120.0,value=25.0)
        cloudy=entry=="Perfil sintético" and st.checkbox("Oscilações de nuvens simuladas",value=True)
        count=int(duration//step)+1
        if count>250000:st.warning("Reduza o horizonte ou aumente o passo para até 250 mil amostras.")
        else:
            minutes=np.arange(0,duration+1,step)
            phase=minutes/max(duration,1)
            g=np.full(len(minutes),peak) if entry=="Condição constante" else peak*np.sin(np.pi*phase).clip(0)
            if cloudy:g*=1-.45*np.sin(phase*9*np.pi)**8
            try:
                method={"Tmod":"measured_back","Tcell":"measured_cell","Tamb":"noct"}[tempkind]
                tm,tc=thermal_conditions(g,t_back=temperature,t_cell=temperature,t_ambient=temperature,
                    method=method,spec=spec,delta_T_C=delta)
                frame=pd.DataFrame(dict(timestamp=pd.Timestamp("2026-01-01 10:00")+pd.to_timedelta(minutes,unit="min"),
                    Gpoa=g,Tmod_C=tm,Tcell_C=tc))
                provenance=dict(kind="simulated",profile=entry,temperature_kind=tempkind,
                    duration_minutes=duration,step_minutes=step,clouds=bool(cloudy))
            except ValueError as e:st.warning(str(e))
    gap=st.number_input("Maior intervalo integrado (min)",min_value=.1,value=7.5,
        help="Intervalos maiores são lacunas e não geram energia. Ajuste conforme a cadência do CSV.")
    if model=="M3":st.caption("Huld usa Tmod e coeficientes tecnológicos CSi CURRENT; sem ajuste específico ao fabricante.")
    if st.button("Simular potência DC",type="primary",disabled=bool(reason) or frame is None):
        try:
            result=predict(model,frame.Gpoa,frame.Tmod_C,frame.Tcell_C,spec,params)
            out=frame.copy();out["P_module_W"]=result.power_W;out["P_array_W"]=result.power_W*ns*np_
            out["converged"]=result.converged;out["status"]=result.status
            meta=dict(spec=asdict(spec),model=model,n_series=ns,n_parallel=np_,delta_T_C=delta,
                max_gap_minutes=gap,provenance=provenance,parameters=asdict(params) if params else None)
            iv=None
            if model=="M4":
                idx=int(out.Gpoa.to_numpy().argmax())
                v,i,p=iv_curve(out.Gpoa.iloc[idx],out.Tcell_C.iloc[idx],spec,params,ns,np_)
                iv=pd.DataFrame({"V_array_V":v,"I_array_A":i,"P_array_W":p})
                meta["iv_condition"]=dict(Gpoa=float(out.Gpoa.iloc[idx]),Tcell=float(out.Tcell_C.iloc[idx]))
            st.session_state["simulation"]=(out,meta,iv)
        except Exception as e:st.error(str(e))
    if "simulation" in st.session_state:
        out,meta,iv=st.session_state["simulation"]
        st.divider();st.subheader("Resultado da simulação")
        st.caption(f"{meta['spec']['id']} · {meta['model']} · {meta['n_series']} série × {meta['n_parallel']} paralelo. Resultado da última execução; alterar controles exige simular novamente.")
        energy,hours,_=energy_integral(out,"P_array_W",out.converged,meta["max_gap_minutes"])
        a,b,c=st.columns(3)
        a.metric("Pico DC",f"{out.P_array_W.max():.2f} W");b.metric("Energia parcial",f"{energy:.4f} kWh");c.metric("Tempo integrado",f"{hours:.2f} h")
        line(out,"timestamp",["P_array_W"],"Potência DC do arranjo (W)",gap_minutes=meta["max_gap_minutes"])
        if not out.converged.all():st.warning(f"{(~out.converged).sum()} falhas de convergência; veja a tabela.")
        if iv is not None:
            a,b=st.columns(2)
            with a:chart(px.line(iv,x="V_array_V",y="I_array_A",labels={"V_array_V":"Tensão (V)","I_array_A":"Corrente (A)"}))
            with b:chart(px.line(iv,x="V_array_V",y="P_array_W",labels={"V_array_V":"Tensão (V)","P_array_W":"Potência (W)"}))
            st.caption(f"Curvas SDM na condição de maior irradiância: {meta['iv_condition']}. Escala elétrica aplicada ao arranjo.")
        st.download_button("Exportar simulação · CSV, SVG e PNG",simulation_zip(out,meta,iv),
            file_name="CIM26_simulacao.zip",mime="application/zip")
        with st.expander("Dados e proveniência"):st.dataframe(out,width="stretch",hide_index=True);st.json(meta)


@st.cache_data(show_spinner=False)
def read_saved(path):
    return load_run(Path(path))


def results_panel(path):
    try:data=read_saved(str(path))
    except Exception as e:
        st.error("Execução incompleta: "+str(e))
        if (Path(path)/"metadata.json").exists():st.json(json.loads((Path(path)/"metadata.json").read_text()))
        if (Path(path)/"benchmark_runs.csv").exists():st.dataframe(pd.read_csv(Path(path)/"benchmark_runs.csv"))
        return
    meta=data["metadata"];cfg=data["config"];pred=data["predictions"].copy()
    pred["timestamp"]=pd.to_datetime(pred.timestamp)
    st.divider()
    st.subheader("Resultados experimentais")
    st.caption(f"Execução {meta['run_id']} · {meta['status']} · {cfg['rounds']} rodadas · ΔT={cfg['delta_T_C']} °C · G≥{cfg['g_min_Wm2']} W/m²")
    st.info("Diagnóstico de engenharia. Repetições de software medem variabilidade computacional; não são novas medições físicas. Energia parcial, sem preencher lacunas.")
    a,b,c,d=st.columns(4)
    a.metric("Amostras recebidas",f"{meta['N_total']:,}")
    b.metric("Interseção válida",f"{meta['N_common']:,}")
    c.metric("Rodadas × modelos",str(meta["benchmark_records"]))
    d.metric("Cobertura temporal",f"{data['metrics_energy'].covered_hours.iloc[0]:.1f} h")
    st.download_button("Exportar execução completa · ZIP",run_zip(path),file_name=Path(path).name+".zip",mime="application/zip",key="export_run")
    tabs=st.tabs(["Energia","Computacional","Análise integrada","Estatística"])
    if not meta["N_common"]:
        with tabs[0]:st.error("Sem previsões válidas comuns. A energia não está disponível; consulte as falhas computacionais.")
        with tabs[1]:st.dataframe(data["benchmark_runs"],width="stretch",hide_index=True)
        with tabs[2]:st.info("Análise integrada indisponível sem interseção válida.")
        with tabs[3]:st.info("Não há amostras válidas para inferência estatística.")
        return
    with tabs[0]:
        metrics=data["metrics_energy"]
        st.dataframe(metrics[["model","RMSE_W","MAE_W","MBE_W","NRMSE_pct_Pstc","E_observed_kWh","E_predicted_kWh","E_error_pct","coverage_of_eligible"]],
            width="stretch",hide_index=True)
        st.caption("MBE positivo indica superestimação. NRMSE normalizado por Pstc. Todas as métricas comparativas usam a mesma interseção.")
        days=sorted(pred.timestamp.dropna().dt.date.unique())
        window=st.date_input("Janela da curva de potência",value=(days[0],days[min(1,len(days)-1)]),
            min_value=days[0],max_value=days[-1],key="result_dates_"+str(path))
        if isinstance(window,(list,tuple)) and len(window)==2:
            mask=pred.timestamp.dt.date.between(window[0],window[1])
            line(pred[mask],"timestamp",["Pm",*MODELS],"Potência DC por módulo (W)","energy_power",gap_minutes=cfg["max_gap_minutes"])
        cumulative=pd.DataFrame({"timestamp":pred.timestamp})
        for m in ["Pm",*MODELS]:
            _,_,cumulative[m]=energy_integral(pred,m,pred.common_valid,cfg["max_gap_minutes"])
        line(cumulative,"timestamp",["Pm",*MODELS],"Energia parcial acumulada (kWh)","energy_cum")
        stats=data["statistics"]
        daily=stats[(stats.kind=="group")&(stats.dimension=="day")]
        chart(px.line(daily,x="group",y="RMSE_W",color="model",labels={"group":"Dia","RMSE_W":"RMSE diário (W)"}),"daily_rmse")
        a,b=st.columns(2)
        with a:
            m=st.selectbox("Resíduos de",MODELS,key="residual_model")
            chart(px.scatter(pred[pred.common_valid],x="Gpoa",y=m+"_residual_W",color="Tmod_C",
                labels={"Gpoa":"POA (W/m²)",m+"_residual_W":"Predito − medido (W)","Tmod_C":"Tmod (°C)"}),"residual_G")
        with b:
            chart(px.scatter(pred[pred.common_valid],x="Tmod_C",y=m+"_residual_W",color="Gpoa",
                labels={"Tmod_C":"Tmod (°C)",m+"_residual_W":"Predito − medido (W)","Gpoa":"POA (W/m²)"}),"residual_T")
        with st.expander("Tabela completa de previsões e flags"):
            st.dataframe(pred,width="stretch",hide_index=True)
            st.download_button("Baixar predictions.csv",pred.to_csv(index=False),"predictions.csv","text/csv")
    with tabs[1]:
        bench=data["benchmark_runs"];summary=data["benchmark_summary"]
        st.dataframe(summary,width="stretch",hide_index=True)
        chart(px.line(bench,x="round",y="us_per_sample",color="model",markers=True,log_y=True,
            labels={"round":"Rodada","us_per_sample":"Tempo por amostra (µs, log)"}),"rounds")
        a,b=st.columns(2)
        with a:chart(px.bar(summary,x="model",y="median_RSS_peak_MiB",color="model",
            labels={"median_RSS_peak_MiB":"Pico RSS amostrado (MiB)"}),"rss_abs")
        with b:chart(px.bar(summary,x="model",y="median_RSS_delta_MiB",color="model",
            labels={"median_RSS_delta_MiB":"Incremento RSS (MiB)"}),"rss_delta")
        st.caption("RSS inclui o processo Python e suas bibliotecas. O pico é amostrado, pode perder eventos breves e não equivale a tracemalloc. Deltas pequenos podem ser zero.")
        scale=data["scalability"]
        if not scale.empty:
            grouped=scale.groupby(["model","batch_N"],as_index=False).us_per_sample.median()
            full=summary[["model","median_us_per_sample"]].rename(columns={"median_us_per_sample":"us_per_sample"})
            full["batch_N"]=meta["N_eligible"]
            chart(px.line(pd.concat([grouped,full]).sort_values("batch_N"),x="batch_N",y="us_per_sample",color="model",markers=True,
                log_x=True,log_y=True,labels={"batch_N":"Tamanho do lote","us_per_sample":"Mediana por amostra (µs)"}),"scale")
            st.caption("Lotes menores: 3 rodadas adicionais sobre prefixos cronológicos. Lote completo: rodadas principais. Não misturar suas distribuições.")
        st.write(f"Preparação: {meta['prepare_s']:.3f} s · Calibração: {meta['calibration_s']:.3f} s · Pós-processamento e ensaios auxiliares: {meta['postprocess_s']:.2f} s")
        with st.expander("Registros individuais, falhas e ambiente"):
            st.dataframe(bench,width="stretch",hide_index=True)
            st.json(meta["environment"])
            st.caption(meta["memory_policy"])
    with tabs[2]:
        integrated=data["integrated"]
        st.caption("Fronteira de Pareto: nenhum outro modelo tem erro e custo menores ou iguais, com pelo menos uma desigualdade estrita. Somente rodadas completas e deterministas são elegíveis.")
        a,b=st.columns(2)
        with a:chart(px.scatter(integrated,x="median_us_per_sample",y="RMSE_W",color="model",symbol="pareto_time",
            symbol_map={True:"diamond",False:"circle"},text="model",log_x=True,
            labels={"median_us_per_sample":"Mediana por amostra (µs)","RMSE_W":"RMSE (W)","pareto_time":"Não dominado"}),"pareto_t")
        with b:chart(px.scatter(integrated,x="median_RSS_peak_MiB",y="RMSE_W",color="model",symbol="pareto_memory",
            symbol_map={True:"diamond",False:"circle"},text="model",
            labels={"median_RSS_peak_MiB":"Pico RSS (MiB)","RMSE_W":"RMSE (W)","pareto_memory":"Não dominado"}),"pareto_m")
        st.dataframe(integrated[["model","RMSE_improvement_vs_M1_W","time_ratio_vs_M1","RMSE_gain_W_per_extra_us","pareto_time","pareto_memory"]],hide_index=True,width="stretch")
        st.caption("Ganho marginal = (RMSE de M1 − RMSE do modelo) / tempo adicional por amostra. Indisponível se o custo adicional não for positivo.")
        st.subheader("Sensibilidade à temperatura da célula")
        chart(px.line(data["thermal_sensitivity"],x="delta_T_C",y="RMSE_W",color="model",markers=True,
            labels={"delta_T_C":"ΔT a 1000 W/m² (°C)","RMSE_W":"RMSE (W)"}),"thermal_sens")
        st.caption("Parâmetros SDM fixos e mesmos dados. M1 não usa temperatura; Huld usa Tmod. Varia apenas a conversão para Tcell de M2/M4.")
        st.dataframe(pd.DataFrame([
            ["M1","Pstc","Sem identificação"],["M2","Pstc, gamma, Tcell","Coeficiente experimental/datasheet"],
            ["M3","Pstc, família c-Si, Tmod","6 coeficientes tecnológicos JRC"],
            ["M4","5 parâmetros efetivos, alpha, Tcell","66 observações de laboratório ou datasheet completo"]],
            columns=["Modelo","Requisitos","Identificação"]),width="stretch",hide_index=True)
    with tabs[3]:
        bench=data["benchmark_runs"]
        chart(px.box(bench,x="model",y="us_per_sample",color="model",points="all",log_y=True,
            labels={"us_per_sample":"Tempo por amostra (µs, log)"}),"boxplot")
        stats=data["statistics"]
        st.write("**Intervalos exploratórios de 95% e diferenças pareadas**")
        st.dataframe(stats[stats.kind!="group"].dropna(axis=1,how="all"),width="stretch",hide_index=True)
        st.caption("Bootstrap de dias completos, com os mesmos dias sorteados para todos os modelos. Tempos reamostrados por rodada. Dependência temporal e sazonalidade limitam a interpretação; não há p-valores automáticos.")
        dimension=st.selectbox("Estabilidade por",["month","G_bin_Wm2","Tmod_bin_C"],
            format_func=lambda x:{"month":"Mês","G_bin_Wm2":"Faixa de irradiância","Tmod_bin_C":"Faixa de temperatura"}[x])
        groups=stats[(stats.kind=="group")&(stats.dimension==dimension)]
        chart(px.bar(groups,x="group",y="RMSE_W",color="model",barmode="group",labels={"group":"Condição","RMSE_W":"RMSE (W)"}),"group_stats")
        m=st.selectbox("Distribuição de resíduos",MODELS,key="hist_model")
        chart(px.histogram(pred[pred.common_valid],x=m+"_residual_W",nbins=50,labels={m+"_residual_W":"Resíduo (W)"}),"residual_hist")
    with st.expander("Configuração congelada, hashes e limitações"):
        st.json(cfg);st.json(meta)


def article_mode():
    st.markdown('<div class="eyebrow">PESQUISA CIM26 · SUPSI</div>',unsafe_allow_html=True)
    st.title("Exatidão e custo, no mesmo ensaio")
    st.markdown('<div class="intro">Validação independente em campo e execução sequencial dos quatro modelos. Prepare o protocolo, acompanhe cada rodada e inspecione os resultados com rastreabilidade.</div>',unsafe_allow_html=True)
    root=ROOT/"runs";root.mkdir(exist_ok=True)
    histories=sorted([p for p in root.iterdir() if p.is_dir() and (p/"metadata.json").exists()],reverse=True)
    with st.expander("Histórico · reabrir, importar e comparar",expanded=False):
        history=st.selectbox("Execução salva",["Selecione…"]+[p.name for p in histories])
        if st.button("Reabrir resultados",disabled=history=="Selecione…"):
            st.session_state["active_run"]=str(root/history)
        imported=st.file_uploader("Importar ZIP de uma execução",type="zip",key="import_run")
        if imported and st.button("Importar e abrir"):
            try:
                st.session_state["active_run"]=str(import_run_zip(imported.getvalue(),root));st.success("Execução importada sem recalcular modelos.")
            except Exception as e:st.error(str(e))
        chosen=st.multiselect("Comparar métricas de execuções salvas",[p.name for p in histories])
        if chosen:
            tables=[]
            for name in chosen:
                try:
                    run=read_saved(str(root/name));df=run["metrics_energy"].copy()
                    df.insert(0,"run",name);df["delta_T_C"]=run["config"]["delta_T_C"];df["g_min_Wm2"]=run["config"]["g_min_Wm2"]
                    tables.append(df)
                except Exception:st.warning(name+": execução incompleta.")
            if tables:st.dataframe(pd.concat(tables),hide_index=True,width="stretch")
            st.caption("Diferenças de filtros, suporte temporal e ambiente impedem uma comparação direta sem revisão do protocolo.")
    with st.expander("Preparar experimento",expanded="active_run" not in st.session_state):
        months=st.multiselect("Meses SUPSI",[f"2019_{i:02}" for i in range(1,7)],default=[f"2019_{i:02}" for i in range(1,7)])
        uploads=st.file_uploader("Ou importar CSVs SUPSI no mesmo formato (2019_01.csv … 2019_06.csv)",type="csv",accept_multiple_files=True,key="article_inputs")
        sources=None
        if uploads:
            names=[Path(u.name).name for u in uploads]
            allowed={f"2019_{i:02}.csv" for i in range(1,7)}
            if not set(names)<=allowed or len(set(names))!=len(names):
                st.error("Use nomes mensais SUPSI únicos de janeiro a junho de 2019.");return
            digest=hashlib.sha256(b"".join(u.getvalue() for u in uploads)).hexdigest()[:16]
            dest=ROOT/"user_inputs"/digest;dest.mkdir(parents=True,exist_ok=True)
            sources=[]
            for u,name in zip(uploads,names):
                p=dest/name
                if not p.exists():p.write_bytes(u.getvalue())
                sources.append(p)
            months=[Path(n).stem for n in names]
            st.caption("Os CSVs importados definem os meses desta execução: "+", ".join(months))
        a,b,c=st.columns(3)
        rounds=a.number_input("Rodadas",min_value=3,max_value=101,value=11,step=1)
        gmin=b.number_input("POA mínima (W/m²)",min_value=0.0,max_value=1500.0,value=100.0)
        delta=c.selectbox("ΔT célula–módulo (°C)",[0.0,2.0,3.0,5.0],index=2)
        a,b,c=st.columns(3)
        thermal=a.selectbox("Estratégia térmica",["measured_back","faiman"],format_func=lambda s:"A · Tbom medida" if s=="measured_back" else "B · Faiman (Tamb e vento)")
        order=b.selectbox("Ordem por rodada",["rotate","random"],format_func=lambda x:"Rotativa" if x=="rotate" else "Aleatória com semente")
        seed=c.number_input("Semente",min_value=0,max_value=2147483647,value=2026,step=1)
        a,b,c=st.columns(3)
        gap=a.number_input("Lacuna máxima integrada (min)",min_value=.1,max_value=60.0,value=7.5)
        boot=b.number_input("Reamostragens bootstrap",min_value=100,max_value=10000,value=1000,step=100)
        scale=c.checkbox("Medir escalabilidade (128 e 1024 amostras)",value=True)
        if not months:st.info("Selecione pelo menos um mês.");return
        try:
            config=ExperimentConfig(months=tuple(months),rounds=int(rounds),seed=int(seed),order=order,thermal=thermal,
                delta_T_C=delta,g_min_Wm2=gmin,max_gap_minutes=gap,bootstrap_reps=int(boot),scalability_sizes=(128,1024) if scale else ())
            raw=read_field(sources or field_paths(months));quality=quality_flags(raw,config)
            a,b,c=st.columns(3)
            a.metric("Registros",f"{len(quality):,}");b.metric("Elegíveis",f"{quality.eligible.sum():,}");c.metric("Excluídos",f"{(~quality.eligible).sum():,}")
            st.caption("POA e Tbom preservadas. GHI não substitui POA. ΔT é hipótese; coeficientes U0/U1 vêm da caracterização SUPSI.")
            with st.expander("Ver dados, qualidade e caracterização"):
                st.dataframe(quality.groupby("source_file").eligible.agg(total="size",elegiveis="sum").reset_index(),hide_index=True)
                counts=quality.exclusion_reasons.replace("","elegível").value_counts().rename_axis("motivo").reset_index(name="N")
                st.dataframe(counts,hide_index=True,width="stretch")
                variable=st.selectbox("Variável medida",["Pm","Gpoa","Tbom","AIR_TEMP","WIND_SPEED","Gglob_hor"])
                line(quality,"timestamp",[variable],variable+" · unidades originais","input_plot",gap_minutes=config.max_gap_minutes)
                st.dataframe(quality,width="stretch",hide_index=True)
                st.dataframe(laboratory(),hide_index=True,width="stretch")
                st.json(asdict(supsi_spec()))
                st.caption("Identificação comercial, Vmp, Imp e número de células SUPSI não publicados. Matriz: 22 pares G–T × Pmp/Isc/Voc. O ajuste é feito somente com laboratório, antes do benchmark.")
            if st.button("Executar experimento científico",type="primary",disabled=not quality.eligible.any()):
                bar=st.progress(0,text="Preparando execução…")
                try:
                    path=run_experiment(config,sources=sources,progress=lambda f,msg:bar.progress(min(1.0,float(f)),text=msg))
                    st.session_state["active_run"]=str(path)
                    status=json.loads((path/"metadata.json").read_text())["status"]
                    if status=="complete":st.success("Execução concluída e preservada no histórico.")
                    else:st.warning("Execução finalizada com falhas registradas; verifique a área Computacional.")
                except Exception as e:st.error(str(e))
        except Exception as e:st.error(str(e))
    if "active_run" in st.session_state:results_panel(Path(st.session_state["active_run"]))
    elif histories:
        # A bundled engineering run is immediately reviewable; no computation on startup.
        complete=[p for p in histories if (p/"metrics_energy.csv").exists()]
        if complete:
            st.caption("Resultados abaixo: última execução salva. O protocolo acima só é aplicado ao executar um novo experimento.")
            results_panel(complete[0])
    else:st.info("Configure o ensaio ou importe uma execução para abrir as quatro áreas de resultados.")


with st.sidebar:
    st.markdown("## CIM26")
    st.caption("Laboratório fotovoltaico")
    mode=st.radio("Modo",["Modelos","Artigo"],key="main_mode")
    st.divider()
    st.markdown("**4 modelos. Uma base física.**")
    st.caption("DC por módulo · dados locais · parâmetros rastreáveis")
    st.caption(f"Versão {__version__}")
    with st.expander("Convenções"):
        st.write("POA: irradiância no plano.\n\nTmod/Tbom: temperatura posterior.\n\nTcell: temperatura da célula.\n\nEnergia: somente intervalos válidos.")
if mode=="Modelos":models_mode()
else:article_mode()
