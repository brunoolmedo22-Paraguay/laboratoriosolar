import numpy as np
from streamlit.testing.v1 import AppTest
from cim26.schema import ROOT,ModuleSpec,EffectiveSDM
from cim26.physics import predict


def find(elements,label):
    return next(e for e in elements if e.label==label)


def test_models_simulation_matches_engine_and_sdm_button():
    at=AppTest.from_file(str(ROOT/"app.py"),default_timeout=45).run()
    assert not at.exception
    find(at.button,"Simular potência DC").click().run()
    assert not at.exception
    frame,meta,iv=at.session_state["simulation"]
    expected=predict(meta["model"],frame.Gpoa,frame.Tmod_C,frame.Tcell_C,ModuleSpec(**meta["spec"]))
    np.testing.assert_array_equal(frame.P_module_W,expected.power_W)
    assert iv is None
    find(at.button,"Estimar parâmetros SDM").click().run()
    assert not at.exception
    find(at.selectbox,"Modelo").set_value("M4").run()
    find(at.button,"Simular potência DC").click().run()
    assert not at.exception
    frame,meta,iv=at.session_state["simulation"]
    expected=predict("M4",frame.Gpoa,frame.Tmod_C,frame.Tcell_C,ModuleSpec(**meta["spec"]),EffectiveSDM(**meta["parameters"]))
    np.testing.assert_array_equal(frame.P_module_W,expected.power_W)
    assert iv is not None and iv.P_array_W.max()>0


def test_article_four_tabs_history_and_no_calculation_on_rerun(monkeypatch):
    import cim26.experiment
    def forbidden(*args,**kwargs):raise AssertionError("Opening history may not execute the engine")
    monkeypatch.setattr(cim26.experiment,"run_experiment",forbidden)
    at=AppTest.from_file(str(ROOT/"app.py"),default_timeout=45).run()
    at.radio(key="main_mode").set_value("Artigo").run()
    assert not at.exception and not at.error
    assert [t.label for t in at.tabs]==["Energia","Computacional","Análise integrada","Estatística"]
    find(at.selectbox,"Estabilidade por").set_value("G_bin_Wm2").run()
    assert not at.exception
    assert len(at.dataframe)>=8


def test_incomplete_custom_datasheet_and_unavailable_sdm():
    at=AppTest.from_file(str(ROOT/"app.py"),default_timeout=45).run()
    find(at.selectbox,"Módulo").set_value("Novo módulo comercial").run()
    assert not at.exception
    at.text_input(key="newname").set_value("Módulo de ensaio")
    at.number_input(key="newp").set_value(300.0)
    at.text_area(key="newsource").set_value("Ficha de ensaio fornecida pelo usuário").run()
    assert not at.exception
    find(at.selectbox,"Modelo").set_value("M4").run()
    assert find(at.button,"Simular potência DC").disabled
    find(at.button,"Estimar parâmetros SDM").click().run()
    assert not at.exception
    assert any("Datasheet incompleto" in e.value for e in at.error)
