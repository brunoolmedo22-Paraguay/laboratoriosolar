# CIM26 Lab 1.0.0

Aplicação científica Python + Streamlit para estudar exatidão de estimativas fotovoltaicas DC e
custo computacional. Dois modos: **Modelos** e **Artigo**. Quatro modelos: proporcional,
PVWatts DC, Huld/PVGIS CSi CURRENT e SDM. Interface em português brasileiro.

Inclui os dados originais, catálogo Canadian Solar, caracterização SUPSI, motor independente,
calibração clássica e multiponto, histórico local, testes, documentação e uma execução de engenharia
com seis meses. Não é um artigo científico nem uma conclusão de superioridade de um modelo.

## Iniciar

Requer Python 3.11 ou superior. Internet é necessária apenas para instalar as dependências.
Depois disso, os dois modos e todos os dados incluídos funcionam sem Internet, APIs ou credenciais.

### Windows

Extraia o ZIP inteiro. Na pasta que contém `app.py`, execute `iniciar_windows.bat`, ou:

```powershell
py -3 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

### Linux / macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Abra a URL local mostrada no terminal, normalmente `http://localhost:8501`. Para encerrar, Ctrl+C.
O ambiente testado está descrito em `docs/VERIFICACAO.md` e no `environment_freeze.txt` de cada execução.
As versões principais testadas estão em `requirements-tested.txt`; o freeze completo registra também
pacotes existentes no ambiente de teste, não sendo necessário instalar todos eles para usar a aplicação.

## Modo Modelos

1. Selecione um dos 16 módulos Canadian Solar, SUPSI ou crie uma ficha com seus dados e fonte.
2. Para M4, clique em **Estimar parâmetros SDM**. A ficha comercial usa o calibrador legado;
   SUPSI usa apenas a matriz de laboratório. Confira o diagnóstico e exporte os parâmetros.
3. Selecione M1–M4 e o arranjo série/paralelo. Modelos inviáveis mostram a informação ausente.
4. Use uma condição constante, perfil sintético ou CSV com mapeamento explícito de colunas.
   Declare Tmod, Tcell ou Tamb; Tamb usa a aproximação NOCT/NMOT informada na ficha.
   GHI não é renomeada como POA. A transposição GHI→POA não está implementada.
5. Execute e exporte CSV, configuração e figuras SVG/PNG. I–V e P–V só existem para SDM.

O horizonte não está limitado a 120 minutos. A UI limita uma simulação a 250 mil amostras para
proteger a memória; o usuário pode aumentar o passo. A energia respeita a lacuna máxima configurada.
Os campos alterados em Modelos não alteram o protocolo SUPSI de Artigo.

## Modo Artigo

Abra **Histórico** para reabrir uma execução existente sem recalcular, importar um ZIP exportado
ou comparar tabelas de diferentes configurações. A última execução já incluída também aparece
automaticamente como resultado salvo; não é executada novamente na inicialização.

Em **Preparar experimento**, escolha meses, filtro G, estratégia térmica, ΔT, rodadas, ordem e semente.
A tabela de qualidade conserva todas as linhas e indica os motivos de exclusão. A caracterização
é separada das medições de validação. O botão de execução cria sempre um diretório novo.

- **Energia:** potência medida/prevista, energia parcial, erros, curvas e resíduos por condição.
- **Computacional:** 44 registros principais por padrão, tempos, RSS, convergência e escalabilidade.
- **Análise integrada:** Pareto erro×tempo/memória, melhoria marginal e sensibilidade térmica.
- **Estatística:** distribuições, dispersão, bootstrap pareado por rodada/dia e estabilidade temporal.

O benchmark é estritamente sequencial. Preparação, calibração, exportação e gráficos ficam fora do
tempo do modelo. Escalabilidade usa três rodadas adicionais por tamanho, em arquivo separado.
RSS é um pico amostrado, não um máximo exato. Não foi medida a energia da CPU.

## CLI e reprodução

Use o Python do ambiente virtual nos comandos abaixo.

```bash
python -m cim26 run
python -m cim26 run --months 2019_01 --rounds 3 --no-scalability
python -m cim26 run --config runs/ID_DA_EXECUCAO/config.json
python -m cim26 verify runs/ID_DA_EXECUCAO
python -m cim26 figures runs/ID_DA_EXECUCAO
python -m cim26 extract-lab --output matriz_reextraida.csv
python -m pytest -q
```

`run --config` usa os snapshots CSV associados ao diretório e cria outro resultado. A caracterização
SUPSI é fixa nesta versão; hashes e parâmetros são registrados. `verify` confere o manifesto e
recalcula métricas energéticas a partir de predictions.csv, sem recalcular modelos. `figures` altera
somente figuras e atualiza seu manifesto; preserve uma cópia se desejar manter o export byte a byte.

Cada execução exporta configuração, fontes congeladas, qualidade de todas as linhas, parâmetros e
resíduos de calibração, previsões, resultados de cada rodada, métricas, estatísticas, sensibilidade,
escalabilidade, ambiente/hashes e figuras. Não é preciso instalar Excel. O XLSX é somente leitura.

Se o processo for encerrado abruptamente, o histórico parcial permanece. Um bloqueio do sistema
operacional em `.experiment.lock` evita duas execuções simultâneas da mesma instalação e é liberado
automaticamente ao terminar o processo. A presença do arquivo, sozinha, não indica execução ativa;
não é necessário apagá-lo. Não execute duas instalações em paralelo ao medir tempos para publicação.

## Organização

```text
app.py                     Interface Streamlit
cim26/schema.py            Unidades, fichas, configuração congelada
cim26/physics.py           Quatro modelos, térmica, I–V e referência numérica
cim26/calibration.py       Datasheet clássico e SUPSI multiponto
cim26/data.py              Leitores, rastreabilidade e qualidade
cim26/experiment.py        Orquestração serial e persistência
cim26/worker.py            Processo isolado, relógio e RSS
cim26/metrics.py           Energia, erros, bootstrap e Pareto
cim26/report.py            Exportação, reapertura e figuras
cim26/legacy/              Calibrador herdado com imports qualificados
reference/handoff/         Pacote recebido, integral e inalterado
runs/                      Execuções independentes e imutáveis
tests/                     Testes científicos, de integração e da UI
docs/                      Auditoria e evidências de verificação
```

Leia `docs/AUDITORIA.md` para diferenças encontradas nas fontes, hipóteses térmicas, alteração
algorítmica explícita no solver vetorizado, limites instrumentais e referências. O código novo não
adiciona termos de licença aos materiais de terceiros: preserva os arquivos e atribuições recebidos.
