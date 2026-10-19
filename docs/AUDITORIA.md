# Auditoria de entrada e decisões de implementação

Data: 19/10/2026. Fonte contratual: `reference/handoff/`, cópia integral do pacote recebido.
Todos os hashes do manifesto original foram verificados. CSVs, XLSX, PDFs e código de referência permanecem inalterados.

## Fontes examinadas

README_INICIO, PROMPT_PARA_CLAUDE, documentos 01–06 e notas da UI; catálogo, solver, corrente SDM,
MPP, térmica, irradiância e multimodelo legados; implementações de referência e piloto; seis CSVs;
planilha `61853-1 & 2`; proposta original (2 páginas); relatório IEA, especialmente pp. 99 e 164–165.

O contrato atual de quatro modelos prevalece sobre a proposta antiga de três modelos. O piloto não
foi utilizado como parametrização fixa nem como resultado validado. O ajuste multiponto é executado novamente.

## Achados que afetam a leitura

1. O pacote documenta um sentinela `32`; nesta cópia do XLSX, as 18 posições ausentes contêm
   strings vazias. O importador reconhece essas posições pelo par G–T, aceita vazio ou `32` somente
   ali e exige 22 observações para cada grandeza. Nunca elimina um valor 32 arbitrário em dados reais.
2. A célula B27 diz `Voc[A]`. É erro de rótulo da fonte: a grandeza é tensão, em V. O XLSX não foi
   editado. O derivado identifica a unidade como V e conserva linha e letra da coluna de origem.
3. Há 18.705 linhas de campo: 2.398, 2.536, 3.273, 3.489, 3.680 e 3.329 de janeiro a junho.
   Existem NaNs e lacunas. Os arquivos não representam cada minuto das 24 h de todos os dias.
4. O fuso horário da série não foi estabelecido pelos arquivos. Preserva-se o relógio informado,
   sem inventar UTC ou aplicar transições de horário de verão. Dias estatísticos seguem esse relógio.
5. Gpoa e Gglob_hor são distintos. Um NaN de GHI não exclui uma amostra com os insumos principais
   válidos. Os filtros principais são congelados antes de avaliar erros de previsão.
6. Identidade comercial, Ns, Vmp e Imp medidos do SUPSI continuam nulos. Valores Vmp/Imp calculados
   por um circuito ajustado são previsões, não medições ou parâmetros de fabricante.

## Compatibilidade e arquitetura nova

O legado importava `config`, `models` e `simulation` como pacotes globais e sua interface multimodelo
restringia perfis a 120 minutos, com rótulos GHI/POA ambíguos. A nova aplicação não usa esse fluxo.
O calibrador comercial e as dependências necessárias estão em `cim26/legacy/`, com alteração apenas
dos imports para nomes qualificados. As equações, cotas, chute inicial e critérios de convergência
do calibrador clássico foram preservados. Todos os 16 módulos Canadian Solar são ensaiados em STC.

O núcleo novo recebe vetores Gpoa/Tmod/Tcell explicitamente. Uma única função `predict` serve a UI,
calibração, workers, CLI e testes. O motor não importa Streamlit. Processamento de dados, física,
calibração, experimentos, estatística e exportação têm módulos separados, sem servidor de banco de dados.

## Escolhas físicas e numéricas

- P por módulo em W; G em W/m²; temperaturas de entrada em °C. O SDM converte internamente a K.
- M2 converte gamma de %/°C para 1/°C exatamente uma vez. M2/M3 cortam potência negativa em zero e
  sinalizam o corte. Huld nunca avalia log(0); coeficientes CSi CURRENT estão congelados no código.
- Ensaio A: Tmod=Tbom medida; Tcell=Tbom+Gpoa/1000×ΔT. ΔT=3 °C é hipótese, com análise em 0/2/3/5 °C.
- Ensaio B: Faiman usa U0=29,84 W/(m² K), U1=3,44 W s/(m³ K), Tamb e vento da fonte. É outro protocolo.
- Na identificação de laboratório, a temperatura reportada é tratada como Tcell do ensaio em
  equilíbrio térmico. Não se aplica o aquecimento célula–traseira de campo ao simulador flash.
  Esta é uma hipótese explícita, não uma medição adicional.
- SDM multiponto: cinco parâmetros efetivos, resíduos normalizados por Pstc/Isc/Voc, três arranques
  deterministas, cotas positivas, relatório do Jacobiano e resíduos por grandeza. Não há ajuste no campo.
- Conserva-se a equação Lambert W e a translação De Soto do legado. A implementação em lotes usa
  `wrightomega(log_z)=W(exp(log_z))`, evitando a aproximação assintótica do piloto para argumentos grandes.
  Voc usa bisseção; MPP usa busca áurea acotada em [0,Voc], tolerância absoluta em volts. A mudança
  algorítmica é explícita. Não se alega que seus tempos sejam os do solver escalar original.
- Todos os modelos são avaliados como lotes NumPy, no mesmo modo vetorizado. Uma referência
  independente resolve a equação implícita em I com Brent e maximiza P com `minimize_scalar`.
  Convergência, resíduos da equação e concordância numérica são testados; falhas produzem NaN e status.
- O limite retangular de laboratório 100–1100 W/m² e 15–75 °C é uma indicação de extrapolação;
  a grade não é cheia e estar dentro do retângulo não significa ter uma medição naquela condição.

## Medição computacional

Cada par modelo–rodada tem um processo novo. Não há paralelismo entre modelos. O processo lê os
vetores e parâmetros, aquece uma vez e mede somente chamadas de `predict`. Os modelos rápidos
usam repetições internas de um mesmo lote, fixadas pela primeira rodada; tempo total do lote e
número de repetições são exportados. A rotação de ordem controla parcialmente efeitos de posição.
Com 11 rodadas, a contagem por posição não é perfeitamente balanceada; os registros permitem inspecionar isso.

RSS é amostrada pelo próprio worker, em thread auxiliar, após warmup; baseline, pico absoluto,
delta, intervalo e contagem de amostras são registrados. Em Linux lê-se `/proc/self/statm`, inclusive
em namespaces em que getpid e a montagem /proc discordam. Em Windows usa-se psutil.Process.
O pico amostrado pode perder eventos curtos. Importações e bibliotecas entram na RSS absoluta.
Tempo da thread de monitoramento e do loop de chamadas é parte da política comum de medição.
CPU compartilhada, frequência e carga externa não são controladas fisicamente neste ambiente.

## Limites de interpretação

A identificação não é necessariamente única. O Jacobiano é local e depende das coordenadas escaladas.
Não foi propagada a incerteza instrumental da caracterização até os erros de campo. Efeitos espectrais,
AOI, sujeira e envelhecimento não foram ajustados para nenhum modelo. Não há conversão AC, MPPT,
armazenamento, DDM ou estimativa de energia consumida pela CPU. Resultados são diagnósticos verificáveis,
pendentes de revisão científica antes do artigo.

## Fontes primárias consultadas

- JRC, [Data sources & calculation methods](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/general-information/data-sources-calculation-methods_en), §§5.2.3–5.2.4: coeficientes CURRENT e equação Faiman. Consulta 19/10/2026.
- pvlib, [PVWatts DC](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html): convenção de irradiância efetiva e temperatura de célula.
- SciPy, [Wright Omega](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.wrightomega.html): relação com Lambert W.
- IEA PVPS T13-20:2020, cópia integral incluída, pp. 99, 164–165: relação célula–traseira e descrição do ensaio SUPSI.
- A URL Sandia indicada no handoff não respondeu à consulta; a auditoria dos dados se apoiou nos arquivos locais e no relatório fornecido, sem substituir a fonte por outro conjunto.
