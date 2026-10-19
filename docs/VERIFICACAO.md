# Verificação da entrega CIM26 Lab 1.0.0

Execução final: `20261019T143146_8d245537`.

## Resultado da aceitação

- 52 testes automatizados aprovados: fórmulas, unidades, leitor, matriz 22×3, fontes originais, 16 módulos comerciais, ajuste multiponto, referência numérica independente, qualidade, lacunas, bootstrap, concorrência, falhas, exportação e UI.
- Streamlit iniciado em subprocesso; endpoint de saúde HTTP 200. Os modos Modelos e Artigo, botão de calibração, simulação M4, ficha incompleta, histórico e quatro abas foram exercitados com Streamlit AppTest.
- 18.705 linhas de campo preservadas; 15.457 amostras elegíveis e válidas comuns. 44 avaliações principais (11 por modelo) sem falhas ou não convergências, todas deterministas.
- Comparação dos timestamps monotônicos confirma que as janelas de compute não se sobrepõem. Os 44 registros compartilham o mesmo hash dos vetores e timestamps.
- Saídas das 11 rodadas de cada modelo coincidem exatamente. Não foram promediadas para produzir falsa replicação física.
- 24 medições adicionais de escalabilidade: dois tamanhos × três rodadas × quatro modelos, separadas das 44 principais.
- Hashes da execução e métricas energéticas recalculadas dos CSV verificados. Exportação ZIP reaberta com a função física desabilitada no teste, confirmando que não há recálculo.
- Hashes dos arquivos do motor na execução final coincidem com os arquivos desta entrega.

## Diagnóstico de engenharia do protocolo A

Gpoa≥100 W/m², Tbom medida, ΔT=3 °C, um módulo SUPSI, janeiro–junho/2019. Integração em 1243.417 h de intervalos válidos; energia observada parcial 206.390872 kWh. Não representa toda a energia semestral.

| Modelo | RMSE (W) | MAE (W) | MBE (W) | Energia parcial (kWh) | Mediana (µs/amostra) | RSS pico mediano (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 24.520677 | 16.625313 | 11.034324 | 220.293897 | 0.010722 | 64.934 |
| M2 | 18.806599 | 9.123368 | -0.973583 | 205.018820 | 0.013703 | 64.977 |
| M3 | 18.951726 | 9.262297 | -2.014312 | 203.770203 | 0.022158 | 65.539 |
| M4 | 18.766672 | 9.116261 | 0.102370 | 206.380621 | 9.179125 | 66.199 |

O ganho de RMSE de M4 frente a M2 é pequeno em valor absoluto neste conjunto e protocolo. Os intervalos pareados por dia estão no arquivo statistics.csv. Isso não estabelece um vencedor universal, relevância prática, desempenho em outro clima ou viabilidade em microcontroladores.

## Ambiente e alcance da verificação

Sistema: Linux-6.18.55-x86_64-with-glibc2.39. Python: 3.12.14 (main, Aug 25 2026, 14:00:49) [Clang 22.1.3 ]. Processador informado: x86_64; o modelo comercial da CPU não foi estabelecido pelo ambiente. Threads por worker: 1.

RSS Linux lida por /proc/self/statm. O primeiro intento falhou com psutil por incompatibilidade de namespace de PID; está preservado no histórico e documentado em EXECUCOES_DE_DESENVOLVIMENTO.md. O adaptador Windows usa psutil e o bloqueio usa msvcrt; não foi executado em uma máquina Windows nesta sessão.

O servidor e a UI foram verificados com AppTest e HTTP. Não houve inspeção visual em navegador real: o runtime não forneceu Chromium utilizável. As figuras científicas PNG foram inspecionadas visualmente; curvas têm separadores em lacunas, e os SVG mantêm texto editável.

Tempos foram medidos em ambiente compartilhado. Pico de RAM é amostrado, não um máximo exato; deltas incluem alocações transitórias e podem variar com o gerenciador de memória. Temperatura e energia da CPU não foram medidas. Repita o protocolo na máquina de interesse antes de usar esses custos computacionais na pesquisa.

## Hipóteses ainda sujeitas a revisão científica

- Conversão campo Tbom→Tcell com ΔT assumido e sensibilidade 0/2/3/5 °C; não há medição direta de célula.
- Temperatura do laboratório tratada como Tcell de ensaio em equilíbrio; Ns/n/Vmp/Imp de fabricante SUPSI desconhecidos.
- Ajuste normalizado por escalas STC; incerteza instrumental não propagada e unicidade global não demonstrada.
- Irradiância POA tratada igualmente nos quatro modelos sem correções ópticas/espectrais específicas.
- Bootstrap por dias supõe blocos representativos; autocorrelação e sazonalidade podem limitar a inferência. Onze rodadas são poucas para intervalos de tempo muito estáveis.

As evidências completas são os CSV/JSON/NPZ, hashes, ambiente e logs de testes incluídos. Não foi redigido o artigo científico.
