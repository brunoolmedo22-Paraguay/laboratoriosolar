# Registro de desenvolvimento

Os diretórios de execução incluídos são preservados, com configurações e tentativas individuais.

- `20261019T141735_5ec8975f`: 44 tentativas falharam na leitura RSS (`NoSuchProcess`), antes de qualquer janela de compute. O status é `complete_with_failures`, não há previsões válidas, e não se devem interpretar zeros ou campos ausentes dessa primeira versão de exportação como estimativas físicas. Esse histórico documenta o defeito de infraestrutura corrigido.
- `20261019T142131_8364eee3`: primeira execução completa, após corrigir RSS para `/proc/self/statm`. 44 avaliações e 15.457 previsões comuns válidas. Mantida para rastreabilidade das etapas de desenvolvimento.
- A execução mais recente é a verificação final da versão entregue. Foram acrescentadas proteções de exportação por manifesto, bloqueio de concorrência com liberação pelo sistema operacional e separadores visuais em lacunas. Nenhuma dessas revisões foi um ajuste de parâmetros aos resultados de campo.

O primeiro smoke de janeiro também encontrou a falha RSS. A conclusão inicial de que ele havia terminado referia-se ao término da orquestração; a auditoria dos registros individuais revelou que não era um benchmark válido. Esse resultado não foi usado para inferência de desempenho.

Resultados científicos não foram selecionados pela obtenção de menores tempos. O relatório final identifica a execução final e seu ambiente. Comparações entre execuções de desenvolvimento devem considerar os hashes de código e as revisões descritas.
