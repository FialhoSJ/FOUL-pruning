# Protocolo de esquecimento seletivo de raça

Para executar automaticamente treino, referência, seis controles, gráficos e diagnósticos na pasta `results-2`, rode no WSL:

```bash
cd /mnt/e/Lacis-dev/FOUL/system
bash run_pet_results2.sh
```

O script aproveita checkpoints e resultados completos já salvos e para antes das ablações quando a acurácia de validação da raça alvo fica abaixo de 30%. Nesse caso, melhore a classificação inicial antes de comparar métodos de esquecimento.

O script usa 100 rodadas de treino e 30 de tratamento. Para repetir com outra seed, use `SEED=2 bash run_pet_results2.sh` ou `SEED=3 bash run_pet_results2.sh`. Para ajustar o orçamento, use `TRAIN_ROUNDS=150 TREATMENT_ROUNDS=50 bash run_pet_results2.sh`. Não execute o mesmo seed novamente sem preservar os arquivos anteriores, pois os nomes se repetem. Todos os JSON, H5, gráficos, diagnósticos e checkpoints do script ficam em `FOUL/results-2`. As execuções deste protocolo usam `--results-dir ../results-2`; comandos manuais sem essa opção continuam no diretório padrão.

Hipótese: modificar as representações compartilhadas e aplicar pruning estrutural pode aproximar o comportamento de uma referência retreinada sem supervisão da raça alvo, preservando a classificação de família e das demais raças, com menor custo de inferência.

O objetivo implementado é remover a supervisão da **raça**, preservando a supervisão cachorro/gato inclusive nas imagens alvo. Não corresponde à remoção integral dessas imagens do treinamento. A referência `--pet-retrain-reference` segue exatamente essa definição. Acurácia zero da raça não comprova apagamento de conhecimento.

## Experimento principal: CNN aleatória, todas as camadas treináveis

Execute no WSL, dentro de `FOUL/system`. Os checkpoints agora têm nomes por experimento; o antigo `FOUL_server.pt` permanece preservado. Os exemplos usam seed 1; repita com seeds 2 e 3 alterando `--seed`, `-go` e o caminho do checkpoint. Mantenha `--pet-split-seed 1` para comparar as mesmas partições.

```bash
cd /mnt/e/Lacis-dev/FOUL/system
uv run -- python main.py -data OxfordPet -algo FOUL -nc 20 -lm learn -lmr 100 -gr 100 -ls 2 -lbs 32 -dev cpu -fb Chihuahua --dataset-root ../dataset/OxfordPet20 --pet-backbone cnn --pet-validation-fraction 0.2 --pet-split-seed 1 --pet-augment --seed 1 -go cnn_train_s1
```

A validação estratificada por cliente é retirada somente da partição de treino. O teste oficial aparece apenas na avaliação final. Classes com um único exemplo local permanecem no treino. A fração real pode diferir de 20% devido ao arredondamento e à preservação de exemplos de treino. Para reproduzir execuções antigas, a fração padrão continua sendo zero (monitoramento no teste); use explicitamente 0.2 nas novas execuções.

Antes de unlearning, examine o desempenho na validação. Cem rodadas são um orçamento inicial, não garantia de convergência. Uma CNN aleatória pode ter desempenho inferior à ResNet pré-treinada. Não interprete esquecimento de uma raça que o modelo não aprendeu.

```bash
# Pruning discriminativo de 25% + FOUL, 30 atualizações.
uv run -- python main.py -data OxfordPet -algo FOUL -nc 20 -lm unlearn -lmr 0 -gr 30 -ls 2 -lbs 32 -dev cpu -fb Chihuahua --dataset-root ../dataset/OxfordPet20 --pet-backbone cnn --pet-validation-fraction 0.2 --pet-split-seed 1 --pet-augment --seed 1 --pet-checkpoint models/OxfordPet/FOUL_cnn_train_s1_0_trained.pt --pruning-rate 0.25 --pet-unlearning-method foul -go cnn_p25_foul_s1
```

Execute as seguintes ablações a partir desse mesmo checkpoint, trocando opções e `-go` no comando acima:

| Comparação | Opções | `-go` |
|---|---|---|
| FOUL sem pruning | `--pruning-rate 0` | `cnn_foul_s1` |
| Apenas pruning | `--pruning-rate 0.25 --pet-pruning-only` | `cnn_p25_s1` |
| Pruning + retenção | `--pruning-rate 0.25 --pet-unlearning-method retain` | `cnn_p25_retain_s1` |
| Pruning aleatório + FOUL | `--pruning-rate 0.25 --pet-pruning-strategy random` | `cnn_random25_foul_s1` |
| Apenas retenção | `--pruning-rate 0 --pet-unlearning-method retain` | `cnn_retain_s1` |

Substitua opções existentes; não acrescente duas versões da mesma opção. A recuperação `retain` usa média das atualizações ponderada pelo número de exemplos locais e não calcula a atualização de esquecimento. Para pruning aleatório, não há coleta de ativações. A CNN reduz fisicamente conv1, conv2, conv3 e a projeção; a ResNet reduz somente a projeção, mesmo com encoder treinável.

Referência independente, com mesma inicialização e orçamento, preservando os rótulos de família das imagens alvo:

```bash
uv run -- python main.py -data OxfordPet -algo FOUL -nc 20 -lm learn -lmr 100 -gr 100 -ls 2 -lbs 32 -dev cpu -fb Chihuahua --dataset-root ../dataset/OxfordPet20 --pet-backbone cnn --pet-validation-fraction 0.2 --pet-split-seed 1 --pet-augment --seed 1 --pet-retrain-reference -go cnn_reference_s1
```

## ResNet como experimento adicional

No comando de treinamento, use `--pet-backbone resnet18 --pet-encoder-training full --pet-backbone-lr 0.00001` e outro `-go`. O padrão usa pesos ImageNet. No tratamento, use o checkpoint correspondente, `--pet-backbone resnet18 --pet-encoder-unlearning full`. A opção `preserve` mantém a política salva, e `frozen` reproduz o cenário de encoder congelado. Features em cache só são aceitas quando o encoder está congelado.

As estatísticas móveis de BatchNorm ficam fixas em todos os cenários ResNet para evitar deriva local sem agregação de buffers; os parâmetros afins e convoluções são treináveis com `full`. O pré-treinamento ImageNet pode já conter conhecimento sobre raças: a comparação mede remoção do efeito da supervisão específica deste experimento, não de todo o pré-treinamento.

## Diagnósticos adicionais

```bash
uv run -- python evaluate_pet_representations.py models/OxfordPet/FOUL_cnn_train_s1_0_trained.pt models/OxfordPet/FOUL_cnn_p25_foul_s1_0_pruning_unlearning.pt models/OxfordPet/FOUL_cnn_reference_s1_0_retrain_reference.pt --dataset-root ../dataset/OxfordPet20 --validation-fraction 0.2 --target Chihuahua --relearn-steps 30 --output ../results/cnn_diagnostics_s1.json
```

O probe linear mede separabilidade da raça alvo contra outras raças da mesma família; ajuste e padronização usam somente treino, avaliação usa teste. Na ResNet, mede projeção e encoder separadamente. A reaprendizagem é um diagnóstico centralizado com cópias em memória, mesmos exemplos rotulados por raça, ordem, SGD e orçamento; não altera os checkpoints. Compare as curvas com a referência, sem interpretar baixa separabilidade como prova de ausência de informação. Os probes usam supervisão da raça alvo deliberadamente para testar sua recuperabilidade.

São registrados latência de inferência com imagens reais (aquecimento e sincronização CUDA), parâmetros, bytes dos tensores e tamanho do arquivo. A latência inclui apenas forward do lote já no dispositivo, sem leitura de disco ou transferência. Execute no mesmo hardware, lote e condições de carga; não confunda redução de parâmetros treináveis com redução do modelo inteiro. Não há medição de pico de RAM neste script.

Os JSON de treinamento incluem split, matriz de confusão, confiança média no alvo, perda nas demais raças e acurácia por família/grupo. Compare métodos no mesmo split; resultados antigos sem holdout não são diretamente equivalentes aos novos.

Para o artigo, use média e dispersão de pelo menos três seeds, taxa de pruning escolhida na validação e reporte conjuntamente retenção, aproximação à referência, probes/reaprendizagem e custo. Os novos experimentos ainda precisam ser executados: a implementação e os testes não estabelecem a hipótese científica.
