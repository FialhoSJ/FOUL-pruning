# Protocolo de remoção integral de Chihuahua

## Comparação controlada recomendada

Com o baseline e a referência sem Chihuahua já presentes em `results-2`, execute no WSL:

```bash
cd /mnt/e/Lacis-dev/FOUL/system
bash run_pet_ablation.sh
```

O script executa 30 rodadas por tratamento, sempre a partir do mesmo checkpoint inicial e com a mesma partição. Ele completa as comparações faltantes: FOUL sem poda, apenas retenção sem poda, poda isolada e poda seguida apenas de retenção. Reaproveita a execução já concluída de poda + FOUL. Ao terminar, grava CSV e gráficos em `results-2/comparisons/` e executa probes de representação e uma prova de reaprendizagem em cópias dos modelos. Para adiar esse diagnóstico mais pesado, use `RUN_PROBES=0 bash run_pet_ablation.sh`. Cada rodada com o encoder inteiro treinável é custosa no CPU; espere várias horas para a bateria completa.

A comparação agora gera `breed_accuracy_s1.png` e `chihuahua_predictions_s1.png` (também PDF e CSV). O primeiro mostra a acurácia no teste para todas as 37 raças, permitindo ver se as demais continuam funcionando. O segundo mostra quais raças foram previstas para as 100 imagens de Chihuahua. A coluna da raça alvo aparece destacada em vermelho. Para reconstruir os gráficos de resultados já concluídos, rode `uv run -- python compare_pet_ablations.py --results-dir ../results-2 --seed 1` dentro de `FOUL/system`.

O JSON de cada tratamento agora inclui `update_diagnostics`: norma da atualização do servidor e alinhamento com as atualizações locais de retenção e esquecimento. Um alinhamento positivo com o passo de esquecimento indica que parte do efeito deste passo permanece na direção final; examine isso antes de aumentar `beta_foul`. A fórmula FOUL foi mantida para comparabilidade. A seleção de configurações deve usar validação; o teste oficial fica para a comparação final.

Este protocolo define `D_forget` como **todos os exemplos de Chihuahua do treino**. As atualizações de retenção e a referência retreinada excluem esses exemplos tanto da perda de raça quanto da perda cachorro/gato. O ramo de esquecimento do FOUL usa esses exemplos para calcular a atualização que é subtraída. O teste continua contendo Chihuahua para medir o efeito. A classe “cachorro” continua no modelo; reconhecer um Chihuahua como cachorro depois do tratamento não prova que seus exemplos permaneceram, pois outras raças ensinam características compartilhadas. Compare sempre com a referência sem nenhum exemplo de Chihuahua.

Os experimentos antigos de `results` e `results-2` anteriores a este protocolo usavam `attribute`: preservavam a supervisão de família das imagens alvo. Use novos nomes de `-go` para que resultados incompatíveis não se misturem. Os comandos abaixo partem do checkpoint treinado com validação separada, que classificou Chihuahua em 70% da validação e 69% do teste.

Execute de `FOUL/system`, no WSL. A referência custa aproximadamente o treino congelado; unlearning com encoder completo levou mais de uma hora no CPU no ensaio anterior.

```bash
cd /mnt/e/Lacis-dev/FOUL/system

# Referência: retreinar do início sem qualquer exemplo de Chihuahua.
uv run -- python main.py -data OxfordPet -algo FOUL -nc 20 -lm learn -lmr 50 -gr 50 -ls 2 -lbs 32 -dev cpu -fb Chihuahua --dataset-root ../dataset/OxfordPet20 --pet-backbone resnet18 --pet-validation-fraction 0.2 --pet-split-seed 1 --seed 1 --results-dir ../results-2 --pet-forget-scope examples --pet-retrain-reference -go resnet_examples_reference_s1

# FOUL sem compressão: mede o esquecimento sozinho.
uv run -- python main.py -data OxfordPet -algo FOUL -nc 20 -lm unlearn -lmr 0 -gr 30 -ls 2 -lbs 32 -dev cpu -fb Chihuahua --dataset-root ../dataset/OxfordPet20 --pet-backbone resnet18 --pet-validation-fraction 0.2 --pet-split-seed 1 --seed 1 --results-dir ../results-2 --pet-checkpoint ../results-2/models/OxfordPet/FOUL_resnet_train_s1_0_trained.pt --pet-forget-scope examples --pet-encoder-unlearning full --pruning-rate 0 -go resnet_examples_foul_s1

# Compressão por magnitude, medida isoladamente. Não usa rótulos Chihuahua.
uv run -- python main.py -data OxfordPet -algo FOUL -nc 20 -lm unlearn -lmr 0 -gr 30 -ls 2 -lbs 32 -dev cpu -fb Chihuahua --dataset-root ../dataset/OxfordPet20 --pet-backbone resnet18 --pet-validation-fraction 0.2 --pet-split-seed 1 --seed 1 --results-dir ../results-2 --pet-checkpoint ../results-2/models/OxfordPet/FOUL_resnet_train_s1_0_trained.pt --pet-forget-scope examples --pet-pruning-strategy magnitude --pruning-rate 0.25 --pet-pruning-only -go resnet_examples_magnitude_only_s1

# Mesma compressão, depois FOUL com encoder completo.
uv run -- python main.py -data OxfordPet -algo FOUL -nc 20 -lm unlearn -lmr 0 -gr 30 -ls 2 -lbs 32 -dev cpu -fb Chihuahua --dataset-root ../dataset/OxfordPet20 --pet-backbone resnet18 --pet-validation-fraction 0.2 --pet-split-seed 1 --seed 1 --results-dir ../results-2 --pet-checkpoint ../results-2/models/OxfordPet/FOUL_resnet_train_s1_0_trained.pt --pet-forget-scope examples --pet-pruning-strategy magnitude --pet-encoder-unlearning full --pruning-rate 0.25 -go resnet_examples_magnitude_foul_s1

# Controle: compressão + ajuste somente nos exemplos retidos.
uv run -- python main.py -data OxfordPet -algo FOUL -nc 20 -lm unlearn -lmr 0 -gr 30 -ls 2 -lbs 32 -dev cpu -fb Chihuahua --dataset-root ../dataset/OxfordPet20 --pet-backbone resnet18 --pet-validation-fraction 0.2 --pet-split-seed 1 --seed 1 --results-dir ../results-2 --pet-checkpoint ../results-2/models/OxfordPet/FOUL_resnet_train_s1_0_trained.pt --pet-forget-scope examples --pet-pruning-strategy magnitude --pet-encoder-unlearning full --pet-unlearning-method retain --pruning-rate 0.25 -go resnet_examples_magnitude_retain_s1
```

O JSON de cada execução registra `forget_scope`, política do encoder, estratégia de pruning, acurácia antes/depois da compressão e custo do modelo. Na ResNet atual, apenas a projeção é fisicamente reduzida; 25% de pruning nela removeu cerca de 0,31% dos parâmetros totais. Isso não demonstra uma grande economia de inferência. Para estudar compressão substancial, será necessária uma arquitetura cujas convoluções possam ser reduzidas estruturalmente com desempenho inicial forte.

Compare o modelo tratado com a referência em acurácia e confiança para Chihuahua, outras raças de cachorro e gatos; depois execute probes de representação e reaprendizagem com `evaluate_pet_representations.py`. Acurácia zero da raça sozinha continua insuficiente para concluir apagamento completo. Os pesos ImageNet também são uma fonte anterior de informação sobre cães; a comparação testa sobretudo a influência dos exemplos deste dataset.

## Ajuste da intensidade do FOUL com poda fixa

Com o baseline e a referência já salvos, execute no WSL:

```bash
cd /mnt/e/Lacis-dev/FOUL/system
bash run_pet_beta_sweep.sh
```

O script compara `beta_foul` igual a 1, 2 e 4, com 25% de poda por magnitude, 30 rodadas, mesmo checkpoint e mesma divisão de validação. As três execuções não avaliam o teste oficial. A seleção usa acurácia e confiança de Chihuahua, acurácia das raças preservadas, acurácia cachorro/gato e acurácia da família nas imagens alvo, comparando cada candidato à referência retreinada sem Chihuahua. A regra e os valores ficam em `results-2/comparisons/beta_selection_s1.json`.

Depois da seleção, o script avalia **somente o checkpoint escolhido** no teste e grava `beta_selected_test_s1.json`. Também cria `beta_selected_breeds_s1.png` e `beta_selected_chihuahua_predictions_s1.png`, com o desempenho de todas as raças e as previsões para imagens de Chihuahua. Cada candidato pode levar cerca de uma hora no CPU; a bateria pode durar várias horas. O script reaproveita candidatos já concluídos.

Como o teste da configuração anterior (`beta_foul=2`) já foi visto nesta pesquisa, esta varredura é exploratória. Para estimar generalização de forma mais forte no artigo, repita o protocolo completo com novas sementes e referência correspondente, mantendo a escolha dos hiperparâmetros separada do teste de cada repetição.
