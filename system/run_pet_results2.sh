#!/usr/bin/env bash
# Run from any directory under WSL: bash /mnt/e/Lacis-dev/FOUL/system/run_pet_results2.sh
set -euo pipefail
cd "$(dirname "$0")"

RESULTS_DIR="../results-2"
DATASET_ROOT="../dataset/OxfordPet20"
SEED="${SEED:-1}"
TRAIN_ROUNDS="${TRAIN_ROUNDS:-100}"
TREATMENT_ROUNDS="${TREATMENT_ROUNDS:-30}"
TARGET="Chihuahua"
mkdir -p "$RESULTS_DIR"

common=(-data OxfordPet -algo FOUL -nc 20 -ls 2 -lbs 32 -dev cpu
        -fb "$TARGET" --dataset-root "$DATASET_ROOT" --pet-backbone cnn
        --pet-validation-fraction 0.2 --pet-split-seed 1 --pet-augment
        --pet-forget-scope attribute --pet-pruning-strategy discriminative
        --seed "$SEED" --results-dir "$RESULTS_DIR" -t 1)
train_goal="cnn_train_s${SEED}"
reference_goal="cnn_reference_s${SEED}"
checkpoint="$RESULTS_DIR/models/OxfordPet/FOUL_${train_goal}_0_trained.pt"
reference="$RESULTS_DIR/models/OxfordPet/FOUL_${reference_goal}_0_retrain_reference.pt"

echo "Training CNN, seed $SEED"
if [[ ! -f "$checkpoint" ]]; then
  uv run -- python main.py "${common[@]}" -lm learn -lmr "$TRAIN_ROUNDS" -gr "$TRAIN_ROUNDS" -go "$train_goal"
else
  echo "Reusing baseline checkpoint: $checkpoint"
fi

# Do not spend hours on a reference and ablations when forgetting cannot be measured.
uv run -- python - "$RESULTS_DIR/OxfordPet_FOUL_${train_goal}_0_hierarchy.json" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1], encoding='utf-8'))['metrics']
final = next(row for row in reversed(metrics) if row['stage'] == 'after_learning')
accuracy = final['target_breed_accuracy']
if accuracy is None or accuracy < 0.30:
    raise SystemExit(
        f"Baseline Chihuahua validation accuracy is {accuracy}; at least 0.30 is required. "
        "Improve training before running the reference and ablations."
    )
print(f"Baseline Chihuahua validation accuracy: {accuracy:.1%}")
PY

echo "Training retain-only retraining reference"
if [[ ! -f "$reference" ]]; then
  uv run -- python main.py "${common[@]}" -lm learn -lmr "$TRAIN_ROUNDS" -gr "$TRAIN_ROUNDS" \
    --pet-retrain-reference -go "$reference_goal"
else
  echo "Reusing reference checkpoint: $reference"
fi

run_treatment() {
  local goal="$1"
  shift
  local metrics="$RESULTS_DIR/OxfordPet_FOUL_${goal}_0_hierarchy.json"
  if [[ ! -f "$metrics" ]]; then
    echo "Running $goal"
    uv run -- python main.py "${common[@]}" -lm unlearn -lmr 0 -gr "$TREATMENT_ROUNDS" \
      --pet-checkpoint "$checkpoint" -go "$goal" "$@"
  else
    echo "Reusing treatment results: $metrics"
  fi
  uv run -- python plot_pet_results.py \
    "$metrics" \
    --baseline-json "$RESULTS_DIR/OxfordPet_FOUL_${train_goal}_0_hierarchy.json" \
    --output-dir "$RESULTS_DIR/plots/$goal"
}

run_treatment "cnn_foul_s${SEED}" --pruning-rate 0 --pet-unlearning-method foul
run_treatment "cnn_p25_s${SEED}" --pruning-rate 0.25 --pet-pruning-only
run_treatment "cnn_p25_retain_s${SEED}" --pruning-rate 0.25 --pet-unlearning-method retain
run_treatment "cnn_random25_foul_s${SEED}" --pruning-rate 0.25 --pet-pruning-strategy random
run_treatment "cnn_retain_s${SEED}" --pruning-rate 0 --pet-unlearning-method retain
run_treatment "cnn_p25_foul_s${SEED}" --pruning-rate 0.25 --pet-unlearning-method foul

uv run -- python plot_pet_results.py \
  "$RESULTS_DIR/OxfordPet_FOUL_${train_goal}_0_hierarchy.json" \
  --output-dir "$RESULTS_DIR/plots/$train_goal"
uv run -- python plot_pet_results.py \
  "$RESULTS_DIR/OxfordPet_FOUL_${reference_goal}_0_hierarchy.json" \
  --output-dir "$RESULTS_DIR/plots/$reference_goal"

uv run -- python evaluate_pet_representations.py \
  "$checkpoint" \
  "$RESULTS_DIR/models/OxfordPet/FOUL_cnn_p25_foul_s${SEED}_0_pruning_unlearning.pt" \
  "$RESULTS_DIR/models/OxfordPet/FOUL_cnn_p25_retain_s${SEED}_0_pruning_unlearning.pt" \
  "$reference" \
  --dataset-root "$DATASET_ROOT" --validation-fraction 0.2 --split-seed 1 \
  --target "$TARGET" --seed "$SEED" --relearn-steps 30 \
  --output "$RESULTS_DIR/cnn_diagnostics_s${SEED}.json"

echo "Completed: $RESULTS_DIR"
