#!/usr/bin/env bash
# Controlled comparisons of complete-example Chihuahua unlearning.
set -euo pipefail
cd "$(dirname "$0")"
RESULTS_DIR="../results-2"
SEED="${SEED:-1}"
ROUNDS=30
CHECKPOINT="$RESULTS_DIR/models/OxfordPet/FOUL_resnet_train_s${SEED}_0_trained.pt"
REFERENCE="$RESULTS_DIR/OxfordPet_FOUL_resnet_examples_reference_s${SEED}_0_hierarchy.json"
[[ -f "$CHECKPOINT" ]] || { echo "Missing baseline: $CHECKPOINT" >&2; exit 1; }
[[ -f "$REFERENCE" ]] || { echo "Missing retain-only retraining reference: $REFERENCE" >&2; exit 1; }

common=(-data OxfordPet -algo FOUL -nc 20 -lm unlearn -lmr 0 -gr "$ROUNDS"
        -ls 2 -lbs 32 -dev cpu -fb Chihuahua
        --dataset-root ../dataset/OxfordPet20 --pet-backbone resnet18
        --pet-validation-fraction 0.2 --pet-split-seed 1 --seed "$SEED"
        --results-dir "$RESULTS_DIR" --pet-checkpoint "$CHECKPOINT"
        --pet-forget-scope examples --pet-encoder-unlearning full)

run_case() {
  local goal="$1"
  shift
  local result="$RESULTS_DIR/OxfordPet_FOUL_${goal}_0_hierarchy.json"
  if [[ -f "$result" ]]; then
    echo "Already completed: $goal"
    return
  fi
  echo "Running: $goal"
  uv run -- python main.py "${common[@]}" -go "$goal" "$@"
}

# Run these from the same original checkpoint. Each changes one factor.
run_case "resnet_examples_noprune_foul_s${SEED}" --pruning-rate 0 --pet-unlearning-method foul
run_case "resnet_examples_noprune_retain_s${SEED}" --pruning-rate 0 --pet-unlearning-method retain
run_case "resnet_examples_magnitude_only_s${SEED}" --pruning-rate 0.25 --pet-pruning-strategy magnitude --pet-pruning-only
run_case "resnet_examples_magnitude_retain_s${SEED}" --pruning-rate 0.25 --pet-pruning-strategy magnitude --pet-unlearning-method retain
run_case "resnet_examples_magnitude_foul_s${SEED}" --pruning-rate 0.25 --pet-pruning-strategy magnitude --pet-unlearning-method foul

uv run -- python compare_pet_ablations.py --results-dir "$RESULTS_DIR" --seed "$SEED"

if [[ "${RUN_PROBES:-1}" == "1" ]]; then
  diagnostic="$RESULTS_DIR/comparisons/representation_relearning_s${SEED}.json"
  if [[ ! -f "$diagnostic" ]]; then
    uv run -- python evaluate_pet_representations.py \
      "$CHECKPOINT" \
      "$RESULTS_DIR/models/OxfordPet/FOUL_resnet_examples_reference_s${SEED}_0_retrain_reference.pt" \
      "$RESULTS_DIR/models/OxfordPet/FOUL_resnet_examples_noprune_foul_s${SEED}_0_unlearning_only.pt" \
      "$RESULTS_DIR/models/OxfordPet/FOUL_resnet_examples_magnitude_foul_s${SEED}_0_pruning_unlearning.pt" \
      --dataset-root ../dataset/OxfordPet20 --target Chihuahua \
      --validation-fraction 0.2 --split-seed 1 --seed "$SEED" \
      --relearn-steps 30 --output "$diagnostic"
  fi
fi
