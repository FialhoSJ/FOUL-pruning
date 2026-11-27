#!/usr/bin/env bash
# Tune FOUL strength against the validation-only retraining reference.
set -euo pipefail
cd "$(dirname "$0")"

RESULTS_DIR="../results-2"
SEED="${SEED:-1}"
CHECKPOINT="$RESULTS_DIR/models/OxfordPet/FOUL_resnet_train_s${SEED}_0_trained.pt"
REFERENCE="$RESULTS_DIR/OxfordPet_FOUL_resnet_examples_reference_s${SEED}_0_hierarchy.json"
[[ -f "$CHECKPOINT" ]] || { echo "Missing baseline: $CHECKPOINT" >&2; exit 1; }
[[ -f "$REFERENCE" ]] || { echo "Missing retraining reference: $REFERENCE" >&2; exit 1; }

common=(-data OxfordPet -algo FOUL -nc 20 -lm unlearn -lmr 0 -gr 30
        -ls 2 -lbs 32 -dev cpu -fb Chihuahua
        --dataset-root ../dataset/OxfordPet20 --pet-backbone resnet18
        --pet-validation-fraction 0.2 --pet-split-seed 1 --seed "$SEED"
        --results-dir "$RESULTS_DIR" --pet-checkpoint "$CHECKPOINT"
        --pet-forget-scope examples --pet-pruning-strategy magnitude
        --pruning-rate 0.25 --pet-encoder-unlearning full
        --pet-unlearning-method foul --pet-skip-final-test)

for beta in 1 2 4; do
  goal="resnet_examples_magnitude_foul_beta${beta}_s${SEED}"
  result="$RESULTS_DIR/OxfordPet_FOUL_${goal}_0_hierarchy.json"
  if [[ -f "$result" ]]; then
    echo "Already completed: $goal"
  else
    echo "Running FOUL beta=$beta"
    uv run -- python main.py "${common[@]}" -beta_foul "$beta" -go "$goal"
  fi
done

uv run -- python select_pet_beta.py --results-dir "$RESULTS_DIR" --seed "$SEED"
selection="$RESULTS_DIR/comparisons/beta_selection_s${SEED}.json"
selected_goal="$(uv run -- python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["goal"])' "$selection")"
selected_checkpoint="$RESULTS_DIR/models/OxfordPet/FOUL_${selected_goal}_0_pruning_unlearning.pt"
test_output="$RESULTS_DIR/comparisons/beta_selected_test_s${SEED}.json"
if [[ -f "$test_output" ]]; then
  echo "Already evaluated selected checkpoint: $test_output"
else
  uv run -- python evaluate_selected_pet.py "$selected_checkpoint" \
    --dataset-root ../dataset/OxfordPet20 --target Chihuahua \
    --output "$test_output"
fi
uv run -- python plot_selected_pet_breeds.py --results-dir "$RESULTS_DIR" --seed "$SEED"
echo "Validation selection: $selection"
echo "Selected official test: $test_output"
