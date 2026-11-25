# Fine-Grained Federated Unlearning for Pet Breeds Tasks

## Execution Protocol

Implement these tasks with the `tlc-spec-driven` skill. Make only the changes listed in each task and preserve unrelated existing work.

---

**Design**: `.specs/features/semantic-subclass-pruning/design.md`  
**Status**: Approved  
**Approved**: 2026-11-25 - user: "aprovo"  
**Build**: `python3 -m compileall -q dataset system`

## Test Coverage Matrix

> No test files, test runner, or repository testing/quality guideline were found. Do not add or run project tests under the current instruction. Use Python syntax compilation and source review as local checks.

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Hierarchical data and dataset generation | none | Source review of mappings, sample alignment, and official split handling; compile changed modules | `dataset/generate_pet_datasets.py`, `system/utils/hierarchical_data.py` | none |
| Model, client, server, and pruning | none | Source review of interfaces, dimensions, empty groups, and compact-model synchronization; compile changed modules | `system/flcore/**/*.py`, `system/main.py` | none |
| Optional CLIP selection | none | Source review of lazy dependency and non-CLIP fallback; compile changed module | `system/utils/clip_breed_compare.py` | none |

## Gate Check Commands

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | After each code task | `python3 -m py_compile <task's changed Python file>` |
| Full | Not configured; repository has no test suite | none |
| Build | At phase completion | `python3 -m compileall -q dataset system` |

## Execution Plan

### Phase 1: Dataset and model foundations

```text
T1 -> T2
T1 -> T3
T1 -> T4
T2 -> T4
```

### Phase 2: Client and unlearning behavior

```text
T3 -> T5
T1 -> T5
T3 -> T6
T5 -> T6
T5 -> T7
T6 -> T7
```

### Phase 3: CLI integration

```text
T2 -> T8
T3 -> T8
T4 -> T8
T7 -> T8
```

## Task Breakdown

### T1: Add hierarchical client-data reader

**What**: Add a reader for client images with aligned parent-family and subclass-breed labels.  
**Where**: `system/utils/hierarchical_data.py`  
**Depends on**: None  
**Reuses**: NPZ conventions in `system/utils/data_utils.py`  
**Requirement**: DATA-01, DATA-02, DATA-04

**Done when**:

- [ ] Loads `x`, `y`, and `y_subclass` into aligned image and integer-label tensors.
- [ ] Converts uint8 images to normalized float tensors.
- [ ] Missing fields and unknown IDs produce an error naming the source file/client and invalid value.
- [ ] Leaves generic single-label readers unchanged.

**Tests**: none  
**Gate**: quick

**Status**: Complete - `python3 -m py_compile system/utils/hierarchical_data.py` succeeded.

### T2: Generate Oxford and Stanford federated datasets

**What**: Add one dataset generator entry point with separate Oxford-IIIT Pet and Stanford Dogs adapters writing the shared hierarchy schema.  
**Where**: `dataset/generate_pet_datasets.py`  
**Depends on**: T1  
**Reuses**: `dataset/utils/dataset_utils.py` storage conventions  
**Requirement**: DATA-01, DATA-02, DATA-03, DATA-04

**Done when**:

- [ ] Oxford combines official train and validation images for federated training and reserves official test images.
- [ ] Stanford reads its local breed folders and official split lists, maps parent to `dog`, and preserves train/test membership.
- [ ] Both write stable label maps and aligned `x`, `y`, `y_subclass` arrays per client.
- [ ] Allocation is deterministic from a seed and reports breed counts per client.
- [ ] Missing source files or invalid breed labels fail before client partitions are written.

**Tests**: none  
**Gate**: quick

**Status**: Complete - `python3 -m py_compile dataset/generate_pet_datasets.py` succeeded.

### T3: Add compact hierarchical CNN

**What**: Implement one shared convolutional feature extractor with family and breed heads plus a feature-layer interface for pruning.  
**Where**: `system/flcore/trainmodel/hierarchical_cnn.py`  
**Depends on**: T1  
**Reuses**: Existing CNN patterns in `system/flcore/trainmodel/models.py`  
**Requirement**: MOD-01, MOD-02

**Done when**:

- [ ] Forward returns distinct family logits and breed logits sized from dataset config.
- [ ] Exposes eligible feature layers in a stable ordered interface.
- [ ] Supports the configured image size and preserves both output dimensions.

**Tests**: none  
**Gate**: quick

### T4: Add optional CLIP comparison with breed labels

**What**: Add lazy CLIP prompt comparison for annotated breed membership without changing the ground-truth forget set.  
**Where**: `system/utils/clip_breed_compare.py`  
**Depends on**: T1, T2  
**Reuses**: Hierarchical image and breed mappings  
**Requirement**: SEL-04

**Done when**:

- [ ] Imports CLIP dependencies only when comparison is requested.
- [ ] Returns prompt-selected examples or nearest cluster and reports agreement with annotated breed membership.
- [ ] Missing optional package or model weights gives an actionable error while label-based runs remain available.

**Tests**: none  
**Gate**: quick

### T5: Train and split sample-group updates in the FOUL client

**What**: Extend the FOUL client for two-head loss and per-client breed-based retain/forget updates.  
**Where**: `system/flcore/clients/clientfoul.py`  
**Depends on**: T1, T3  
**Reuses**: Existing FOUL local optimizer and round lifecycle  
**Requirement**: SEL-01, SEL-02, FUL-02, FUL-03, PRN-01

**Done when**:

- [ ] Normal training optimizes family and breed heads using their respective labels.
- [ ] During unlearning, every example contributes to family retention; non-target breeds contribute to breed retention; target-breed breed updates are returned as forget updates.
- [ ] Client activation summaries are class-wise and layer-wise, with no raw images in the result.
- [ ] Empty local target or retain groups return no update without aborting the client round.
- [ ] Existing single-label FOUL remains on its legacy path.

**Tests**: none  
**Gate**: quick

### T6: Add federated class-discriminative structural pruning

**What**: Implement FCUP activation aggregation, TF-IDF ranking, and physical model shrinking at the selected 50% rate.  
**Where**: `system/flcore/servers/foul_pruning.py`  
**Depends on**: T3, T5  
**Reuses**: Feature-layer interface from T3 and client summaries from T5  
**Requirement**: PRN-01, PRN-02, PRN-03, PRN-06

**Done when**:

- [ ] Averages per-class activation summaries and applies FCUP's TF-IDF score to rank target-discriminative channels.
- [ ] Physically removes the top 50% of eligible channels per feature layer, keeping at least one channel in each layer.
- [ ] Slices downstream convolutions, flattened inputs, and both head inputs consistently while preserving output dimensions.
- [ ] Rejects incompatible shapes before returning the compact model.

**Tests**: none  
**Gate**: quick

### T7: Integrate sample-group FOUL aggregation and metrics

**What**: Add hierarchical unlearning rounds, pruning/synchronization, checkpoint saving, and parent/subclass metrics while preserving legacy FOUL behavior.  
**Where**: `system/flcore/servers/serverfoul.py`  
**Depends on**: T5, T6  
**Reuses**: Existing `foul_update` and FOUL round lifecycle  
**Requirement**: FUL-01, FUL-03, MET-01, MET-02, PRN-04, PRN-05

**Done when**:

- [ ] Aggregates client retain/forget update rows by sample group rather than client ID.
- [ ] Omits empty local groups and stops if the global target or retain group is empty.
- [ ] Applies pruning before FOUL, synchronizes compact architecture to server and clients, and resets client optimizers.
- [ ] Reports Oxford family, target-breed, and non-target-breed accuracy; Stanford reports target and non-target breeds without constant-parent accuracy.
- [ ] Saves compact checkpoint and per-layer counts only after a successful shape check.
- [ ] Legacy non-hierarchical FOUL behavior remains on its current path.

**Tests**: none  
**Gate**: quick

### T8: Wire dataset and target options into the CLI

**What**: Add Oxford/Stanford dataset selection, breed target, pruning amount, image size, and optional CLIP comparison arguments.  
**Where**: `system/main.py`  
**Depends on**: T2, T3, T4, T7  
**Reuses**: Existing parser and `run(args)` setup  
**Requirement**: SEL-03, PRN-04

**Done when**:

- [ ] Dataset selection sets correct label counts and the hierarchy-aware FOUL model without wrapping it in `BaseHeadSplit`.
- [ ] CLI accepts a breed name or ID, defaults pruning to 0.5, and offers optional CLIP comparison.
- [ ] Invalid breed fails before checkpoint changes.
- [ ] Existing PACS and other dataset commands retain their current route.

**Tests**: none  
**Gate**: build

## Checks

**Build**: `python3 -m compileall -q dataset system`

**C1** - Reader preserves aligned family/breed labels and validates the schema (DATA-01, DATA-02, DATA-04)  
Proof: manual source review after T1; compile gate

**C2** - Both generators preserve dataset mappings and splits in the shared schema (DATA-01, DATA-02, DATA-03, DATA-04)  
Proof: manual source review after T2; compile gate

**C3** - Model returns distinct family/breed logits with stable configured dimensions (MOD-01, MOD-02)  
Proof: manual source review after T3; compile gate

**C4** - Optional CLIP comparison reports agreement and is not required for label-based runs (SEL-04)  
Proof: manual source review after T4; compile gate

**C5** - Client trains both tasks and splits updates by target breed within clients (SEL-01, SEL-02, FUL-02, FUL-03, PRN-01)  
Proof: manual source review after T5; compile gate

**C6** - FCUP scoring and 50% structural pruning preserve dependent tensor shapes (PRN-02, PRN-03, PRN-06)  
Proof: manual source review after T6; compile gate

**C7** - Server aggregates sample groups, synchronizes compact models, and reports/save required metrics (FUL-01, FUL-03, MET-01, MET-02, PRN-04, PRN-05)  
Proof: manual source review after T7; compile gate

**C8** - CLI routes both datasets and validates breed target before checkpoint modification (SEL-03, PRN-04)  
Proof: manual source review after T8; compile gate

## Handoff

**Settled mid-build**: none
