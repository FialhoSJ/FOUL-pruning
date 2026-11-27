# Fine-Grained Federated Unlearning for Pet Breeds

**Approved**: 2026-11-25 - user: "aprovo"

## Problem Statement

The current FOUL workflow uses one label per image and one classification head, so it cannot explicitly represent a parent class and a breed inside that class. This feature will use the annotated Oxford-IIIT Pet dataset first, treating family (cat/dog) as the parent task and breed as the subclass task. The dataset interface will also support Stanford Dogs as a later dog-only dataset, while CLIP remains an optional comparison for selecting examples.

## Goals

- [ ] Select a real annotated breed as the forget target while retaining the parent family prediction.
- [ ] Apply class-discriminative structured pruning at 50% before FOUL and keep the compact model.
- [ ] Run the first experiment on Oxford-IIIT Pet and provide the same parent/subclass data interface for Stanford Dogs.
- [ ] Report target-breed, retained-breed, and parent-task results separately.

## Out of Scope

| Feature | Reason |
| --- | --- |
| PACS label correction or PACS data regeneration | The first experiment now uses Oxford-IIIT Pet; PACS is not required to test an annotated breed target. |
| Treating CLIP clusters as ground-truth breeds | Breed annotations are the authoritative target labels; CLIP is only an optional comparison. |
| Unlearning the CLIP model | CLIP is only used for optional text-image similarity and selection comparison. |
| Automatically downloading Stanford Dogs in the first experiment | Stanford is the next dataset to test; the first run should use Oxford-IIIT Pet. |

---

## Assumptions & Open Questions

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| First dataset | Oxford-IIIT Pet; use its breed labels and family labels. | It explicitly annotates 25 dog breeds and 12 cat breeds, enabling a parent/child task. | y - user selected Oxford first |
| Future dataset | Prepare a shared loader interface for Stanford Dogs; all samples have parent `dog` and one of 120 breed labels. | Stanford is dog-only, so the parent label is constant and parent accuracy is not a meaningful metric there. | y - user selected Stanford next |
| Target selector | Select the target by the dataset's real breed label. CLIP-based selection is optional and reports agreement with the annotated labels. | The user chose real breed labels with optional CLIP. | y - user selected real label; CLIP optional |
| Oxford split | Combine official train and validation images for federated training; retain the official test split. | This provides more training examples per breed while preserving the official test images. | n - agent discretion |
| Classification structure | Shared feature extractor with a family head and a breed head. On Stanford, use the breed head; the parent is `dog`. | This represents the hierarchy and lets evaluation check family retention separately from breed forgetting. | n - agent discretion |
| Pruning fraction and order | Before FOUL, remove 50% of highest-ranked target-discriminative eligible channels and keep the compact architecture. | User selected 50%, class-discriminative pruning, pre-unlearning, and a compact final model. | y - user selected |
| Image preprocessing | Use deterministic resizing/cropping suitable for the existing CNN pipeline; preserve a configurable image size. | A shared preprocessing contract is needed for both loaders and federated clients. | n - agent discretion |

**Open questions:** none. The remaining defaults above are implementation choices; the spec awaits approval.

---

## User Stories

### P1: Represent parent categories and annotated breeds ⭐ MVP

**User Story**: As a researcher, I want each pet image to carry both its family and breed labels so that I can forget one breed while measuring whether the model still recognizes cats and dogs.

**Why P1**: Explicit hierarchy is required to distinguish subclass unlearning from forgetting an entire output class.

**Acceptance Criteria**:

1. [DATA-01] WHEN Oxford-IIIT Pet data is loaded THEN the system SHALL expose a stable parent label (`cat` or `dog`) and the original breed label for every image.
2. [DATA-02] WHEN Stanford Dogs data is loaded THEN the system SHALL expose parent label `dog` and a stable breed label for every image through the same dataset interface.
3. [DATA-03] WHEN the Oxford dataset is prepared THEN the system SHALL use the official training and validation images for federated training and reserve the official test images for evaluation.
4. [DATA-04] IF a sample has an unknown family or breed label THEN the system SHALL stop with an error identifying the sample and invalid label before writing client partitions.

**Independent Test**: Load small Oxford and Stanford fixtures and verify the parent/breed labels and stable mappings.

### P1: Select and forget an annotated breed ⭐ MVP

**User Story**: As a researcher, I want to select a breed such as Chihuahua by its annotation so that the forget set corresponds to a real named subclass.

**Why P1**: Annotated membership makes it possible to measure forgetting of a specific breed rather than an approximate visual cluster.

**Acceptance Criteria**:

1. [SEL-01] WHEN an experiment specifies a valid breed name or ID THEN the system SHALL select all local training examples whose annotated breed matches that target, regardless of client ID.
2. [SEL-02] WHEN a client contains both target-breed and other examples THEN the system SHALL form separate forget and retain groups by breed membership.
3. [SEL-03] IF the requested breed is absent from the dataset mapping or training split THEN the system SHALL stop with an actionable error before modifying a checkpoint.
4. WHERE optional CLIP comparison is enabled, WHEN a text prompt is provided THEN the system SHALL compare CLIP-selected examples or clusters with the annotated breed labels and report selection agreement without replacing the ground-truth target set.

**Independent Test**: Use a fixture with one target breed split across multiple clients and verify that only matching breed samples are selected.

### P1: Preserve family knowledge while unlearning a breed ⭐ MVP

**User Story**: As a researcher, I want the model to forget a target breed while retaining family recognition and performance on other breeds.

**Why P1**: This is the measurable distinction between forgetting a subclass and damaging the entire parent class.

**Acceptance Criteria**:

1. [MOD-01] WHEN the model is initialized for Oxford-IIIT Pet THEN it SHALL expose a family prediction head and a breed prediction head over a shared feature extractor.
2. [MOD-02] WHEN the model is initialized for Stanford Dogs THEN it SHALL use the same breed-output contract and map every example to parent `dog`.
3. [FUL-01] WHEN FOUL unlearning starts for a breed THEN the server SHALL aggregate forget and retain updates according to sample membership within each client, not by client ID.
4. [FUL-02] WHEN the target breed is unlearned THEN target-breed samples SHALL remain available to the family-retention objective while their breed-specific contribution is treated as forget data.
5. [FUL-03] WHEN a client has no target or no retain samples THEN the system SHALL omit that empty group for the round without aborting training.
6. [MET-01] WHEN evaluation runs on Oxford-IIIT Pet THEN the system SHALL report family accuracy, target-breed accuracy, and non-target breed accuracy separately.
7. [MET-02] WHEN evaluation runs on Stanford Dogs THEN the system SHALL report target-breed and non-target breed metrics and SHALL omit parent accuracy because all parent labels are `dog`.

**Independent Test**: Evaluate a fixture containing target-breed and non-target examples and verify separate family and breed metrics.

### P1: Prune before unlearning and keep the model compact ⭐ MVP

**User Story**: As a researcher, I want target-discriminative channels removed before FOUL so that the unlearning stage runs on a smaller model.

**Why P1**: The user selected class-discriminative pruning as the compute-reduction step and wants to retain the compact architecture.

**Acceptance Criteria**:

1. [PRN-01] WHEN a client prepares for pruning THEN it SHALL calculate channel activation summaries for target-breed and retained examples without uploading raw images.
2. [PRN-02] WHEN the server receives activation summaries THEN it SHALL rank eligible channels using the selected paper's class-discriminative, TF-IDF-style procedure.
3. [PRN-03] WHEN pruning is applied THEN the system SHALL physically remove 50% of eligible target-discriminative channels per eligible feature layer while preserving at least one channel per layer and preserving both head output dimensions.
4. [PRN-04] WHEN the compact architecture is built THEN server and client models SHALL use the same channel structure and reset optimizer state before FOUL unlearning.
5. [PRN-05] WHEN FOUL unlearning finishes THEN the system SHALL save the compact model and report kept/removed channel counts by layer.
6. [PRN-06] IF a layer cannot be rebuilt or the compact model has incompatible output shapes THEN the system SHALL stop before overwriting the trained checkpoint.

**Independent Test**: Prune a small multi-head model with fixed activation scores and verify channel counts, output shapes, and server/client architecture agreement.

---

## Edge Cases

- IF the CLIP prompt has tied similarity to multiple clusters THEN the optional comparison SHALL use the lowest cluster ID as a deterministic tie-breaker.
- IF a target breed has too few training examples to form non-empty forget groups in the configured client partition THEN the system SHALL report the per-client counts before unlearning and allow the experiment to stop without changing the checkpoint.
- IF the model's family head has only one label, as in Stanford Dogs, THEN family accuracy SHALL be omitted rather than reported as a misleading constant.
- IF pruning or unlearning fails before the compact checkpoint is complete THEN the prior trained checkpoint SHALL remain intact.

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| --- | --- | --- | --- |
| DATA-01 | Represent parent categories and annotated breeds | Design | Pending |
| DATA-02 | Represent parent categories and annotated breeds | Design | Pending |
| DATA-03 | Represent parent categories and annotated breeds | Design | Pending |
| DATA-04 | Represent parent categories and annotated breeds | Design | Pending |
| SEL-01 | Select and forget an annotated breed | Design | Pending |
| SEL-02 | Select and forget an annotated breed | Design | Pending |
| SEL-03 | Select and forget an annotated breed | Design | Pending |
| MOD-01 | Preserve family knowledge | Design | Pending |
| MOD-02 | Preserve family knowledge | Design | Pending |
| FUL-01 | Preserve family knowledge | Design | Pending |
| FUL-02 | Preserve family knowledge | Design | Pending |
| FUL-03 | Preserve family knowledge | Design | Pending |
| MET-01 | Preserve family knowledge | Design | Pending |
| MET-02 | Preserve family knowledge | Design | Pending |
| PRN-01 | Prune before unlearning | Design | Pending |
| PRN-02 | Prune before unlearning | Design | Pending |
| PRN-03 | Prune before unlearning | Design | Pending |
| PRN-04 | Prune before unlearning | Design | Pending |
| PRN-05 | Prune before unlearning | Design | Pending |
| PRN-06 | Prune before unlearning | Design | Pending |

**Coverage**: 20 requirements, all mapped to a story.

## Differences from the earlier PACS/CLIP proposal

| Earlier proposal | Updated proposal | Reason |
| --- | --- | --- |
| PACS visual clusters were the target, with CLIP text selecting a cluster. | Oxford breed annotations define the target; CLIP is an optional comparison. | User selected an annotated breed dataset and confirmed real breed labels with optional CLIP. |
| PACS labels would be corrected and regenerated. | Oxford-IIIT Pet is used first; Stanford Dogs is supported through a shared parent/breed interface. | User selected Oxford first and Stanford next. |
| A single classification output was assumed. | Use family and breed heads for Oxford; Stanford uses breed labels under parent `dog`. | This makes parent retention and subclass forgetting measurable. |

## Sources

- [Oxford-IIIT Pet original paper](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/ukieri/files_pubs/parkhi12a.pdf) - 7,349 images, 25 dog breeds, 12 cat breeds, family and breed annotations, and official splits.
- [Stanford Dogs original paper](https://www-cs.stanford.edu/groups/vision/documents/KhoslaJayadevaprakashYaoFeiFei_FGVC2011.pdf) - fine-grained dog-breed classification dataset including Chihuahua.
- [Federated Unlearning via Class-Discriminative Pruning](https://arxiv.org/abs/2110.11794) - federated channel scoring and structured pruning reference.
- [PUMA pruning implementation supplied by the user](https://github.com/VeigarGit/FL_ServerPuma/blob/main/src/system/prunning.py) - reference for the requested 50% pruning rate; its model-specific reconstruction is not copied directly.
- [CLIP original paper](https://arxiv.org/abs/2103.00020) - reference for optional text-image similarity comparison.

## Success Criteria

- [ ] Oxford loader yields family and breed labels with stable IDs and preserves the official test set.
- [ ] Selecting a named breed selects its annotated examples across clients, independent of client IDs.
- [ ] After unlearning, target-breed recognition falls while family accuracy and non-target breed accuracy are reported separately.
- [ ] Pruning removes 50% of eligible target-discriminative channels, preserves both output heads, and leaves one compact model architecture shared by server and clients.
- [ ] Stanford Dogs can later be loaded through the same label interface without changing FOUL's target-selection contract.
