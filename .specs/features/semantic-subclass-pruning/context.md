# Fine-Grained Federated Unlearning for Pet Breeds Context

**Gathered:** 2026-11-25  
**Spec:** `.specs/features/semantic-subclass-pruning/spec.md`  
**Status:** Ready for specification review

## Feature Boundary

Use Oxford-IIIT Pet first to target a real annotated dog or cat breed, preserve parent-family recognition, prune 50% of target-discriminative eligible channels before FOUL, and keep the compact model. Prepare a shared parent/subclass label interface for Stanford Dogs as the next dataset. CLIP is an optional comparison against real breed annotations.

## Confirmed Decisions

- Start with Oxford-IIIT Pet.
- Keep Stanford Dogs as the next compatible dataset.
- Select the target using the real breed label; CLIP is optional.
- Use class-discriminative pruning at 50%, before FOUL, and keep the compact model.
- Measure target-breed forgetting separately from parent and non-target performance.

## Agent's Discretion

- Use a family head and a breed head for Oxford-IIIT Pet.
- Map Stanford Dogs to parent `dog` with breed labels and omit parent accuracy because the parent is constant.
- Combine Oxford official train and validation splits for federated training while holding the official test split out.
- Preserve deterministic label maps and image preprocessing across dataset adapters.

## Deferred Ideas

- PACS mapping correction and regeneration are not needed for the first Oxford experiment.
- CLIP-selected visual clusters are deferred to an optional comparison; they are not the authoritative forget set.
- Training or unlearning the CLIP model itself remains out of scope.
