# BVTS Migration Inventory

Date: 2025-11-14

This file captures an initial inventory of places in the codebase that create, transform, or assume tensor/field shapes. It was produced by a repository-wide search for common tensor-shape operations (permute, transpose, reshape/view, unsqueeze/squeeze, .shape usage) and for references to the field conversion utilities.

Purpose: use this as a reference when implementing the BVTS (batch, vector, time, spatial) canonical layout migration. Each entry lists the file, the operations found, and a short note about the shape assumptions or next inspection steps.

Status: As of this change the codebase has been migrated to BVTS as the canonical in-memory and on-disk layout. The old configuration flag `migration.enable_bvts` is no longer referenced in `src/` code. The only occurrences of `enable_bvts` found in the repository are historical Hydra run artifacts under `outputs/` and can be ignored or removed manually if desired. The checklist below remains useful for auditing caches, evaluator and visualization code.

---

## Summary (top-level)
- Files touching tensor layouts or shape logic were found primarily under `src/utils/field_conversion/` and `src/data/` plus evaluation code.
- Common operations: `permute`, `unsqueeze`/`squeeze`, `reshape`, `np.transpose`, `.shape` comparisons and indexing.
- Many converters assume a canonical layout already; others reorder to time-first or spatial-first layouts. We will need to unify these to BVTS and/or add adapters.

---

## Findings (by file)

### src/utils/field_conversion/layout.py
- Matches found: `permute(...)`, `unsqueeze(...)`, `squeeze(...)`, `reshape(...)`, many `.shape[...]` usages.
- Notes: This module performs explicit reorders and reshapes between different "native" and "canonical" layouts. Examples/comments indicate shapes such as `[T, *spatial]`, `[*spatial, V, T]`, and later reshaping into `(b * t, v, *spatial)`.
- Action: review this file first — it already centralises layout conversion and will be a key place to implement/route to BVTS helpers.

### src/utils/field_conversion/centered.py
- Matches found: references to `tensor.shape` with negative indices and comments indicating channel dim is `-3`.
- Notes: `CenteredConverter` appears to assume a particular layout: channel dim at -3. Needs careful reading to map its input/output layout contracts. Likely a primary conversion point between phi-field objects and tensors.
- Action: inspect its public methods `field_to_tensor` and `tensor_to_field` to understand current conventions and add BVTS-aware wrappers.

### src/utils/field_conversion/base.py
- Matches found: uses `field.shape.spatial.names` and similar metadata.
- Notes: uses phi-field metadata to determine spatial dims; helpful for preserving metadata during conversion.
- Action: ensure metadata mapping remains correct when moving to BVTS.

### src/utils/field_conversion/batch.py
- Matches found: checks `tensor.shape[channel_dim]`, validates channel counts.
- Notes: enforces expected channel (vector) counts by indexing into a specific channel dim. Channel dim indexing must be updated for BVTS canonical order.

### src/utils/field_conversion/factory.py
- Matches found: factory that returns `CenteredConverter` for centered fields.
- Notes: this is the instantiation point — good place to inject BVTS-aware converter or to centralize the change.

### src/utils/field_conversion/staggered.py
- Matches found: calls into `CenteredConverter` (staggered->centered, centered->staggered).
- Notes: relies on CenteredConverter contract; verify end-to-end conversion keeps correct layout.

### src/data/tensor_dataset.py
- Matches found: `native_tensor.permute(3, 2, 0, 1)  # -> [time, vector, x, y]` and `permute(2, 0, 1).unsqueeze(1)  # -> [time, 1, x, y]` and `squeeze(0)`.
- Notes: the dataset produces time-first tensors (time, vector, x, y). Many datasets may yield tensors without an explicit batch dimension (time-first or time-major). This will need an adapter that ensures returned tensors are BVTS with batch possibly added/expanded.
- Action: add adapters in dataset `__getitem__` and/or dataset wrapper to output BVTS (batch, vector, time, spatial).

### src/data/field_dataset.py
- Matches found: `unsqueeze(0)` to add batch dim in some code paths, and per-timestep indexing with `target_tensor[t].unsqueeze(0)`.
- Notes: some dataset code expects to operate without a batch dim and manually unsqueezes. Standardizing to always have batch dim will simplify migration.

### src/data/data_manager.py
- Matches found: `is_vector = centered.shape.channel.rank > 0`.
- Notes: code inspects phi-field metadata rather than raw tensor dims here; that's useful. Ensure conversion keeps phi metadata consistent.

### src/evaluation/visualizations.py
- Matches found: `np.transpose(..., (0,2,1))` in several places, converting `[T, H, W] -> [T, W, H]`.
- Notes: visualization code manipulates numpy arrays with hard-coded axis orders. These need review so the plotting receives BVTS-shaped data (probably slices over time and spatial dims).

### src/evaluation/metrics.py
- Matches found: `reshape` to flatten spatial dims and many `if prediction.shape != ground_truth.shape` checks.
- Notes: metric functions assert prediction and ground truth shapes match; converting both to BVTS will keep these checks valid, but the order of dims when flattening must be correct (vector/time/spatial axes accounted for).

### src/evaluation/evaluator.py
- Matches found: prints like `prediction_tensor.shape[0]` used as "frames" count.
- Notes: code uses shape[0] as number of frames in some contexts — after BVTS migration frame/time may be at dim index 2 (batch, vector, time, spatial). Update indexing or use helpers to query time dimension.

### src/models/synthetic/base.py
- Matches found: comments describing conversion of StaggeredGrids into multi-channel CenteredGrid tensor and back.
- Notes: models assume network input/output are CenteredGrid-based. Confirm model input expectations and adapt wrapper/decorator to accept BVTS tensors.

---

## Initial conclusions and priorities
1. `src/utils/field_conversion/layout.py` and `centered.py` are the canonical places where layout orders are handled — start here to implement BVTS-aware helpers or to plug in new `to_bvts` / `from_bvts` helpers.
2. Dataset code under `src/data/` often yields time-first tensors or missing batch dims; add small dataset adapters to guarantee returned tensors are BVTS (add batch dim if missing, move vector/time dims into canonical positions).
3. Evaluation and visualization code do axis manipulations with numpy — update these to use helpers or documented expected axis names instead of raw index tuples.
4. Model code should be wrapped with an input-compatibility decorator to accept either legacy or BVTS and convert internally until the migration is complete.

---

## Next inspection steps (recommended)
- Do a focused search for `torch.save`, `torch.load`, `.state_dict()`, `load_state_dict` to find checkpoint code that may save tensors in a specific order. Update or document expected checkpoint layout.
- Search for `torch.stack`, `torch.cat`, `torch.tensor([...])`, and places that construct tensors from lists — they can encode an ordering assumption.
- Inspect model forward methods directly under `src/models/*` to check expected input shapes and channel ordering.
- Search for tests that assert concrete shapes to update/add BVTS-aware tests.

---

If you'd like, I can now:
- (A) Expand this inventory automatically by searching for `torch.save`/`load` and `state_dict` and add results to this file, or
- (B) Start implementing the BVTS helper module `src/utils/field_conversion/bvts.py` and unit tests (non-breaking), or
- (C) Produce a small PR skeleton that introduces the helpers and adapters behind the config flag.

Tell me which next action you prefer and I'll proceed.

---

## Checkpoint & construction sites (search results)

I ran focused searches for checkpointing (torch.save/torch.load/state_dict/load_state_dict) and tensor construction (torch.stack/cat/tensor and np.save/np.load/np.stack/concatenate). Below are the concrete matches and short notes about why they matter for BVTS migration.

- `src/data/data_manager.py`
	- `torch.load(cache_path, weights_only=False)` and `torch.save(cache_data, cache_path)` are used to persist cached dataset artifacts.
	- Note: cached data may contain tensors written in the current layout (often time-first). When we change canonical layout, the cache read/write path must convert to/from BVTS or the cache format must be versioned.

- `scripts/generate_cache.py`
	- Loads checkpoints via `checkpoint = torch.load(checkpoint_path, map_location='cpu')` and then calls `model.load_state_dict(...)` with different possible keys (`model_state_dict`, `state_dict`, or the checkpoint itself).
	- Notes: generator script uses model checkpoints to synthesize data — ensure model input/output conversion is applied consistently when generating cache data.

- `src/training/tensor_trainer.py` and `src/training/hybrid/trainer.py`
	- Save and load training checkpoints: `torch.save(checkpoint, self.checkpoint_path)` and `torch.load(path, map_location=...)` / `self.model.load_state_dict(checkpoint["model_state_dict"])`.
	- Notes: trainer checkpoints include model weights (not input tensors), so they are not directly impacted by BVTS ordering, but any saved cached outputs or logged tensors in checkpoints would be.

- `src/data/dataset_utilities.py`
	- Uses `data = torch.load(file_path, map_location="cpu")` to read pre-saved tensors/files.
	- Notes: this is another cache-loading site that must convert loaded tensors to BVTS if the cache files are in old format.

- `src/evaluation/evaluator.py`
	- Uses `torch.load(checkpoint_path, map_location=self.device)` and `model.load_state_dict(checkpoint["model_state_dict"])`.
	- Also constructs `prediction_tensor = torch.cat(predictions, dim=0)  # [T, C, H, W]` and `gt_tensor = torch.cat(...)` when assembling rollouts.
	- Notes: evaluator currently assumes concatenated predictions are time-major with shape `[T, C, H, W]`. With BVTS we'll want concatenation to preserve time axis in the canonical position (time index 2), or use helpers to build BVTS-shaped tensors.

- `src/utils/field_conversion/batch.py`
	- Uses `torch.cat(tensors, dim=-3)` when concatenating along channel/vector dimension.
	- Notes: this uses negative indexing to find channel dim; negative indices will change meaning if we standardize dimension order — update to use explicit helper queries (e.g., get_vector_dim(tensor)).

- `src/data/tensor_dataset.py`
	- Multiple uses of `torch.cat(all_field_tensors, dim=1)  # [T, C_all, H, W]`, `torch.stack(tensors, dim=0)`, and earlier `permute` / `unsqueeze` to yield `[time, vector, x, y]`.
	- Notes: dataset construction currently creates time-first tensors and concatenates per-field along channel dim; the dataset cache writer likely persists that layout. We should add an explicit conversion to BVTS before caching.

- Other places to review (found via the earlier searches):
	- `src/evaluation/visualizations.py` (numpy axis reorderings)
	- `src/evaluation/metrics.py` (reshape/flatten behaviour assumes certain axis order)

Action items from these results:
- Add conversion/versioning when writing cache files in `src/data/data_manager.py` and `scripts/generate_cache.py` (write data in BVTS or write a format version + layout flag).
- Update any code that builds batches via `torch.cat`/`torch.stack` to use helpers or explicit axis constants (avoid negative indices like `-3`).
- Ensure evaluator and visualizations convert outputs to the expected plotting shapes (helpers should expose utilities to get time/vector/spatial dims by name).

If you want, I can now automatically update `docs/migration/inventory_bvts.md` with a small checklist for cache migration (add version field, convert on load, convert on save), or start implementing a small `bvts` helper module and example conversions in the dataset cache paths. Which would you prefer?
