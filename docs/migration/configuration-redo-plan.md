# Configuration Rework Plan

This document outlines a detailed plan to redo configuration setup for this repository.

Goals
- Create a consistent, typed, validated configuration layout (Hydra-based) with well-defined overrides and defaults.
- Make configuration easy to discover, test, and use for developers (and reproducible for experiments).
- Provide migration tooling and documentation that minimizes disruption.

Contract
- Input: existing config files in `conf/` and any other YAML config files in repo; CLI usage that references configurations.
- Output: a canonical configuration layout under `conf/` (Hydra groups + typed configs), unit tests and CI checks, a migration script (where needed), and updated documentation.
- Error modes: missing required fields, wrong field types (validation failures), top-level incompatible keys, backward compatibility for simple transformations.

Edge Cases to handle
- Partial configs (users only override a small subset of keys).
- Fields that have changed types (int -> list, scalar -> nested dict).
- Deprecated keys that must map to new keys.
- Secrets and sensitive fields (keep them separate and out of config in CI).
- Complex overrides using Hydra `+` or `?` composition.

Acceptance Criteria
- All major components (`data`, `model`, `trainer`, `generation`, `evaluation`) have typed configs.
- Config loader validates all required keys and rejects malformed config YAML with meaningful errors.
- Tests cover config load + validation and a basic run path (e.g., load config -> instantiate key objects -> run a no-op method).
- Migration steps documented and a script provided to map old config instances to the new format where feasible.
- CI validates config schema on PRs.

Plan — high-level phases
1. Inventory & analysis (1-2 days)
   - Inspect `conf/` and any other YAML used as config (e.g., `doc` samples, `examples`, CLI arguments in the code base) and create a mapping of where each key is read.
   - Identify duplicates and multiple sources of truth.
   - Collect a list of config files and the Hydra groups (if already used).

2. Canonical structure design (1-2 days)
   - Decide on a standard top-level structure: e.g., `data`, `model`, `trainer`, `generation`, `evaluation`, `logging`, `experiment`.
   - Decide on a `default.yaml`, `schema/` and `typed_configs/` layout.
   - Draft YAML examples and typed dataclasses for each area.
   - Decide on the strategy for environment variables, sensitive secrets, and optional components.

3. Implement typed configs & validation (2-4 days)
   - Add a `dataclasses`/`pydantic` module to define typed configs.
   - Add Hydra config group files referencing the typed configs.
   - Update code to instantiate via these typed configs.
   - Add validators for critical keys and checks for ranges.

4. Migration & Compatibility (1-3 days)
   - Add a migration script that can convert existing configs to the new layout or validate compatibility.
   - Maintain backward compatibility by supporting `legacy` keys as mapping in the loader to the new keys with warnings.
   - Identify any behavior changes and document them.

5. Testing and CI (1-2 days)
   - Add unit tests to assert: loading typed configs correctly, default values set, overrides work, invalid configs raise clear errors.
   - Add end-to-end test(s) to run a minimal experiment from sample config.
   - Add CI checks that run the config lint and tests on PRs.

6. Documentation & migration instructions (0.5-1 day)
   - Update `docs/` with examples and migration guidance.
   - Add a `conf/README.md` describing the new structure and usage with Hydra and CLI examples.

7. Rollout & Cleanup (1-2 days)
   - Roll out changes in smaller steps where feasible and communicate change to contributors.
   - Remove old configs after a grace period and after migration tools are used.

Implementation details
- Suggested canonical layout (in `conf/`):
   - `conf/defaults.yaml` - references the Hydra groups and default picks.
   - `conf/data/` - data group configs (e.g., `advection_128.yaml`, `burgers_128.yaml`).
   - `conf/model/` - model groups.
   - `conf/trainer/` - training strategies and hyperparameters.
   - `conf/generation/` - generation options (if used separately).
   - `conf/evaluation/` - evaluation config for metrics.
   - `conf/logging/` - logging config and level adjustments.
   - `conf/schema/` - typed config dataclasses or pydantic models (Python files under `src/config/`).
- Typed config approach: Hydra strongly typed configs using `@dataclass` (OmegaConf) or Pydantic validated models. Both are compatible; prefer dataclasses since Hydra common patterns use `dataclasses` and OmegaConf.
- Validation: Add a `validate_config(config)` utility that runs cross-field validations (e.g., `if model.type==X then require `num_channels` present`). Also add strict schema checks using `omegaconf` or `pydantic`.
- Migration: keep a script at `scripts/migrate-config.py` that reads old YAML config(s), maps keys, and outputs new YAML with a `--dry-run` mode.
- Secrets: instruct to keep secrets out of the repo and use environment variables or a secrets manager (Hydra supports `env` interpolation). Add a `conf/secrets.yaml` template with placeholders that are not committed.

Testing & CI details
- Unit tests to add:
  - `test_config_load.py` – load default config, check typed dataclass, and verify defaults.
  - `test_config_overrides.py` – override a nested key via Hydra and verify the value.
  - `test_config_validation.py` – invalid configs fail with correct messages.
  - `test_migration_script.py` – assert migration script maps old -> new.
- Add a `flake`/`linters` note for config code.
- Add a `config-linting` target in CI to run `pytest tests/test_config_*.py` or additional checks.

Developer Steps — Example Flow
1. Run inventory script to print a list of YAMLs referencing each group.
2. Create `src/config/typed.py` dataclass definitions.
3. Implement migrations (if mapping needed).
4. Add unit tests and commit changes.
5. Update CI YAML to include the config-lint test.

Rollout strategy
- Create a `config-redesign/` feature branch.
- Introduce typed configs in a first PR; update the code to use them with a fallback that uses raw OmegaConf dicts.
- Introduce migration mappings and run tests in a second PR.
- Finally, fully switch over and remove legacy config support.

Migration Checklist (practical):
- [ ] Run `scripts/inventory-configs.py` and collect all YAMLs.
- [ ] For each config key in old configs, decide mapping.
- [ ] Implement mapping in `scripts/migrate-config.py` with best-effort automated conversion.
- [ ] Update tests to assert transformed config matches the new schema.
- [ ] Announce the changes and tag important users before full removal.

Examples & CLI
- Loading config (Hydra):
  - `python run.py experiment=advection_hybrid` -> `conf/defaults.yaml` picks `experiment: advection_hybrid`.
  - `python run.py +trainer.lr=1e-4 data.batch_size=16` for override.
- Ensure `run.py` uses the same typed config entry point to avoid manual parsing.

Follow-ups and Extras
- Consider adding a `config show` command which loads the final composed config to inspect values (for debugging/QA). Hydra has `hydra.utils.instantiate` and `hydra` CLI subcommands for this.
- Optionally add a `config-lint` tool to detect unused/deprecated keys.
- Create a deprecation policy: `deprecated_keys: {old: new, ...}` and warn in logs.

Estimated time
- Overall: ~1-2 weeks (split into multiple PRs). The time varies based on how many configs and how intertwined code is with existing config layout.

Notes
- If the repository already has Hydra integrated, we should keep it and improve usage. The goal is to unify usage and ensure typed/validated configs.
- If you prefer Pydantic over dataclasses, it can also be used, but please ensure consistent representation.

That's the plan. If you'd like, I can begin by performing the inventory step (creating a script and listing all config files and their YAML keys) and generate the first PR with typed dataclasses; tell me if that's the next immediate action you want.