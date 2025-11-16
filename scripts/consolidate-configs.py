#!/usr/bin/env python3
"""
Consolidate configs under conf/ into a single monolithic `conf/config_new.yaml`
and `conf/experiments_new.yaml`.

Usage:
  python scripts/consolidate-configs.py --path conf/ [--dry-run] [--out-dir conf/]

This tool is conservative: it will not overwrite existing files unless `--overwrite` is passed.
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
import yaml


def list_yaml_files(path):
    res = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(('.yml', '.yaml')):
                res.append(os.path.join(root, f))
    return sorted(res)


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as fh:
        try:
            return yaml.safe_load(fh)
        except Exception as e:
            return {'_parse_error': str(e)}


def set_nested(dct, keys, value):
    """Set a nested key path in dict dct given a list of keys."""
    cur = dct
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def consolidate(conf_path):
    all_files = list_yaml_files(conf_path)
    monolithic = {}
    experiments = {}

    for f in all_files:
        rel = os.path.relpath(f, conf_path)
        parts = Path(rel).parts
        data = load_yaml(f)
        if not data:
            continue

        # Skip config.yaml if it already is present - treat as base defaults
        if parts == ('config.yaml',):
            monolithic.update(data)
            continue

        # conf/data/<name>.yaml -> data.<name> = content
        if parts[0] == 'data' and len(parts) == 2:
            name = Path(parts[1]).stem
            set_nested(monolithic, ['data', name], data)
            continue

        # conf/model/<type>/<name>.yaml -> model.<type>.<name>
        if parts[0] == 'model' and len(parts) == 3:
            category = parts[1]
            name = Path(parts[2]).stem
            set_nested(monolithic, ['model', category, name], data)
            continue

        # conf/trainer/<name>.yaml -> trainer.<name>
        if parts[0] == 'trainer' and len(parts) == 2:
            name = Path(parts[1]).stem
            set_nested(monolithic, ['trainer', name], data)
            continue

        # conf/generation/default.yaml -> generation
        if parts[0] == 'generation' and len(parts) == 2:
            name = Path(parts[1]).stem
            set_nested(monolithic, ['generation', name], data)
            continue

        # conf/evaluation/default.yaml -> evaluation.default
        if parts[0] == 'evaluation' and len(parts) == 2:
            name = Path(parts[1]).stem
            set_nested(monolithic, ['evaluation', name], data)
            continue

        # conf/experiment/<name>.yaml -> experiments.<name> = minimal mapping
        if parts[0] == 'experiment' and len(parts) == 2:
            name = Path(parts[1]).stem
            # Keep only the sections that are relevant for mapping: run_params, model, generation_params, trainer_params, evaluation_params
            mapping = {}
            for candidate in ('run_params', 'model', 'generation_params', 'trainer_params', 'evaluation_params'):
                if candidate in data:
                    mapping[candidate] = data[candidate]
            # Also include a minimal root-level project_root if present
            if 'project_root' in data:
                mapping['project_root'] = data['project_root']
            experiments[name] = mapping
            continue

        # conf/hydra/default.yaml -> hydra
        if parts[0] == 'hydra' and len(parts) == 2:
            monolithic.setdefault('hydra', {})[Path(parts[1]).stem] = data
            continue

        # conf/logging/default.yaml -> logging
        if parts[0] == 'logging' and len(parts) == 2:
            monolithic.setdefault('logging', {})[Path(parts[1]).stem] = data
            continue

        # Fallback: put under 'other' with path mapping
        set_nested(monolithic, ['other', '/'.join(parts)], data)

    return monolithic, experiments


def write_yaml(obj, path):
    with open(path, 'w', encoding='utf-8') as fh:
        yaml.safe_dump(obj, fh, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='conf/', help='Path to config directory')
    parser.add_argument('--out-dir', default='conf/', help='Output directory for consolidated configs')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    monolithic, experiments = consolidate(args.path)

    out_conf = Path(args.out_dir) / 'config_new.yaml'
    out_exps = Path(args.out_dir) / 'experiments_new.yaml'

    print(f'Found {len(monolithic.keys())} top-level keys in monolithic result.')
    print(f'Found {len(experiments.keys())} experiments to consolidate.')

    if args.dry_run:
        print('Dry run mode: writing sample files to stdout instead of disk.')
        print('\n=== config_new.yaml ===')
        print(yaml.safe_dump(monolithic, sort_keys=False))
        print('\n=== experiments_new.yaml ===')
        print(yaml.safe_dump(experiments, sort_keys=False))
        return

    if out_conf.exists() and not args.overwrite:
        print(f'File {out_conf} already exists; pass --overwrite to replace.')
        sys.exit(1)
    if out_exps.exists() and not args.overwrite:
        print(f'File {out_exps} already exists; pass --overwrite to replace.')
        sys.exit(1)

    write_yaml(monolithic, out_conf)
    write_yaml(experiments, out_exps)
    print(f'Wrote {out_conf} and {out_exps}')


if __name__ == '__main__':
    main()
