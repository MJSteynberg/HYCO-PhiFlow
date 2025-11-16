#!/usr/bin/env python3
"""
Inventory config files under conf/ and print top-level keys and nested structures.

Usage:
    python scripts/inventory-configs.py [--path conf/]

This helps consolidate multiple YAML config files by showing what's defined across them.
"""
import argparse
import os
import sys
from collections import defaultdict

try:
    import yaml
except Exception:
    print("PyYAML is required to run this script. Please install with `pip install pyyaml`.")
    sys.exit(1)


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


def summarize(data):
    if data is None:
        return []
    if isinstance(data, dict):
        keys = list(data.keys())
        # For nested dicts, show two-level keys
        nested = {}
        for k, v in data.items():
            if isinstance(v, dict):
                nested[k] = list(v.keys())
        return keys, nested
    return [type(data).__name__], {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='conf/', help='Path to config directory')
    parser.add_argument('--format', choices=['text', 'json'], default='text')
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f'Path not found: {args.path}')
        sys.exit(2)

    files = list_yaml_files(args.path)
    if not files:
        print(f'No YAML files found under {args.path}')
        sys.exit(0)

    summary = defaultdict(dict)
    for f in files:
        data = load_yaml(f)
        keys, nested = summarize(data)
        summary[f]['top_keys'] = keys
        summary[f]['nested'] = nested

    # Print results
    if args.format == 'text':
        print('\nConfiguration inventory:')
        for f, v in summary.items():
            print(f'\n- {f}')
            print('  top-level keys:', ', '.join(v['top_keys']))
            if v['nested']:
                print('  nested keys:')
                for k, n in v['nested'].items():
                    print(f'    - {k}: {n}')
    else:
        import json

        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
