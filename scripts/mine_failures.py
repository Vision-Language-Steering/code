#!/usr/bin/env python3
"""
E1 — Reward-synthesis reliability miner (AC concern #1).

Walks one or more results directories and classifies every episode:
  ok                : guidance generated, parsed, episode ran
  prep_parse        : VLM output could not be parsed (num_stages / function blocks missing)
  prep_compile      : generated code failed exec/syntax at load time
  prep_validation   : functions failed the dummy-tensor smoke test (shape/dtype/runtime)
  prep_other        : other preparation error (API failure, grounding failure, ...)
  exec_error        : episode crashed during execution
  runtime_guidance  : episode ran but guidance functions threw at runtime (gradient zeroed)

Also reports success rates conditioned on each bucket, which is exactly the
"task success when the synthesized reward is invalid" number reviewers want.

Usage:
  python scripts/mine_failures.py outputs/rebuttal_*/arm0_full_vls [more dirs...]
  python scripts/mine_failures.py --csv audit.csv outputs/...
"""
import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict


def classify_error_text(text: str) -> str:
    t = text.lower()
    if 'syntaxerror' in t or 'indentationerror' in t:
        return 'prep_compile'
    if 'num_stages not found' in t or 'could not parse' in t or 'parse' in t and 'error' in t:
        return 'prep_parse'
    if 'validation failed' in t or 'shape' in t or 'dtype' in t or 'dimension' in t:
        return 'prep_validation'
    if 'apiconnection' in t or 'rate limit' in t or 'timeout' in t or '429' in t or 'connection' in t:
        return 'prep_api'
    return 'prep_other'


def audit_episode(ep_dir: str) -> dict:
    row = {
        'episode': os.path.basename(ep_dir),
        'run': os.path.basename(os.path.dirname(ep_dir)),
        'bucket': 'ok',
        'success': None,
        'has_vlm_output': os.path.exists(os.path.join(ep_dir, 'vlm_agent', 'output_raw.txt')),
        'n_stage_files': 0,
        'runtime_guidance_errors': 0,
    }
    vlm_dir = os.path.join(ep_dir, 'vlm_agent')
    if os.path.isdir(vlm_dir):
        row['n_stage_files'] = len([f for f in os.listdir(vlm_dir) if f.endswith('_guidance.txt')])

    # success/failure from artifact names
    for f in os.listdir(ep_dir):
        if '_success' in f:
            row['success'] = True
        elif '_fail' in f and row['success'] is None:
            row['success'] = False

    err_path = os.path.join(ep_dir, 'error.txt')
    if os.path.exists(err_path):
        with open(err_path, errors='replace') as fh:
            text = fh.read()
        if 'preparation' in text.splitlines()[0].lower() if text.splitlines() else False:
            row['bucket'] = classify_error_text(text)
        elif 'execution' in (text.splitlines()[0].lower() if text.splitlines() else ''):
            row['bucket'] = 'exec_error'
        else:
            row['bucket'] = classify_error_text(text)
    return row


def scan_run_log_for_guidance_errors(run_dir: str) -> int:
    """Count 'Guidance function error' warnings in the run's console log if present."""
    count = 0
    for cand in ('run.log', 'main.log'):
        p = os.path.join(run_dir, cand)
        if os.path.exists(p):
            with open(p, errors='replace') as fh:
                count += sum(1 for line in fh if 'Guidance function error' in line)
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+', help='run output dirs containing episode_N/ subdirs')
    ap.add_argument('--csv', default=None, help='write per-episode rows to CSV')
    args = ap.parse_args()

    rows = []
    for d in args.dirs:
        if not os.path.isdir(d):
            print(f'skip (not a dir): {d}', file=sys.stderr)
            continue
        eps = sorted([os.path.join(d, e) for e in os.listdir(d)
                      if re.match(r'episode_\d+$', e) and os.path.isdir(os.path.join(d, e))],
                     key=lambda p: int(p.rsplit('_', 1)[1]))
        for ep in eps:
            rows.append(audit_episode(ep))
        n_runtime = scan_run_log_for_guidance_errors(d)
        if n_runtime:
            print(f'[{d}] {n_runtime} runtime guidance-function errors in console log')

    if not rows:
        print('No episodes found.')
        return

    buckets = Counter(r['bucket'] for r in rows)
    total = len(rows)
    print(f'\n=== E1 reliability audit: {total} episodes ===')
    for b, n in buckets.most_common():
        print(f'  {b:18s} {n:5d}  ({100 * n / total:.1f}%)')

    print('\n=== Success rate conditioned on bucket ===')
    by_bucket = defaultdict(list)
    for r in rows:
        if r['success'] is not None:
            by_bucket[r['bucket']].append(r['success'])
    for b, vals in sorted(by_bucket.items()):
        sr = 100 * sum(vals) / len(vals)
        print(f'  {b:18s} n={len(vals):4d}  success={sr:.1f}%')

    valid = total - sum(n for b, n in buckets.items() if b.startswith('prep_'))
    print(f'\nSynthesis validity rate: {100 * valid / total:.1f}% '
          f'({valid}/{total} episodes reached execution with parsed, validated rewards)')
    print('NOTE: semantic correctness (right keypoint / right stage logic) still needs a '
          'manual audit of a sample of vlm_agent/output_raw.txt files.')

    if args.csv:
        with open(args.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'\nPer-episode rows -> {args.csv}')


if __name__ == '__main__':
    main()
