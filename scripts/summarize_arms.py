#!/usr/bin/env python3
"""
Build the rebuttal Table A (ablation) and Table C (latency) from arm output dirs.
Works on partial runs — uses whatever episodes have finished.

Usage: python scripts/summarize_arms.py /mnt/nas/ishneet/vls-rebuttal/code/outputs/rebuttal_*/
"""
import glob
import json
import math
import os
import re
import sys
from statistics import mean, median


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def audit_arm(arm_dir: str):
    eps = sorted(glob.glob(os.path.join(arm_dir, 'episode_*')))
    eps = [e for e in eps if os.path.isdir(e)]
    n = succ = prep_fail = 0
    ep_lens, chunk_guided, chunk_unguided, stage_qs, preps = [], [], [], [], []
    for ep in eps:
        files = os.listdir(ep)
        is_succ = any('_success' in f for f in files)
        is_fail = any('_fail' in f for f in files)
        if not (is_succ or is_fail):
            continue  # still running
        n += 1
        succ += int(is_succ)
        prep_fail += int(any(f == 'error.txt' for f in files) and not is_succ)
        tj = os.path.join(ep, 'timing.json')
        if os.path.exists(tj):
            try:
                t = json.load(open(tj))
                steps = t.get('steps', [])
                ep_lens.append(len(steps))
                if t.get('prep_seconds'):
                    preps.append(t['prep_seconds'])
                for s in steps:
                    if s.get('new_chunk'):
                        (chunk_guided if s.get('guidance') else chunk_unguided).append(s['select_action_s'])
                        if s.get('stage_update_s', 0) > 0.3:  # >0.3s = an actual VLM query
                            stage_qs.append(s['stage_update_s'])
            except Exception:
                pass
    return {
        'arm': os.path.basename(arm_dir.rstrip('/')), 'n': n, 'succ': succ,
        'prep_fail': prep_fail, 'ep_lens': ep_lens, 'chunk_guided': chunk_guided,
        'chunk_unguided': chunk_unguided, 'stage_qs': stage_qs, 'preps': preps,
    }


def fmt_s(vals):
    return f"{mean(vals):.2f}s (med {median(vals):.2f})" if vals else "—"


def main():
    dirs = []
    for pat in sys.argv[1:]:
        dirs += [d for d in glob.glob(os.path.join(pat, 'arm*')) if os.path.isdir(d)]
    if not dirs:
        print('usage: summarize_arms.py <OUTROOT ...>  (dirs containing arm*/episode_N)')
        return
    rows = [audit_arm(d) for d in sorted(dirs)]

    print('\n## Table A — Ablation (partial-safe)\n')
    print('| Arm | n done | Success % | 95% CI | Avg ep len | Prep failures |')
    print('|---|---|---|---|---|---|')
    for r in rows:
        if r['n'] == 0:
            print(f"| {r['arm']} | 0 | — | — | — | — |")
            continue
        p = 100 * r['succ'] / r['n']
        lo, hi = wilson_ci(r['succ'], r['n'])
        el = f"{mean(r['ep_lens']):.0f}" if r['ep_lens'] else '—'
        print(f"| {r['arm']} | {r['n']} | {p:.1f} | [{100*lo:.1f}, {100*hi:.1f}] | {el} | {r['prep_fail']} |")

    print('\n## Table C — Latency (from timing.json)\n')
    print('| Arm | Prep (1x/ep) | Chunk gen (guided) | Chunk gen (unguided) | Stage query | #queries/ep |')
    print('|---|---|---|---|---|---|')
    for r in rows:
        nq = f"{len(r['stage_qs'])/max(r['n'],1):.1f}" if r['n'] else '—'
        print(f"| {r['arm']} | {fmt_s(r['preps'])} | {fmt_s(r['chunk_guided'])} | "
              f"{fmt_s(r['chunk_unguided'])} | {fmt_s(r['stage_qs'])} | {nq} |")

    # Effective control frequency estimate from arm0 if present
    for r in rows:
        if 'arm0' in r['arm'] and r['chunk_guided']:
            chunk_s = mean(r['chunk_guided'])
            print(f"\nEffective chunk-level control: {chunk_s:.2f}s per 10-action chunk "
                  f"=> {10/chunk_s:.1f} actions/s equivalent (excl. env step + stage queries)")
            break


if __name__ == '__main__':
    main()
