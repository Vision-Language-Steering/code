#!/usr/bin/env python3
"""
Merge the killed fleet's complete task blocks (outputs/rebuttal_final) with the
patch runs (outputs/rebuttal_patch) into the final per-arm 10-task x 20-episode table.

Salvage map: the original run was killed mid-flight; only episodes belonging to
COMPLETE 20-episode task blocks are used from it (partial blocks are discarded and
fully rerun in the patch, same deterministic init states 0-19 per task).
"""
import glob
import json
import math
import os
import sys
from statistics import mean, median

FINAL = '/mnt/nas/ishneet/vls-rebuttal/code/outputs/rebuttal_final'
PATCH = '/mnt/nas/ishneet/vls-rebuttal/code/outputs/rebuttal_patch'

# arm -> (complete tasks in original run K, episodes to take = 20*K, patch target eps)
SALVAGE = {
    'arm0_full_vls':       (9, 20),
    'arm1_best_of_b':      (7, 60),
    'arm2_reward_only':    (9, 20),
    'arm3_generic_prompt': (7, 60),
    'arm4_const_lambda':   (9, 20),
    'arm5_det_stage':      (10, 0),   # fully completed in original run
    'arm6_kp_noise_2cm':   (8, 40),
    'arm7_kp_noise_5cm':   (8, 40),
}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, c - h), min(1.0, c + h))


def episode_outcome(ep_dir):
    files = os.listdir(ep_dir)
    if any('_success' in f for f in files):
        return True
    if any('_fail' in f for f in files):
        return False
    return None  # unfinished


def collect(run_dir, max_ep=None):
    """Return (n, succ, ep_lens, chunk_guided, chunk_unguided, stage_qs, preps)."""
    eps = sorted(glob.glob(os.path.join(run_dir, 'episode_*')),
                 key=lambda p: int(p.rsplit('_', 1)[1]))
    if max_ep is not None:
        eps = [e for e in eps if int(e.rsplit('_', 1)[1]) <= max_ep]
    n = succ = 0
    ep_lens, cg, cu, sq, pr = [], [], [], [], []
    for ep in eps:
        out = episode_outcome(ep)
        if out is None:
            continue
        n += 1
        succ += int(out)
        tj = os.path.join(ep, 'timing.json')
        if os.path.exists(tj):
            try:
                t = json.load(open(tj))
                steps = t.get('steps', [])
                ep_lens.append(len(steps))
                if t.get('prep_seconds'):
                    pr.append(t['prep_seconds'])
                for s in steps:
                    if s.get('new_chunk'):
                        (cg if s.get('guidance') else cu).append(s['select_action_s'])
                        if s.get('stage_update_s', 0) > 0.3:
                            sq.append(s['stage_update_s'])
            except Exception:
                pass
    return n, succ, ep_lens, cg, cu, sq, pr


def main():
    rows = []
    for arm, (k_tasks, patch_eps) in SALVAGE.items():
        n1, s1, el1, cg1, cu1, sq1, pr1 = collect(os.path.join(FINAL, arm), max_ep=20 * k_tasks)
        n2 = s2 = 0
        el2, cg2, cu2, sq2, pr2 = [], [], [], [], []
        if patch_eps:
            n2, s2, el2, cg2, cu2, sq2, pr2 = collect(os.path.join(PATCH, arm))
        n, s = n1 + n2, s1 + s2
        rows.append({
            'arm': arm, 'n': n, 's': s, 'complete': n >= 200,
            'ep_lens': el1 + el2, 'cg': cg1 + cg2, 'cu': cu1 + cu2,
            'sq': sq1 + sq2, 'pr': pr1 + pr2,
        })

    print('## FINAL Table A — merged (10 tasks x 20 eps per arm)\n')
    print('| Arm | n | Success % | 95% CI | Avg ep len | status |')
    print('|---|---|---|---|---|---|')
    for r in rows:
        if r['n'] == 0:
            continue
        p = 100 * r['s'] / r['n']
        lo, hi = wilson_ci(r['s'], r['n'])
        el = f"{mean(r['ep_lens']):.0f}" if r['ep_lens'] else '—'
        st = 'FINAL' if r['complete'] else f"partial ({r['n']}/200)"
        print(f"| {r['arm']} | {r['n']} | {p:.1f} | [{100*lo:.1f}, {100*hi:.1f}] | {el} | {st} |")

    print('\n## FINAL Table C — latency (merged)\n')
    print('| Arm | Prep 1x/ep | Chunk (guided) | Chunk (unguided) | Stage query | q/ep |')
    print('|---|---|---|---|---|---|')
    for r in rows:
        f = lambda v: f"{mean(v):.2f}s (med {median(v):.2f})" if v else '—'
        qep = f"{len(r['sq'])/max(r['n'],1):.1f}"
        print(f"| {r['arm']} | {f(r['pr'])} | {f(r['cg'])} | {f(r['cu'])} | {f(r['sq'])} | {qep} |")


if __name__ == '__main__':
    main()
