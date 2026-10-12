"""Obsidian-compatible Phase16 benchmark report from JSON-safe results."""
from __future__ import annotations

from holdem_ai.evaluation.action_diagnostics import ACTION_LABELS


def markdown_report(result: dict) -> str:
    cfg = result['config']
    summary = result['pool_summary']
    lines = [
        '# Phase 16 — HUNL200 Opponent Pool / Strategy Report',
        '',
        f"- Actor fingerprint: `{result['candidate_id']}`",
        f"- Mode: `{cfg['mode']}`; device: `{cfg['device']}`",
        f"- Deal seed range: `{cfg['seed']}` ... `{cfg['seed']+cfg['groups']-1}`",
        f"- Each opponent: $K={cfg['groups']}$ independent seed blocks, $4K={4*cfg['groups']}$ hands.",
        '',
        '## Opponent-pool performance', '',
        '| Opponent | bb/100 | CI 95% | Hands |',
        '|---|---:|---:|---:|',
    ]
    for row in result['opponents']:
        ci = f"[{row['ci95_lower_bb100']:+,.2f}, {row['ci95_upper_bb100']:+,.2f}]"
        lines.append(f"| {row['opponent']} | {row['mean_bb100']:+,.2f} | {ci} | {row['hands']} |")
    lines += [
        '',
        f"Equal-weight mean across these fixed opponents: **{summary['equal_weight_mean_bb100']:+,.2f} bb/100**.",
        f"Worst *observed* pool mean: **{summary['worst_observed_mean_bb100']:+,.2f} bb/100** "
        f"against **{summary['worst_opponent']}**.",
        '',
        '## Action diagnostics', '',
        'Choice frequency is conditional on the corresponding action being legal:',
        '',
        '$$\\widehat p(a\\mid a\\in\\mathcal A(I))=N_{\\mathrm{chosen}}(a)/N_{\\mathrm{legal}}(a).$$',
        '',
        'Policy entropy: $H(\\pi)=-\\sum_{a\\in\\mathcal A(I)}\\pi(a\\mid I)\\log\\pi(a\\mid I)$.',
        '',
    ]
    for row in result['opponents']:
        profile = row['candidate_action_profile']
        lines += [f"### vs {row['opponent']}", '',
                  f"- Actor decisions: **{profile['decisions']}**",
                  f"- Mean policy entropy: **{profile['mean_policy_entropy']}**",
                  f"- Mean normalized entropy: **{profile['mean_normalized_entropy']}**",
                  '', '| Action | Chosen | Legal opportunities | Chosen when legal |',
                  '|---|---:|---:|---:|']
        for a in ACTION_LABELS:
            freq = profile['conditional_action_rate'][a]
            rate = 'N/A' if freq is None else f'{100*freq:.2f}%'
            lines.append(f"| {a} | {profile['actions'][a]} | {profile['legal_opportunities'][a]} | {rate} |")
        for dimension in ('by_street', 'by_position', 'by_spr'):
            lines += ['', f'**{dimension}**', '',
                      '| Group | Decisions | Fold when legal | Check when legal | Call when legal | All-in when legal |',
                      '|---|---:|---:|---:|---:|---:|']
            for key, part in profile[dimension].items():
                rates = part['conditional_action_rate']
                fmt = lambda x: 'N/A' if x is None else f'{100*x:.1f}%'
                lines.append(f"| {key} | {part['decisions']} | {fmt(rates['fold'])} | "
                             f"{fmt(rates['check'])} | {fmt(rates['call'])} | {fmt(rates['all_in'])} |")
        lines.append('')
    lines += [
        '## Statistical protocol and limits', '',
        '$$X_k=\\frac14\\sum_{b=0}^1\\sum_{s=0}^1R_{k,b,s},\\qquad '
        '\\widehat J_{\\mathrm{bb/100}}=100\\cdot\\frac1K\\sum_k X_k.$$',
        '',
        '$$\\mathrm{CI}_{95\\%}\\approx 100\\left(\\overline X\\pm1.96\\frac{s_X}{\\sqrt K}\\right).$$',
        '',
        '- The unit for uncertainty is a whole deck-seed block, not an individual hand.',
        '- The pool mean and worst-opponent value are *descriptive*. They are not exploitability or guarantees.',
        '- Baseline opponents are simple heuristic policies, not poker professionals or GTO agents.',
        '- Different agents can induce different post-action paths even with the same deck seed.',
        '- This evaluation does not train or modify the PPO/Q-Boost model.',
        '- Avoid tuning repeatedly against a held-out test bank; reserve fresh seeds for final tests.',
        '',
    ]
    return '\n'.join(lines)
