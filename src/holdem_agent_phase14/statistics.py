"""Training-seed-level statistical estimates for fixed independent test-deck banks.

Do not falsely treat correlated deck blocks from different policies/runs as
independent training replicates.  The t interval quantifies between-seed
variation conditional on the specified test deck schedule.
"""
from __future__ import annotations

import math
from statistics import mean, stdev

# Student's t two-tailed 0.975 critical values for df 1..30.
_T975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)


def seed_level_summary(values: list[float] | tuple[float, ...]) -> dict:
    """Mean +/- approximate 95% t CI over *training seeds*, not deal groups.

    With one training seed the interval is unidentified and returned as null.
    Values are already in bb/100 units.
    """
    if not values or not all(isinstance(v, (float, int)) and not isinstance(v, bool)
                             and math.isfinite(v) for v in values):
        raise ValueError('need >=1 finite seed-level bb/100 estimates')
    n = len(values)
    mu = mean(values)
    if n == 1:
        return {'seeds': 1, 'mean_delta_bb100': round(mu, 4), 'se_bb100': None,
                'ci95_lower_bb100': None, 'ci95_upper_bb100': None,
                'positive_seed_fraction': round(float(values[0] > 0), 4)}
    se = stdev(values) / math.sqrt(n)
    # For n>31, use an asymptotic 1.96; slightly optimistic at moderate n.
    tcrit = _T975[n-2] if n <= 31 else 1.96
    return {'seeds': n, 'mean_delta_bb100': round(mu, 4), 'se_bb100': round(se, 4),
            'ci95_lower_bb100': round(mu-tcrit*se, 4),
            'ci95_upper_bb100': round(mu+tcrit*se, 4),
            'positive_seed_fraction': round(sum(v > 0 for v in values)/n, 4)}
