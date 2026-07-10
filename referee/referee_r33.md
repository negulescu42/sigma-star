# Internal Referee — Round R33 (post Gibbs/Renyi/conformal upgrade)

Mean: **8.8250** (R32 8.7625, +0.0625) — new high.

| Axis | R32 | R33 |
|---|---|---|
| Novelty | 7.9 | 8.0 |
| Theory | 8.9 | 9.0 |
| Empirical | 9.1 | 9.2 |
| Calibration | 9.3 | 9.1 |
| Clarity | 8.7 | 8.6 |
| Significance | 8.8 | 9.0 |
| Reproducibility | 8.6 | 8.8 |
| Scholarship | 8.8 | 8.9 |

## World-class reviewer's criticisms, adjudicated
- (b) mean-vs-pointwise — SUBSTANTIVELY FIXED (Prop 7 conformal, coverage 0.947/0.943/0.953).
- (c) unnormalised-only — SUBSTANTIVELY FIXED (Theorem 2 Gibbs, bounds normalised far mass directly; 66k query-heads 0 viol). Most substantive addition.
- (e) invalid fixed point — CORRECTLY REPAIRED (order-reversing stated, bisection on monotone envelope).
- (d) attention transfer — MEANINGFULLY IMPROVED + correctly hedged (V_max obstruction removed by Thm 2; descriptive-certification language).
- (f) 'any monotone tail' — PROPERLY NARROWED (log-concave-in-log-distance, Prop 3).
- (a) 'elementary inequality' — NOT addressed; Note 12 itself concedes D_inf collapses to elementary max-weight bound. Caps Theory/Novelty.

## Calibration deduction (−0.2)
Referee caught: changelog claimed underflow 'accuracy at chance' corrected, but Related Work still said it verbatim. FIXED THIS ROUND — reworded to '1-NN rule (stable readout recovers 1-NN accuracy; naive linear-domain underflows to chance, an artifact we avoid)'. Rebuilt main v64 sha e6cff10c9e059806.

## Single highest-value remaining action (referee)
Confront the 'elementary' core directly: either (i) a genuinely non-trivial concentration bound beating worst-case-alignment under a mild regularity assumption on far-source configuration (not tight only in adversarial all-at-radius-d), or (ii) reposition the contribution honestly as a calibration/deployment framework on a simple well-instrumented inequality. More valuable than a 4th certificate variant.
