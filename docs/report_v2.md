# Reconstructing Missing Indoor CO₂ Readings from a Building Sensor Log

[AUTHOR: name, student ID, date]

## 1. Engineering problem

**Context.** Commercial buildings monitor indoor CO₂ because it is a practical proxy for how well a space is ventilated relative to the number of people in it. Building management (BMS) and HVAC analyses, such as ventilation audits, demand-controlled-ventilation tuning and comfort and energy reports, work on logged time series. Wireless sensor logs contain dropouts. Analyses then either discard incomplete periods or fill them, and filling them badly can bias averages, peak estimates and threshold-exceedance counts.

**Problem addressed.** Given a log of genuine CO₂ readings with a missing block, estimate the missing readings and quantify how the error of simple reconstruction methods grows with the length of the gap.

**Research question.** *How does the accuracy of reconstructing missing indoor CO₂ measurements degrade as gap duration increases, and how do linear interpolation and cubic-spline interpolation compare with a simple forward-fill baseline?* [AUTHOR: update the research question to include PCHIP]

**Intended use.** Retrospective gap filling of archived logs before analysis, with an evidence-based limit on how long a gap can be filled before the result should be flagged or discarded. Linear and spline interpolation use the reading *after* a gap, so they are not real-time methods. Only forward-fill could run in real time.

[AUTHOR: state that this follows Dataset 2's listed direction 'reconstructing missing sensor measurements']

[AUTHOR: one sentence on limits of scope: one sensor, retrospective use only]

## 2. Inputs, outputs and data

### 2.1 Inputs

| Input | Format | Units / domain | Handling of invalid input |
|---|---|---|---|
| Raw environmental CSV (`5EnvSensor_MayToDec2024_180kRows.csv`, 353 MB) | CSV file. The `jsondata` column holds each row's readings as JSON text. | Readings with their measurement time (Unix seconds, UTC) and unit | Missing file or missing column: the run stops with an error |
| Sensor ID | Text, exact match | `6012002000869` | No matching rows: the run stops with a message |
| CO₂ value | Number inside the JSON | ppm; accepted range 250–5000 ppm | Rejected and counted, with the reason (for example bad JSON, wrong unit, missing or non-numeric value, out of range) |

Each reading's own timestamp (the JSON `time`) is used. The `createdate` column is not used. It is the time the server stored the row. It mixes two UTC offsets because of daylight saving. It also keeps advancing while the sensor is silent.

Unsupported input: the interpolation methods do not extrapolate. A query time outside the known readings is refused with an error (`ValueError`).

### 2.2 Outputs

| Output | Meaning and intended use |
|---|---|
| `data/processed/co2_6012002000869.csv` (`epoch_s, time_utc, co2_ppm`) | The cleaned series of genuine readings. It is the input to all experiments and stays on the local machine. |
| `results/co2_test_metrics.csv` | On the held-out test set, for each gap duration and method: number of gaps, error measures, share of gaps won and runtime. |
| `results/co2_dev_context_sensitivity.csv` | Results of hyperparameter tuning on the validation set (spline only). |
| `results/*_run_info.json`, `*_profile.txt`, `figures/*.png` | Run settings (periods, seed, versions), a profile of where time is spent, and the figures. |
| Function-level inputs/outputs | Each method returns one estimate per query time, in the same units as the input. See the HTML API docs (`docs/api/`). |

### 2.3 Data audit findings

The environmental file has 179,447 rows from 5 sensors. Only two sensors contain live data. The other three (…227, …241 and …777) repeat one identical reading in every row. [AUTHOR: why not the other live sensor?] This project uses sensor `6012002000869`:

- Source rows: 29,341.
- Genuine CO₂ readings: **13,800**. The other rows are stale copies of an earlier reading.
- Coverage: 90.8 % of the readings expected every 10 minutes.
- Native gaps: the sensor itself misses about 1 reading in 10. There are 1,390 such gaps.

[AUTHOR: one sentence: the occupancy data were not used, and why]

## 3. Computational approach

### 3.1 Methods

Each method estimates the ground truth (the hidden readings) from the known readings around the gap. Interpolation has no training phase. Like KNN, each estimate uses only the nearest known readings. The known readings used on each side are called the **context (a hyperparameter)**. For example, a context of 2 means two known readings before the gap and two after.

- **Forward-fill (baseline model).** Repeats the last reading before the gap. It uses only earlier readings, so it could run in real time. Its error grows as the true signal drifts away from that last value.
- **Linear interpolation.** Draws a straight line between the two readings either side of the gap (`numpy.interp`). The line always stays between those two values. So it can never overshoot, which means going above or below both nearby readings.
- **Cubic spline.** A spline is a smooth curve made of short cubic pieces joined end to end. It passes through every known reading (`scipy.interpolate.CubicSpline`, with "not-a-knot" ends). On smooth signals it becomes much more accurate than a straight line as readings get closer. It has no limit on its shape, so it can overshoot. Times are shifted to start at zero before fitting. This keeps the numbers well scaled, because raw Unix times are about 1.7 × 10⁹ s.
- **PCHIP.** [AUTHOR: write 2-3 sentences on PCHIP: what it is and why it was added]

**Why these methods.** They are the standard first tools for filling missing values in a time series. They are easy to interpret and cost almost nothing to run. The comparison tests one question: does the spline's extra accuracy on smooth curves carry over to a real, noisy sensor signal? [AUTHOR: one clause on why PCHIP was added]

**Assumptions.**
1. Each reading's own timestamp is correct.
2. The readings either side of a gap are genuine.
3. Readings are close enough together that the signal between them is fairly smooth. The spline relies on this the most.
4. Gaps placed at random behave like real dropouts.

### 3.2 Experimental design

- **Chronological split.** The first 70 % of readings form the validation set: 9,660 readings, 29 Feb – 13 May 2024. The last 30 % form the held-out test set: 4,140 readings, 14 May – 13 Jun 2024. No reading is in both. Gaps and their context never cross the boundary.
- **Gap duration.** A gap's duration D is a time window that starts at a genuine reading. D is 10, 30, 60, 180 or 360 min. Every reading in that window is hidden. For example, a 10-minute gap always hides exactly one reading. Its known neighbours were usually 20 min apart (173 of 200 gaps in the held-out test set). In the other 27, a native gap sat next to it, so they were 30 min apart.
- **Placement.** Up to 200 gaps per duration are placed at random. A fixed random seed makes the run repeat exactly. Gaps never overlap. Long gaps did not always reach 200: the held-out test set has 187 three-hour and 92 six-hour gaps.
- **Fairness and no data leakage.** All methods are scored on the same gaps, with the same context. The ground truth is removed before a method sees the data. A check in the code (`reconstruct_gap`) stops the run if any ground-truth value ever reaches a method. This is tested in `tests/test_evaluation.py`.
- **Hyperparameter tuning on the validation set.** The only hyperparameter is the spline's context: 2, 4, 8, 16 or 32 readings each side. It was chosen on the validation set only, by the lowest average spline MAE. PCHIP needs no hyperparameter tuning. Its estimate depends only on the two readings each side, so contexts 2 and 8 give identical predictions (this is tested).
- **Metrics.**
  - MAE (mean absolute error): the average size of the error, in ppm, over all ground-truth readings.
  - RMSE (root-mean-square error): like MAE, but large errors count more.
  - Each comes with a 95 % interval, found by resampling whole gaps 1,000 times (a bootstrap).
  - Maximum overshoot: how far an estimate went beyond the range of the context readings.
  - % of gaps won: the share of gaps in which one method had a lower error than another.

### 3.3 Implementation

The code is a Python package, `src/gap_reconstruction/`. Its modules are `loader`, `extract`, `methods`, `evaluation`, `experiment` and `benchmark`. The repository is on GitHub: https://github.com/nishad27-code/MMA3001-project. Supporting tooling:
- Docstrings in NumPy style, with HTML docs made by pdoc.
- A test suite of 113 tests (pytest).
- Pinned versions: numpy 1.26.4, scipy 1.13.1, matplotlib 3.9.2.

The 353 MB raw file is read once, in about 5 s. Rows from other sensors are skipped before any JSON is decoded. All experiments then use the 0.5 MB cleaned file. A full experiment run takes about 40 s on a laptop.

## 4. Verification

- **Exact answers.** Linear interpolation reproduces a straight line to within 1e-9. The spline reproduces a cubic polynomial to within 1e-8, even with realistic Unix times.
- **Convergence order.** The order says how fast the error shrinks as readings get closer together. The measured order is 2 ± 0.2 for linear interpolation. For the spline it is above 3.5 (theory: 4).
- **No data leakage.** A ground-truth value deliberately set to 10⁶ never reaches a method. The leakage check also stops a deliberately broken gap.
- **PCHIP stays within the boundary readings.** On 300 noisy synthetic gaps, PCHIP never went beyond the two readings either side. The spline did overshoot on the same data.
- **Test suite.** All 113 tests pass (pytest, latest run). [PENDING: regenerate `results/test-report.txt`, which still shows an older count]

## 5. Validation and results

**What was validated.** How well each method recovers genuine CO₂ readings that were hidden on purpose. The true values are known. So this is a check against ground truth, on the held-out test set. That set was not used for any choice.

**Why it is appropriate.** It tests exactly the intended use: filling holes in a real log. It uses this sensor's real noise and behaviour. No ground-truth value is ever used as input.

**Did validation influence model selection?** [AUTHOR: yes, PCHIP was added after test results showed spline overshoot; it has no tunable settings, and the final choice (linear) did not change]

**Hyperparameter tuning on the validation set.** Context 2 had the lowest average spline MAE, so it was selected. It is the smallest context possible. With 4 points, the spline is a single cubic through them. More context made the spline worse (21.8 → 28.9 ppm).

![Validation-set context sensitivity](../results/figures/co2_dev_context_sensitivity.png)
*Figure 1. Spline MAE for each context size, validation set only. Every context uses the same 200 gaps per duration. The dashed line marks the chosen context.*

**Table 1. CO₂ results on the held-out test set** (`results/co2_test_metrics.csv`; spline context 2). MAE is in ppm. The 95 % bootstrap interval over gaps is in brackets. "Boundary interval" is the average time between the known readings either side of a gap. The baseline model, linear and PCHIP never overshot (0.0 ppm at every duration).

| Gap D | Gaps | Boundary interval (min) | Baseline (forward-fill) MAE | Linear MAE | Spline MAE | PCHIP MAE | Max overshoot, spline (ppm) |
|---|---|---|---|---|---|---|---|
| 10 min | 200 | 21.4 | 9.21 [8.31, 10.17] | **7.82** [7.08, 8.62] | 8.97 [8.12, 9.88] | 8.08 [7.32, 8.93] | 6.5 |
| 30 min | 200 | 41.3 | 9.78 [9.05, 10.53] | **7.98** [7.46, 8.52] | 10.83 [10.05, 11.64] | 8.38 [7.82, 8.96] | 17.1 |
| 1 h | 200 | 71.1 | 10.95 [10.12, 11.78] | **8.23** [7.73, 8.72] | 13.01 [12.09, 13.99] | 8.74 [8.19, 9.27] | 27.1 |
| 3 h | 187 | 191.8 | 15.05 [13.43, 16.80] | **10.15** [9.51, 10.84] | 26.13 [23.92, 28.36] | 11.11 [10.35, 11.94] | 81.9 |
| 6 h | 92 | 370.9 | 17.78 [15.07, 20.58] | **12.71** [11.20, 14.49] | 47.76 [42.28, 54.12] | 13.34 [11.61, 15.27] | 179.9 |

[AUTHOR: one sentence: % of gaps where linear beat each other method, from frac_gaps_* columns]

![CO₂ error vs gap duration](../results/figures/co2_error_vs_duration.png)
*Figure 2. MAE and RMSE on the held-out test set against gap duration (log scale), with 95 % bootstrap intervals.*

**Signal context (held-out test set).** CO₂ averaged 470 ppm (SD 39 ppm). Most readings (5th–95th percentile) were between 428 and 555 ppm, and the maximum was 634 ppm. Consecutive readings, 10 minutes apart, differed by 9.0 ppm on average. Readings are reported to 1 ppm. This sets a floor on the error. The 10-minute linear MAE (7.8 ppm) is already below that typical change. So a single missing reading cannot be estimated much better from its neighbours.

[AUTHOR: acceptance criterion: MAE below the typical reading-to-reading change (9.0 ppm, from co2_run_info.json); state which methods/durations pass]

**Unsupported inputs and failure modes.** The methods refuse to extrapolate beyond the known readings. Gaps longer than 6 h were not tested. The spline can overshoot badly, by up to 179.9 ppm on 6-hour gaps.

**What the validation cannot establish.**
- Performance in rooms with high or fast-changing CO₂. This room never went above 634 ppm.
- Performance in other seasons, sensors or buildings.
- Performance on real dropouts, if they do not happen at random times. For example, outages caused by power or network failures could behave differently.
- Whether the measurement timestamps themselves are correct.

## 6. Comparison and performance

**Accuracy.** Linear interpolation is the most accurate method at every duration tested. The spline is worse than the baseline model (forward-fill) from 30 minutes upward.

**Trade-offs.**

| Method | MAE at 1 h (ppm) | MAE at 6 h (ppm) | × change, 10 min → 6 h | Max overshoot (ppm) | Runtime ratio vs linear | Real-time capable? | Complexity |
|---|---|---|---|---|---|---|---|
| Baseline (forward-fill) | 10.95 | 17.78 | ×1.9 | 0.0 | [PENDING: benchmark rerun] | [AUTHOR] | [AUTHOR] |
| Linear | 8.23 | 12.71 | ×1.6 | 0.0 | 1 | [AUTHOR] | [AUTHOR] |
| Cubic spline | 13.01 | 47.76 | ×5.3 | 179.9 | [PENDING: benchmark rerun] | [AUTHOR] | [AUTHOR] |
| PCHIP | 8.74 | 13.34 | ×1.7 | 0.0 | [PENDING: benchmark rerun] | [AUTHOR] | [AUTHOR] |

**Runtime.** [PENDING: benchmark rerun]

**Profiling and FLOP estimate.**
- **Profile** (`results/co2_profile.txt`). For the spline, most of the time goes into setting up the `CubicSpline` object and checking its inputs. Very little goes into arithmetic. Our own input checks are the next biggest cost for all methods.
- **Arithmetic per gap.** With context 2 there are 4 known points and at most 36 ground-truth readings to estimate.
  - Linear: about 5 floating-point operations (FLOPs) per estimated reading, so under 200 per gap.
  - Spline: about 40 FLOPs to set up the curve (one small system of equations), plus about 8 per estimated reading. That is under 400 per gap.
- **Implication.** At about 500 µs per gap, the spline does under 1 million FLOPs per second. A laptop core can do about 10¹⁰ per second. So the time goes on Python and library overhead, not on arithmetic. [PENDING: benchmark rerun – update the µs figure]
- [AUTHOR: one sentence linking FLOP estimate to the Roofline model (Week 1)]
- [AUTHOR: memory use: negligible, 4 known points per gap]
- **Engineering implication.** Filling all 1,390 native gaps in the log takes under a second with any method. So computing cost does not separate the methods. The choice should rest on accuracy and robustness. [PENDING: benchmark rerun – confirm the "under a second" estimate]

**Final design decision.** To fill gaps in this sensor's CO₂ log after the fact, use **linear interpolation**. Mark the filled values as reconstructed. Leave gaps longer than an agreed limit unfilled, for example 6 h, the longest tested. Do not use an unconstrained cubic spline on noisy 10-minute CO₂ data.

## 7. Limitations and lessons learned

**Why the spline fails here.** The spline's O(h⁴) advantage assumes the signal is smooth at the sampling scale. At 10-minute sampling, this CO₂ signal varies by about 9 ppm between consecutive readings, either measurement noise or genuinely fast fluctuation, while longer-term trends are small. In ML terms, the spline is overfitting the noise. It treats the noisy boundary readings as if they defined a true slope and curvature. It continues that curvature across the gap, so overshoot rises from 6.5 ppm at 10 min to 180 ppm at 6 h. Linear interpolation cannot leave the range of its two boundary values, which makes it robust. [AUTHOR: one sentence on temperature: smoother signal, PCHIP slightly better at 3-6 h, consistent with explanation]

**Why long gaps are harder.** All methods only see the edges of a gap. Real changes inside it, such as occupancy events, ventilation cycles or day–night patterns, are invisible to them. The boundary readings are also single noisy samples, so any trend inferred from them is uncertain. As the gap gets longer, the true signal has more time to depart from what the boundaries suggest.

**Data-quality lesson.** Most rows in the raw file were stale copies, and three sensors had no live data at all. Without stale-reading detection, a naive analysis would have treated constant repeated values as real measurements and judged reconstruction on fake data.

**Limitations.**
- One sensor in one low-occupancy room (labelled Compactus_upper).
- 3.5 months of data, with a one-month held-out test set in late autumn 2024.
- Artificial gaps placed at random.
- The device accuracy specification is unknown.
- The room location comes from a workbook that was not independently verified.

**Possible improvements (not implemented).**
- A smoothing spline or local regression to suppress noise before interpolating.
- A daily-profile or seasonal component for long gaps.
- Multivariate reconstruction using humidity or temperature.
- Validating on further sensors if the teaching team can supply repaired data.

**[AUTHOR] Personal lessons learned.** Add your own reflections here: what you learned and what surprised you. These must be your own.

## 8. AI-use reflection

[AUTHOR: ~350-word AI reflection covering the brief's 8 points: tools, uses, contribution level, why, how checked, decisions kept, errors, effect on learning]

## References

- Monash Smart Infrastructure, *MMA3001 Dataset – MSI Occupancy and Environmental* (provided by the MMA3001 teaching team, 2026).
- Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods* 17, 261–272.
- Harris, C. R. et al. (2020). Array programming with NumPy. *Nature* 585, 357–362.
- de Boor, C. (2001). *A Practical Guide to Splines* (rev. ed.). Springer. (Not-a-knot spline conditions and error bounds.)
- Burden, R. L. & Faires, J. D. *Numerical Analysis*. Cengage. (Interpolation error O(h²) and O(h⁴).)
- Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- [AUTHOR: add Fritsch, F. N. & Carlson, R. E. (1980), Monotone piecewise cubic interpolation, SIAM J. Numer. Anal. 17(2), 238–246, if PCHIP is cited; check this reference first]
- [AUTHOR: add a source for typical NDIR CO₂ sensor accuracy, and ventilation guidance such as ASHRAE 62.1 or the NCC/AS 1668.2, before relying on any sensor-accuracy comparison in Section 5.]
