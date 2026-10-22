# Five-minute presentation outline (draft)

The timing guide totals 5:00. There is one idea per slide; use the figures in `results/figures/`.

| # | Time | Slide | Say (key points) | Show |
|---|---|---|---|---|
| 1 | 0:00–0:30 | **Problem** | BMS/HVAC analyses use logged CO₂ as a ventilation proxy. Logs have dropouts. How long a gap can we fill, and with which method? Retrospective, not real-time. | Title and research question |
| 2 | 0:30–1:15 | **Data and inputs/outputs** | Sensor 6012002000869, room G.38, 10-min CO₂ in ppm. Raw file: 29,341 rows → 13,800 genuine readings. Most rows are stale repeats, detected from the JSON measurement timestamp. Three of five sensors had no live data, and occupancy could not be linked to a room, so the scope was narrowed. Output: estimated ppm values at hidden times, plus error tables. | Extraction-count table; small pipeline diagram (raw CSV → extract → processed → experiment → results) |
| 3 | 1:15–2:00 | **Methods and design** | Forward-fill (causal baseline), linear, not-a-knot cubic spline. Chronological 70/30 split. Hide real readings for 10 min – 6 h, with up to 200 non-overlapping gaps each, the same gaps for every method, and an asserted leakage guard. Spline context tuned on the development period only, with every context scored on the same gaps. Note: D is a time window. "10 min" = one reading hidden, with known readings 20 min apart (30 min next to a native dropout). Long windows hide fewer readings than D/10 min (3 h → 16–17) because the sensor already misses about 1 reading in 10. | Gap-definition sketch |
| 4 | 2:00–2:40 | **Verification** | 98 pytest tests. Linear is exact on lines and the spline exact on cubics. Measured convergence orders are about 2 and above 3.5 (theory: 2 and 4). A poisoned hidden value never reaches a method. Metrics reproduce exactly on rerun. | `results/test-report.txt` summary; GitHub structure and pdoc HTML page (brief live demo) |
| 5 | 2:40–3:40 | **Results** | Linear is best at every duration: 7.8 → 12.7 ppm MAE from 10 min to 6 h. Forward-fill: 9.2 → 17.8. The spline is worst from 30 min upward: 47.8 ppm at 6 h, with up to 180 ppm overshoot. Paired day-block bootstrap on the same gaps: linear's advantage is statistically clear at every duration (all difference intervals exclude zero, also after Bonferroni), but it is practically small at 10 min (about 1 ppm) and large at 6 h (35 ppm vs spline). All methods take under 1 ms per gap, so cost does not decide; accuracy does. | `co2_error_vs_duration.png`; `co2_paired_differences.png` |
| 6 | 3:40–4:20 | **Why: interpretation** | The spline assumes smoothness. This CO₂ signal changes by about 9 ppm between consecutive readings, so the spline fits curvature to noise and amplifies it across the gap. Linear cannot leave the boundary range. Temperature (smoother) shows the spline much closer to linear: clearly worse only at short gaps, inconclusive at 3 h. That is consistent with the explanation, but not proof of it. | `co2_dev_context_sensitivity.png`; one example-gap plot if sharing is allowed (it plots raw readings, from `results/local/`) |
| 7 | 4:20–4:45 | **Limitations and lessons** | One sensor in one low-CO₂ room, one test month, random artificial gaps, unknown device specification. Improvements: PCHIP, smoothing, daily-profile models. [AUTHOR: your main lesson] | Bullet list |
| 8 | 4:45–5:00 | **AI use** | [AUTHOR: in your own words, what you used AI for, how you checked it, and one thing it got wrong.] | — |

**Demo checklist.**
- README sections.
- The `src/` layout.
- One docstring in `docs/api/gap_reconstruction/methods.html`.
- A `pytest -q` run (under 2 s).
