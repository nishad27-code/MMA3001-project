"""Reconstruction of missing indoor CO2 readings from a building sensor log.

MMA3001 Numerical Methods and Machine Learning project.

The package is organised as a small pipeline:

- `gap_reconstruction.loader` streams the raw Monash Smart Infrastructure
  environmental CSV and extracts genuine (non-stale) readings for one sensor.
- `gap_reconstruction.extract` is the command-line step that writes the
  processed CO2 time series to ``data/processed/``.
- `gap_reconstruction.methods` implements the three reconstruction methods
  (forward-fill baseline, linear interpolation, cubic-spline interpolation).
- `gap_reconstruction.evaluation` hides contiguous blocks of real readings,
  reconstructs them and scores the reconstructions.
- `gap_reconstruction.experiment` is the command-line step that runs the
  experiments and writes tables and figures to ``results/``.

All reconstruction here is *retrospective*: the sensor log is complete up to
some later time and we fill holes in the past. Linear and spline
interpolation use readings recorded *after* a gap, so they are not real-time
predictors. Forward-fill uses only earlier readings.
"""

__version__ = "0.1.0"
