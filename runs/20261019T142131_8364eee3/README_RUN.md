# CIM26 execution 20261019T142131_8364eee3

Engineering diagnostic, not a scientific article or a universal model ranking.
Open the directory in Artigo > Histórico, or import its exported ZIP. No recalculation occurs.
To reproduce in the supplied application: `python -m cim26 run --config runs/20261019T142131_8364eee3/config.json`.
This creates a NEW directory. It reads the snapshot CSV files in this run's inputs directory.
All four models use the same eligible timestamps. `common_valid` also requires convergence and
deterministic output across all rounds; metrics use that fixed intersection.
Regenerate figures: `python -m cim26 figures runs/20261019T142131_8364eee3` (writes only figures).
Recompute metrics from CSV: `python -m cim26 verify runs/20261019T142131_8364eee3` (read only).

Main benchmark: 11 × 4 = 44 records; scalability is separate.
Each worker performs one untimed warmup. The first round freezes inner repetitions per model
to target 0.1 s. `compute_s` is batch time / inner repetitions;
`us_per_sample` also divides by the common N. Loop and Python API costs are included.
RSS is sampled at 0.002 s in a sampler thread, with common overhead;
very brief peaks may be missed and tiny memory deltas may be zero. No CPU energy was measured.
The raw source clock is retained without assuming a timezone. Calendar days follow that clock.
Energy is partial, on adjacent valid pairs with gap <= 7.5 min, never annualized.
Huld CSi CURRENT coefficients are frozen. DeltaT=3.0 °C is an assumption.
The laboratory reported temperature is treated as cell temperature at thermal equilibrium.
No field measurement was used to calibrate parameters. Multistart does not prove global uniqueness.
Day/round percentile bootstrap CIs are exploratory: dependence, seasonality and 11 rounds limit inference.
Input hashes, code hashes, environment, parameter diagnostics and every raw round are preserved.
