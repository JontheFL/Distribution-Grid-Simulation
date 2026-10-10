# Distribution Grid Simulation

Python-based distribution-grid analysis and optimization project built with
[pandapower](https://www.pandapower.org/).

## What it demonstrates

- Modeling a low-voltage distribution network with transformers, lines, loads,
  and distributed generation
- Running time-series power-flow simulations from measured load and generation
  profiles
- Using a genetic algorithm to optimize the allocation of profiles and reduce
  peak line loading
- Ranking buses by short-circuit impedance to identify electrically weaker
  locations
- Producing analysis figures and reproducible results with NumPy, pandas, and
  Matplotlib

## Run

```bash
pip install -r requirements.txt
python genetic_algorithm.py
```

The main scripts are [`genetic_algorithm.py`](genetic_algorithm.py) for
optimization and [`grid_analyzer.py`](grid_analyzer.py) for grid and
short-circuit analysis. Input profiles are stored in [`Data/`](Data/), with
supporting notebooks and results in [`Notebooks/`](Notebooks/) and
[`Figures/`](Figures/).
