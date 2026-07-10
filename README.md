# Active Source Seeking in Spatio-Temporal Fields

Reference Python implementation for the numerical example of the paper **Active Source Seeking in Spatio-Temporal Fields**.

The code simulates a scalar spatio-temporal field governed by an advection--diffusion PDE on a triangular finite-element mesh. A team of mobile sensors collects point measurements and moves according to a dual D-optimal active-sensing/source-seeking policy with safe motion. The source and field are estimated with a finite-element information marginalized particle filter (FE-IMPF).

## Main features

- finite-element advection--diffusion field model on the included coastal mesh;
- centralized FE-IMPF for joint field/source estimation;
- posterior-expected source Fisher information matrix;
- exact projected-information term for correlated innovations;
- TR-BFGS and fast first-order motion controllers;
- APF-based safe motion with boundary/obstacle and inter-agent repulsion;
- paper-scale, reference, and demo presets;
- optional exact posterior-FIM snapshots for candidate sensor locations.

## Repository structure

```text
active_source_seeking/      Core library modules
mesh/                       Gmsh mesh used in the example
run_paper_example.py        Main simulation script
requirements.txt            Python dependencies
pyproject.toml              Minimal package metadata
CITATION.cff                Citation metadata for GitHub
LICENSE                     MIT license
```

Generated outputs are intentionally ignored by Git and are written under `outputs/`.

## Installation

Use Python 3.10 or newer. Python 3.11 is recommended.

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

The Gmsh executable is not required because the mesh is already included as `mesh/bay_port_island.msh`.

## Quick start

Run a short student-friendly demo:

```bash
python run_paper_example.py --preset demo
```

Run a minimal smoke test:

```bash
python run_paper_example.py --preset demo --trials 1 --steps 5 --particles 30 --sensors 4
```

Run a deterministic reference case useful for checking exact-FIM snapshots and visualizations:

```bash
python run_paper_example.py --preset reference
```

## Paper-scale run

The paper preset uses the main numerical settings of the manuscript:

```bash
python run_paper_example.py --preset paper
```

This is computationally expensive: it runs 50 Monte Carlo trials, each with 400 time steps, 10 sensors, and 250 particles. On a laptop it can take many hours. For a first check, run one or a few trials:

```bash
python run_paper_example.py --preset paper --trials 1 --steps 400 --quiet
python run_paper_example.py --preset paper --trials 5 --steps 400 --quiet
```

You can override the main parameters from the command line:

```bash
python run_paper_example.py --preset paper --trials 10 --steps 200 --particles 250 --sensors 10 --seed 123
```

## Faster paper-like runs

The full paper preset is intentionally heavy. For teaching and development, the following commands preserve the same FE-IMPF estimation loop and active source-seeking structure while reducing runtime.

```bash
# Fast sanity check on a laptop
python run_paper_example.py --preset paper --trials 1 --steps 400 --particles 100 --controller gradient --motion-update-every 5 --quiet --no-save-traj

# Faster run that still uses TR-BFGS, recomputed every 5 steps
python run_paper_example.py --preset paper --trials 1 --steps 400 --particles 150 --controller trbfgs --motion-update-every 5 --quiet --no-save-traj

# Full paper-style motion logic for one trial
python run_paper_example.py --preset paper --trials 1 --steps 400 --controller trbfgs --motion-update-every 1
```

Useful switches:

- `--controller trbfgs`: paper-style trust-region BFGS motion update; most faithful, but slow.
- `--controller gradient`: one first-order active-sensing gradient evaluation; faster and useful for demos.
- `--motion-update-every N`: recompute the active-sensing motion direction every `N` steps and reuse it in between. The MMSE source-seeking direction is still updated every step.
- `--particles M`: reduce the number of source particles. Runtime is approximately linear in `M` for the particle filter and active-sensing parts.
- `--no-save-traj`: do not store full trial histories for animation.
- `--quiet`: reduce console output; `run.log` is still saved inside the dated run folder.
- `--debug`: enable detailed diagnostics for the particle filter and sensor steps.
- `--log-every N`: print progress every `N` time steps.

## Outputs

Each run is saved in a dated folder containing the main parameters:

```text
outputs/<preset>/<YYYYMMDD_HHMMSS>_<preset>_T<trials>_K<steps>_S<sensors>_M<particles>_D<diffusivity>_rf<rfilter>_<controller>_upd<motion_update_every>_seed<seed>/
```

For example:

```text
outputs/paper/20260622_213045_paper_T5_K400_S10_M150_D1500_rf0.0001_trbfgs_upd5_seed123/
```

A human-readable label can be added with `--run-name`:

```bash
python run_paper_example.py --preset paper --trials 5 --particles 150 --controller trbfgs --motion-update-every 5 --run-name m150_trbfgs_upd5
```

Each run folder includes:

- `run.log`: complete execution log;
- `config.json`: human-readable copy of the run configuration;
- `mc_results.npz`: raw Monte Carlo arrays and configuration;
- `rmse_position.png`: source position RMSE;
- `rmse_intensity.png`: source intensity RMSE;
- `field_error.png`: field-estimation error;
- quantile-band plots for MAP and MMSE estimates; when `--trials 1` is used, these plots show only the single realized error curve;
- optional `trials/` histories if trajectory saving is enabled.

## Exact posterior-FIM snapshots

The release can optionally save exact posterior-FIM maps for selected time steps. These maps are computed from the same FE-IMPF/controller expression used online:

```text
Fbar = sum_j w_j G_j^T Psi G_j
```

For each selected step, the code evaluates candidate single-sensor locations on a grid and saves:

- `fbar_grid`: exact `3 x 3` FIM at each valid candidate location;
- `info_logdet`: `logdet(W Fbar W + delta I)`;
- `info_trace`: `trace(W Fbar W)`;
- `team_fbar`: exact FIM for the current multi-sensor configuration;
- current sensor positions, MAP/MMSE estimates, true source, `delta`, and `w_u`.

Example:

```bash
python run_paper_example.py --preset reference --save-exact-fim --exact-fim-steps 1,30,60,110,180,240,last --exact-fim-grid 95 70
```

Exact FIM snapshots are written under:

```text
outputs/<preset>/<run-folder>/trials/trial_000_exact_fim/
```

This option is disabled by default because it is computationally expensive.

## Paper-consistent numerical setup

The `paper` preset uses:

- domain: `6000 x 4000 m` coastal region with port inlet and island obstacle;
- mesh: 411 nodes and 735 triangular elements in the included `.msh` file;
- sampling time: `dt = 5 s`;
- diffusivity: `kappa = 1500 m^2/s`;
- true source intensity: `u = 250`;
- true/filter process-noise levels: `sigma_w = 1e-4`, `sigma_w_filter = 2e-2`;
- true/filter measurement-noise variances: `r = 1e-5`, `r_filter = 1e-4`;
- Monte Carlo setup: `N_trials = 50`, `steps = 400`;
- field NRMSE: computed over `P_field = 200` sampled interior points.

For teaching or quick checks, the `demo` preset intentionally uses fewer trials, fewer steps, and fewer particles.

## Suggested use in teaching

A good first exercise is to run the quick demo, inspect `run_paper_example.py`, and vary one parameter at a time: the number of sensors, the number of particles, or the noise variance. Students can then compare MAP and MMSE source estimates and discuss how the active-sensing objective changes sensor trajectories.

## Citation

If you use this code, please cite:

```text
N. Forti, G. Battistelli, and L. Chisci,
"Active Source Seeking in Spatio-Temporal Fields,"
International Conference on Information Fusion, 2026.
```

A BibTeX entry can be added here once the final proceedings metadata are available.

## License

This project is released under the MIT License. See `LICENSE`.
