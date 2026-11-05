# Changelog

Record of every change made to the original source files when this repository was assembled from the working
folder. Scientific logic, numerical settings and plot styling were not changed. Original files in the working
folder were copied, never moved or edited.

## Repository assembly

### File selection and placement (copies, byte-identical unless listed below)

- `GAN_OSC/main.tex`, `si.tex`, `references.bib`, `figures/*.pdf` -> `manuscript/`. The .tex files were not edited:
  all 13 `\includegraphics{figures/...pdf}` targets and `\bibliography{references}` resolve inside `manuscript/`.
- `try_7th_august/figure_scripts_split/` scripts -> `scripts/figures/`; its `plot_data_all12/` CSVs -> `data/figure_inputs/`.
- Pipeline scripts -> `src/{generation,postprocessing,postgen,synthesizability,plotting,analysis}/` (see `docs/PIPELINE.md`).
  `postgen_pipeline.py` is the run-time copy from `try_7th_august/latest_results/scripts/` (`_MAX_CHEMNET_SMILES_LEN = 350`,
  the value used for the reported run), not the newer root copy (349).
- `hpc/sub_140.sh` is the copy with `SKIP_GENERATION=0` (full generation). The copy in `latest_results/scripts/` has
  `SKIP_GENERATION=1` (re-score only) and is otherwise identical.
- `data/input/`, `data/generated_libraries/`, `results/{summary,postgen,synthesizability,tier2_analysis}/` hold the input data,
  libraries and output tables listed in the manifest. `results/reference_figures_png/` holds the 12 PNGs
  from `try_7th_august/Final_manuscript_figures/`.
- Left out: zip archives, caches, job logs, superseded and duplicate versions, old .bib variants, draft text
  files, `New_prediction_method/` (not used by this manuscript), the older all-in-one figure script, and every file larger than about 50 MB.

### Code changes

Figure scripts (`scripts/figures/`)

- `_common_style.py`: `parse_common_args` help text and `resolve_dirs` defaults. `--data-dir` now defaults to
  `<repo root>/data/figure_inputs` (was the script's own folder) and `--outdir` to `<repo root>/results/regenerated_figures`
  (was `<data-dir>/Final_manuscript_figures`). The repo root is taken from the script location, so the working directory does not matter.
  Style, colours and helpers are untouched.
- `run_all.py`: `--data-dir` is optional and defaults the same way. Usage text updated.
- `01`..`12_*.py`: docstring usage lines only (`plot_data_all12` -> `data/figure_inputs`, default output folder). No code lines changed.
- `README.md`: paths and defaults updated to the repository layout.

Analysis scripts (`src/analysis/`): path constants only.

- `frechet_baseline_analysis.py`: reads `data/input/DMPNN.csv` and `Guzik_data.csv` (was `<script dir>/../`), writes to
  `results/tier2_analysis/` (was `<script dir>/tier2_analysis`). Docstring: usage and output folder.
- `make_representative_figure.py`: reads `data/figure_inputs/high_performance_pairs.csv` and
  `results/tier2_analysis/tier2_sweet_spot_candidates.csv`, writes both PNGs to `results/tier2_analysis/`
  (was `latest_results/...`, `tier2_analysis/`, `figures/`). Docstring: usage.
- `tier2_chemical_analysis.py`: `HP_CSV` -> `data/figure_inputs/high_performance_pairs.csv`; library CSVs ->
  `data/generated_libraries/{donor,acceptor}_molecule_predictions.csv` (byte-identical to the `plot_data` versions the script read);
  `PAIR_PRED_CSV` -> `data/large_inputs/pair_predictions.csv` (not shipped); output -> `results/tier2_analysis/`.
  Docstring states how the large input is obtained.
- `create_eloss_vs_pce_corrected.py`: `DMPNN_CSV` -> `data/input/DMPNN.csv`; the two large pools ->
  `data/large_inputs/` (not shipped); outputs -> `results/eloss_vs_pce/`. Docstring states how the large inputs are obtained.

HPC drivers (`hpc/`)

- `sub_140.sh`, `sub_postgen_full_140.sh`, `sub_synth_scoring_240.sh`: replaced the hard-coded
  `source /home/<user>/Programs/Conda/bin/activate` with `CONDA_ACTIVATE="${CONDA_ACTIVATE:-$HOME/miniconda3/bin/activate}"`
  and `source "$CONDA_ACTIVATE"` (also inside the `srun bash -lc` block of the synthesizability driver).
- `sub_synth_scoring_240.sh`: the absolute cluster path in `PAIRS_CSV` became
  `${PAIRS_CSV:-runs/22051/merged/results/generated_molecules_with_predictions.csv}`, relative to the submit directory.
- New `setup_run_dir.sh`: builds the flat `run/` directory the stage 1 to 3 scripts expect (copies scripts, `DMPNN.csv`,
  `Guzik_data.csv`, and RDKit's `sascorer.py` / `fpscores.pkl.gz` when found).

Header notes (comment lines only, no code changed): `GAN_240_final.py`, `merge_rank_outputs_final.py`, `postgen_pipeline.py`,
`manuscript_plots.py`, `manuscript_plots_revised.py`, `synthesizability_scoring.py` each received a "REPOSITORY NOTE" comment block
stating the stage, how to run it and how any large or HPC-only input is obtained.

New files: `.gitignore`, `docs/PIPELINE.md`, `hpc/setup_run_dir.sh`, this changelog.

### Documentation and metadata added in the final step (no code changed)

- `README.md`, `docs/DATA_AVAILABILITY.md`, `data/large_inputs/README.md` (placeholder, listed in `.gitignore` as the one tracked file in that folder).
- `requirements.txt` and `environment.yml`, pinned to the versions reported by `py -3.12 -m pip list` (Python 3.12.0).
- `LICENSE` (MIT, a default chosen because the manuscript states no licence), `CITATION.cff` (authors and title from `manuscript/main.tex`).
- `.gitignore`: added virtual environments, editor and OS files, temporary files and credential file patterns.
- This changelog: the scan description no longer quotes literal token prefixes.

### Secrets and hygiene scan

Scanned every text file (.py, .sh, .tex, .bib, .md, .txt, .csv) for access-token and key patterns (GitHub, AWS and generic
API-key prefixes, private-key headers, "password", "api_key"), e-mail addresses, IPv4 addresses, user names, home and drive paths and
cluster paths, and the PDF/PNG files for embedded paths and metadata.

| Finding | Where | Action |
|---|---|---|
| Cluster user name and home path (`/home/<user>/Programs/Conda/bin/activate`) | 3 x `hpc/*.sh` (4 lines) | replaced by `CONDA_ACTIVATE` variable |
| Absolute cluster path (`/scratch/<user>/project_solarcell/...`) | `hpc/sub_synth_scoring_240.sh` | replaced by relative default, overridable |
| Author e-mail addresses (two institutional addresses) | `manuscript/main.tex`, `manuscript/si.tex` (`\date` line) | kept: published author contact of the manuscript |
| Author names | .tex, .bib | kept |
| Development folder names (`try_7th_july`, `latest_results`) | comments in `GAN_240_final.py`, `merge_rank_outputs_final.py`; default of `manuscript_plots_revised.py --data-dir` | kept: not personal, no logic dependence beyond the documented default |
| Tokens, passwords, API keys, private keys, IP addresses, Windows drive paths | none found | none |
| PDF/PNG metadata | Matplotlib/cairo producer strings only | none |

No credentials were provided to or written by the build.

### Verification

Run once with `py -3.12` (Matplotlib 3.10.1) from a temporary copy of the repository, working directory at the repo root:

- `python scripts/figures/run_all.py` ran all 12 figure scripts without errors. Every PNG has the same file name, the same pixel size
  and the same bytes as `results/reference_figures_png/` (12 of 12), and `combined_space_analysis_table.tex` is regenerated.
- `src/analysis/frechet_baseline_analysis.py` ran; all outputs match the committed `results/tier2_analysis/` files except the two
  per-split CSVs, which differ only in the last digits of floating-point values (summary text identical).
- `src/analysis/make_representative_figure.py` ran from the repo data.
- `src/analysis/tier2_chemical_analysis.py` stops with `FileNotFoundError` for `data/large_inputs/pair_predictions.csv`, and
  `create_eloss_vs_pce_corrected.py` warns that its two large inputs are missing. Both are documented as HPC-only inputs.
- `sh -n` passes for the four shell scripts; `hpc/setup_run_dir.sh` builds the run directory.
- The PNGs are not pixel-comparable with the manuscript PDFs, which were exported with a different text setup (see `docs/PIPELINE.md`).
