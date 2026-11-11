# Philippine Earthquake Analysis

An EDA notebook with static geographic counts, fault/trench context, and an interactive Folium heatmap. Interpretation prompts are left for the author. A story-focused GMM extension precedes the optional spatial-clustering and anomaly appendix.

## Setup in PowerShell

```powershell
Set-Location 'C:\Users\Jacob\Documents\Earthquake_Analysis'
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name earthquake-analysis --display-name 'Earthquake Analysis (.venv)'
.\.venv\Scripts\python.exe -m jupyterlab Earthquake_PTF10.ipynb
```

Activation is not required. The commands use the virtual environment directly.
Select **Earthquake Analysis (.venv)** in Jupyter. In VS Code, you can instead
select the project's `.venv` Python interpreter as the notebook kernel.

## Start here

1. Run the imports and acquisition cells. KaggleHub downloads to its own cache.
2. If multiple CSV files appear, set `CSV_FILENAME` to the intended relative path.
3. Record the resolved download path/version, download date, and actual coverage.
4. Keep `df_raw` unchanged. Make your decisions and write your code using `df`.
5. Add verified fault/trench files to `data/reference/` and fill the source register.
6. Complete descriptive statistics and EDA before considering the ML section.

The notebook makes no assumption about row count, column meanings, or actual date
coverage. Check the dataset documentation. If download access fails, inspect the
underlying error; Kaggle authentication may be required in your environment.
Never put API tokens in the notebook or commit them.

## ML dependencies

`scikit-learn` is included in `requirements.txt` for the GMM extension and appendix experiments. Install the requirements before running the notebook from the top. The analytical extension remains optional to the EDA story.

## Folders

- `data/reference/`: externally sourced fault/trench and boundary files.
- `data/processed/`: derived datasets you choose to export.
- `outputs/`: figures or tables you choose to save.

Keep source geographic files unchanged and record their attribution and coverage.
The downloaded earthquake CSV remains in the KaggleHub cache.

## Reproducibility

Dependencies are intentionally unpinned for initial setup. Once your environment
works, record its versions before submission:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > requirements-lock.txt
```

Restart the kernel and run your completed notebook from top to bottom. A future
KaggleHub download may resolve to a newer dataset version: retain the source/version
used for the submission and do not silently refresh it during final verification.

## Geographic maps and appendix

The loader caches the recovered ActiveFaultTrenches service and provenance.
Publisher: aechanes_geodata. Original authority and mapping date are unspecified;
this is a provisional, incomplete third-party reference. Actual missing geometries
are listed at runtime. Cached geometry without provenance requires source review.
Faults, approximate traces, trenches, and collision zones have distinct styles.

Outputs include q1_geographic_heatmap.html, static count and tectonic-context
PNGs, and grid counts in CSV. Interactive maps require online tiles/JavaScript;
static PNGs remain usable offline. All geographic summaries use the same subset.
If tectonic data cannot load, maps prominently state that the overlay is unavailable.

DBSCAN is an appendix comparison: the 25 km / 20-record baseline is compared
with radii 15/25/40 km and min_samples 10/20/40. Write your own conclusion from
the sensitivity table; limited added value is a valid finding. The anomaly review
remains optional. Share source/provenance files or rerun acquisition on Colab.


## Story-focused ML extension

Find **ML extension — Do similar earthquakes occur in similar places?** before Appendix A. Run the notebook from the top. It fits magnitude–depth Gaussian mixtures on one reported magnitude scale, maps the profiles, and checks sensitivity to seeds, depth transformation, and repeated catalog depths. The model can select one component. Write your own observations in the supplied prompts.

Static presentation figures: `outputs/ml_profile_characteristics.png`, `outputs/ml_profile_geography.png`, and `outputs/ml_profile_challenge.png`. Model selection, memberships/probabilities, subset audit, summaries, depth-category comparison, and robustness checks are saved as `outputs/ml_profile_*.csv`. BIC values are comparable only within the same variant. All maps retain the project's provisional tectonic attribution.
