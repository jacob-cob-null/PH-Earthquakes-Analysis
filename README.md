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


## Interactive Streamlit dashboard

From this project folder, install `requirements.txt`, then run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Or run `./run_dashboard.ps1` in PowerShell; it prepares a local environment if needed. Open the localhost URL printed in the terminal. The dashboard includes maps/heat/exact grids, timelines, magnitude/depth distributions with lasso selection, tsunami flags, GMM profiles with linked maps, spatial clustering sensitivity, anomaly review, and a source/quality page. Sidebar filters apply across views. Plotly charts provide hover/zoom and PNG export; maps provide layer toggles and HTML downloads.

The app finds the original version-74 Kaggle cache locally. On another machine, upload the original USGS-format CSV in **Data source**, set `EARTHQUAKE_CSV` to its path, place it at `data/Philippine_Earthquakes_USGS.csv`, or use **Download pinned Kaggle version 74**. Downloads are explicit; the app does not silently switch to the latest catalog.

Keep `data/reference/arcgis_active_fault_trenches.geojson` and its `.source.json` together to show tectonic overlays. Missing/unverified geometry produces a visible warning. The tectonic GeoJSON and its provenance JSON are tracked deployment assets, explicitly allowed by `.gitignore`, so Streamlit Cloud receives the same validated snapshot as local runs. Other reference files and generated outputs remain ignored. Matching notebook profile memberships are reused when available; otherwise the app fits the same GMM procedure. Full notebook robustness tables are shown only with matching training records. Profile filters change the view of a fixed full-catalog model; spatial clustering and anomaly review refit the filtered subset.

Maps need online Leaflet/OpenStreetMap assets. Plotly charts and the saved static notebook figures remain useful without map tiles. For Streamlit Cloud, deploy `streamlit_app.py` from the repository; the tectonic overlays load from the included snapshot without a runtime ArcGIS request.


### Q1–Q4 figure coverage in Streamlit

The dashboard uses a compact Q1–Q4 navigation with no presentation takeaways. Q1 includes heat, exact equal-area counts and top cells, epicenters, a tectonic reference-only mode, paired magnitude/depth maps, and equal-span northern/southern close-ups. Q2 compares M ≥ 6 and M ≥ 7 simultaneously, shades potentially partial boundary periods, and retains an all-record calendar. Q3 includes matched-bin magnitude histograms with/without zero values, depth distribution, the selectable magnitude–depth scatter, depth categories, frequent reported depths, and the notebook's Spearman comparisons. Q4 uses two maps with the same initial extent and marker size, sea labels, actual coverage, and annual flagged-record counts. Short encoding/source captions remain; interpretations are left for the report.
