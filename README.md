# Philippine Earthquake Analysis

A starter notebook for your own descriptive statistics and EDA. Only imports,
download/loading code, and section placeholders are supplied. No cleaning,
analysis, charts, models, or conclusions have been implemented or executed.

## Setup in PowerShell

```powershell
Set-Location 'C:\Users\Jacob\Documents\Earthquake_Analysis'
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name earthquake-analysis --display-name 'Earthquake Analysis (.venv)'
.\.venv\Scripts\python.exe -m jupyterlab earthquake_analysis.ipynb
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

## Optional ML dependencies

Only if you choose to work on the extension:

```powershell
.\.venv\Scripts\python.exe -m pip install scikit-learn
```

Then uncomment only the imports you need. No model code or parameters are supplied.

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
