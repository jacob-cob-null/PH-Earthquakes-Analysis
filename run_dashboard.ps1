$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 is required to create the environment.' }
}
& $python -c 'import streamlit, plotly, geopandas, sklearn, folium, kagglehub; assert tuple(map(int, streamlit.__version__.split(".")[:2])) >= (1,67), "Streamlit 1.67+ is required"'
if ($LASTEXITCODE -ne 0) {
    & $python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
& $python -m streamlit run streamlit_app.py @args
