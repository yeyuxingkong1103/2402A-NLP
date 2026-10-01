$localPackages = Join-Path $PSScriptRoot ".deps"
if (Test-Path $localPackages) {
    $env:PYTHONPATH = $localPackages
}
$env:STREAMLIT_GLOBAL_DEVELOPMENT_MODE = "false"
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
python -m streamlit run (Join-Path $PSScriptRoot "app.py") --server.address 127.0.0.1 --server.fileWatcherType none
