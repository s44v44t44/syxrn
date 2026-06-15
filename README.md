# JWO/OSH Online Issue Radar Dashboard

Streamlit submission build for the JWO/OSH online issue radar dashboard.

## Run Locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Deploy To Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload only this deployment folder's contents to the repository root.
3. Go to Streamlit Community Cloud and choose **Create app**.
4. Select the GitHub repository, branch, and `app.py`.
5. Deploy the app.
6. Copy the deployed `https://...streamlit.app` URL and convert it to a QR code.

## Included Data Profile

This folder contains a lightweight submission copy, not the full raw research archive.

- `dashboard_data/issue_briefs.parquet`: full dashboard issue brief table.
- `dashboard_data/issue_timeseries.parquet`: full issue time series used for custom period summaries.
- `dashboard_data/evidence_docs.parquet`: representative evidence documents, capped per issue.
- `dashboard_data/base_docs_light.parquet`: title-level document metadata for keyword charts.
- `dashboard_data/comments_light.parquet`: masked comments only; raw clean comment text is excluded.
- `dashboard_stats/`: statistical analysis outputs used by the statistics pages.

Large raw files, review queues, backup folders, and full comment text are intentionally excluded from this public submission build.

