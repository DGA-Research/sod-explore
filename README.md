# SOD Explorer

Streamlit application for interactively exploring the U.S. House Statement of Disbursements (SOD) data that you host in Google Cloud Storage. The app scans a bucket/prefix, lets you pick the reporting years to load, applies rich sidebar filters, and surfaces both summary and detail analytics with export options.

## Highlights

- **GCS-first workflow** - the UI is dedicated to a single, preconfigured bucket. Users simply select the reporting years they want to scan.
- **Automatic file metadata** - filenames are parsed to infer year, quarter, and whether a CSV represents summary or detail records.
- **Resilient CSV ingestion** - multiple encoding fallbacks, numeric/date coercion, and automatic `PERSON` extraction from organization names.
- **Filtering & views** - sidebar controls for organization, program, BOC/BOC code, vendors, free-text contains queries, amount sliders, and transaction date ranges.
- **Summary insights** - removes Statement roll-up rows (such as `OFFICE TOTALS`) before computing totals, averages, and per-organization charts.
- **Detail insights** - spend totals, medians, vendor counts, and top organizations/vendors, plus a streaming export pipeline that re-applies filters chunk by chunk.

## Prerequisites

- Python 3.10 or newer.
- Google Cloud Storage bucket containing raw SOD CSV files (organized by year folders or a common prefix).
- Service account credentials with read access to the bucket.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configure Secrets

Store credentials and bucket metadata in `.streamlit/secrets.toml` (never commit this file):

```toml
[gcs_sources.sod]
bucket = "sod-files"
prefix = "detail"
service_account_json = ""  # optional; falls back to [gcp_service_account]

[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----"
client_email = "service-account@project.iam.gserviceaccount.com"
```

- `gcs_sources` holds one or more named profiles; the first entry is used automatically.
- Provide either an inline JSON blob per profile or a shared `[gcp_service_account]` block.

## Running the App

```bash
streamlit run streamlit_app.py
```

During startup the sidebar prompts you to select one or more reporting years. The app lists CSV objects within the configured bucket/prefix, filters them down to the selected years, and loads the relevant files into pandas with caching enabled.

## Typical Workflow

1. Launch the app and choose the reporting years from the sidebar multiselect.
2. Pick Summary or Detail files from the grouped year list.
3. Apply sidebar filters (organizations, programs, vendors, amount ranges, transaction dates, etc.).
4. Review the KPIs and charts in the main area.
5. Download the filtered rows: summary data exports immediately; detail exports stream through chunks to keep memory steady.

## Troubleshooting

- **No CSV files found** - confirm the bucket/prefix and that the selected years exist; adjust secrets if needed.
- **Authentication errors** - verify the service account JSON in `.streamlit/secrets.toml` and that the account has `storage.objects.list`/`get` permissions.
- **Decoding errors** - the loader already retries UTF-8, CP1252, and Latin-1; if files still fail, inspect the raw encoding and extend `_read_csv_with_fallback` as needed.

## Repository Structure

- `streamlit_app.py` - main Streamlit application (file discovery, caching, filters, and exports).
- `requirements.txt` - pip dependencies.
- `sod_filtered.csv` / `sod_detail_filtered.csv` - example exports for reference.

## License

MIT License
