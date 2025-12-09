# Statement of Disbursements (SOD) Explorer
An interactive Streamlit application for browsing, filtering, analyzing, and exporting U.S. House **Statement of Disbursements (SOD)** data.  
Supports both **local bulk CSV folders** and **Google Cloud Storage (GCS)** sources, automatically detects reporting years and quarters, and provides data previews, summary metrics, charts, and filtered CSV downloads (including streaming large Detail datasets).

## Overview

The SOD Explorer lets users:

- Load **Summary** and **Detail** SOD CSV files from:
  - A local folder (default: `SOD_Bulk/`)
  - A `.zip` file that is auto-extracted into a folder
  - A **Google Cloud Storage bucket** (optional; requires secrets)
- Automatically detect:
  - Reporting **year**
  - **Quarter** (`Q1/Q2/Q3/Q4`)
  - **Summary** vs **Detail** data
- Apply powerful filters:
  - Member (`PERSON`)
  - `ORGANIZATION`
  - `PROGRAM`
  - `BUDGET OBJECT CLASS`
  - `BUDGET OBJECT CODE`
  - Vendor (`VENDOR NAME`) — Detail only
  - Date range — Detail only
  - Numeric filters for **AMOUNT** or **QTD/YTD** values
- View:
  - Top vendors
  - Top spending organizations
  - Summary tables
  - Interactive charts
- Download:
  - Preview data (first 1,000 rows)
  - Full filtered Summary CSV
  - Full filtered Detail CSV via **streaming**, safe for large datasets
  - Multi-file consolidated Detail CSV across selected years

This app is designed to make SOD analysis fast, intuitive, and scalable.

## Streamlit App Hosted: 
https://sod-explore-kdu9rhexngq8xefa3ldzrk.streamlit.app/

## Key Features

### Intelligent Dataset Detection
Each CSV is analyzed for:
- Year detection from filenames or CSV header rows  
- Quarter mapping from common SOD strings (`JAN-MAR`, `APR-JUN`, `JUL-SEP`, `OCT-DEC`)  
- Summary vs Detail type detection  

### Data Source
#### Google Cloud Storage (Optional)
Add this to `.streamlit/secrets.toml`:
```toml
[gcs_sources]
bucket = "your-bucket-name"
prefix = "optional/prefix/"

[gcp_service_account]
# full JSON key contents
```

### Data Normalization
The app:
- Uppercases and strips column names
- Extracts `PERSON` from `ORGANIZATION` when possible  
- Converts dates from multiple SOD formats  
- Parses numeric fields reliably despite formatting inconsistencies  

### Large-File Handling (Detail CSVs)
For Detail datasets, downloads use a **chunked streaming approach**:
- Reads CSV in chunks of 100k rows
- Applies filters per-chunk
- Writes out a single CSV with proper header handling
- Avoids memory overload on large Detail files

## Tech Stack
- **Python**
- **Streamlit**
- **pandas**
- **gcsfs** (optional; only needed for GCS sources)
- Standard Library: `pathlib`, `dataclasses`, `zipfile`, `re`, etc.

From `requirements.txt`:
```
streamlit==1.39.0
pandas==2.2.3
xlsxwriter==3.2.0
gcsfs==2024.10.0
```

## Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Set up GCS access
Create `.streamlit/secrets.toml`:

```toml
[gcs_sources]
bucket = "your-gcs-bucket"
prefix = "sod-data/"

[gcp_service_account]
# paste full JSON key here
```

## Running the App
```bash
streamlit run streamlit_app.py
```

Runs locally at:
```
http://localhost:8501
```

## Expected Folder Structure
```
.
├── streamlit_app.py        # Main app
├── requirements.txt        # Dependencies
├── .gitattributes
├── SOD_Bulk/               # Local SOD files (optional)
│   ├── SOD_Bulk.zip        # Auto-extracted if present
│   └── *.csv
```

## Data Model

### Summary Files  
Typical columns:
- YEAR
- QUARTER
- ORGANIZATION
- PROGRAM
- DESCRIPTION
- QTD AMOUNT  
- YTD AMOUNT  

### Detail Files  
Typical columns:
- PERSON
- ORGANIZATION
- PROGRAM
- VENDOR NAME
- TRANSACTION DATE
- AMOUNT  
- BUDGET OBJECT CLASS / CODE  
- Other SOD detail fields depending on quarter/year

## How It Works (Step-by-Step)

1. **Discover files**  
2. **User selects dataset**  
3. **App loads & normalizes CSV**  
4. **User applies filters**  
5. **App renders metrics, charts, tables**  
6. **User downloads filtered results**  
