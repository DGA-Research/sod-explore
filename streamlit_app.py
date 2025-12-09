"""
Streamlit application for exploring U.S. House Statement of Disbursements (SOD) CSV files.

The app scans the local ``SOD_Bulk`` folder, lets users choose which detail/summary files to
load, exposes sidebar filters, and visualizes the filtered slice. Launch with:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Set
from zipfile import ZipFile

import pandas as pd
import streamlit as st

try:  # Optional dependency for GCS access.
    import gcsfs  # type: ignore
except ImportError:  # pragma: no cover - handled in UI
    gcsfs = None

# Default location that ships with the repository.
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "SOD_Bulk"

# Limit for how many categorical options we surface as multiselects before
# falling back to text queries to avoid rendering massive lists.
SIDEBAR_OPTION_LIMIT = 500

# Default remote profile used when no secrets-based entry exists.
DEFAULT_GCS_PROFILE = {
    "bucket": "sod-files",
    "prefix": "",  # or e.g. "detail/" if you keep data in a subfolder
    "service_account_json": None,
}

# Tokens that help infer reporting quarter from a file name.
QUARTER_TOKENS: List[Tuple[str, Tuple[str, str]]] = [
    ("JAN-MAR", ("Q1", "Jan–Mar")),
    ("JANUARY-MARCH", ("Q1", "Jan–Mar")),
    ("JAN_MAR", ("Q1", "Jan–Mar")),
    ("APR-JUN", ("Q2", "Apr–Jun")),
    ("APRIL-JUNE", ("Q2", "Apr–Jun")),
    ("APR-JUNE", ("Q2", "Apr–Jun")),
    ("APR_JUN", ("Q2", "Apr–Jun")),
    ("JUL-SEP", ("Q3", "Jul–Sep")),
    ("JUL-SEPT", ("Q3", "Jul–Sep")),
    ("JULY-SEPT", ("Q3", "Jul–Sep")),
    ("JULY-SEPTEMBER", ("Q3", "Jul–Sep")),
    ("JUL-SEPTEMBER", ("Q3", "Jul–Sep")),
    ("OCT-DEC", ("Q4", "Oct–Dec")),
    ("OCTOBER-DECEMBER", ("Q4", "Oct–Dec")),
    ("OCT-DECEMBER", ("Q4", "Oct–Dec")),
]

QUARTER_SORT_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
DATE_FORMATS = ("%d-%b-%y", "%d-%b-%Y", "%m/%d/%Y")
PERSON_PREFIX_PATTERN = re.compile(
    r"^(THE\s+HONORABLE|THE\s+HON\.|HONORABLE|HON\.?|REPRESENTATIVE|REP\.?|SENATOR|SEN\.?|DELEGATE|DEL\.?|RESIDENT\s+COMMISSIONER)\s+",
    re.IGNORECASE,
)
REMOTE_SCHEMES = ("gs://",)


def _is_remote_path(path_str: str) -> bool:
    return any(path_str.startswith(prefix) for prefix in REMOTE_SCHEMES)


def _extract_filename(path_str: str) -> str:
    if not path_str:
        return ""
    if _is_remote_path(path_str):
        return path_str.rstrip("/").split("/")[-1]
    return Path(path_str).name


def _path_exists(path_str: str) -> bool:
    if _is_remote_path(path_str):
        return True
    return Path(path_str).exists()


def _get_secret_dict(key: str) -> Dict[str, Dict[str, object]]:
    try:
        value = st.secrets.get(key)  # type: ignore[attr-defined]
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _serialize_secret_mapping(secret_obj: object) -> Optional[str]:
    if secret_obj is None:
        return None
    if isinstance(secret_obj, str):
        return secret_obj.strip() or None
    if hasattr(secret_obj, "to_dict"):
        secret_obj = secret_obj.to_dict()  # type: ignore[attr-defined]
    if isinstance(secret_obj, dict):
        return json.dumps(secret_obj)
    try:
        return json.dumps(secret_obj)
    except TypeError:
        return None


def resolve_service_account_json(explicit_json: Optional[str]) -> Optional[str]:
    explicit = (explicit_json or "").strip()
    if explicit:
        return explicit
    try:
        secret_account = st.secrets.get("gcp_service_account")  # type: ignore[attr-defined]
    except Exception:
        secret_account = None
    return _serialize_secret_mapping(secret_account)


def get_configured_gcs_sources() -> Dict[str, Dict[str, str]]:
    sources = _get_secret_dict("gcs_sources")
    normalized: Dict[str, Dict[str, str]] = {}
    for name, config in sources.items():
        if not isinstance(config, dict):
            continue
        bucket = str(config.get("bucket", "")).strip()
        if not bucket:
            continue
        sa_value = config.get("service_account_json", "")
        normalized[name] = {
            "bucket": bucket,
            "prefix": str(config.get("prefix", "") or "").strip(),
            "service_account_json": (_serialize_secret_mapping(sa_value) or "").strip(),
        }
    return normalized


def get_preloaded_gcs_profile() -> Dict[str, Optional[str]]:
    configured = get_configured_gcs_sources()
    if configured:
        first_key = sorted(configured.keys())[0]
        return configured[first_key]
    return DEFAULT_GCS_PROFILE


@dataclass
class FileMeta:
    """Metadata about one CSV file on disk."""

    path: str
    data_type: str  # Detail or Summary
    year: Optional[int]
    quarter: Optional[str]
    quarter_label: Optional[str]

    @property
    def label(self) -> str:
        year_txt = str(self.year) if self.year else "Year ?"
        quarter_txt = self.quarter_label or "Quarter ?"
        return f"{year_txt} {quarter_txt} • {self.data_type} • {self.filename}"

    @property
    def filename(self) -> str:
        return _extract_filename(self.path)

    @property
    def sort_key(self) -> Tuple[int, int, str]:
        year_val = self.year or 0
        quarter_val = QUARTER_SORT_ORDER.get(self.quarter or "", 0)
        return (year_val, quarter_val, self.filename)


def normalize_name(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "-", name.upper())


def infer_quarter_bits(name: str) -> Tuple[Optional[str], Optional[str]]:
    normalized = normalize_name(name)
    for token, (quarter_code, quarter_label) in QUARTER_TOKENS:
        if token in normalized:
            return quarter_code, quarter_label
    return None, None


def infer_data_type(name: str) -> str:
    normalized = normalize_name(name)
    if "SUMMARY" in normalized or "SUMM" in normalized:
        return "Summary"
    return "Detail"


def infer_year(name: str) -> Optional[int]:
    match = re.search(r"(20\d{2}|19\d{2})", name)
    if match:
        return int(match.group(1))
    return None


def parse_year_tokens(text: str) -> List[int]:
    tokens = re.split(r"[,\s]+", text.strip())
    years: Set[int] = set()
    for token in tokens:
        if not token:
            continue
        if re.fullmatch(r"\d{4}", token):
            years.add(int(token))
    return sorted(years)


def filter_paths_by_year(paths: Iterable[str], years: Tuple[int, ...]) -> List[str]:
    targets = set(years)
    filtered: List[str] = []
    for path_str in paths:
        filename = _extract_filename(path_str)
        inferred = infer_year(filename)
        if inferred is None:
            continue
        if inferred in targets:
            filtered.append(path_str)
    return filtered


def infer_year_from_file(csv_path: Path) -> Optional[int]:
    """Fallback: peek inside the CSV and capture the first year token."""

    try:
        with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
            # Skip header
            header = handle.readline()
            for _ in range(5):  # Inspect a few lines in case of blanks
                line = handle.readline()
                if not line:
                    break
                match = re.search(r"(20\d{2}|19\d{2})", line)
                if match:
                    return int(match.group(1))
    except OSError:
        return None
    return None


def _build_file_meta(csv_paths: Iterable[str]) -> List[FileMeta]:
    files: List[FileMeta] = []
    for path_str in csv_paths:
        filename = _extract_filename(path_str)
        year = infer_year(filename)
        if year is None and not _is_remote_path(path_str):
            year = infer_year_from_file(Path(path_str))
        quarter_code, quarter_label = infer_quarter_bits(filename)
        data_type = infer_data_type(filename)
        files.append(
            FileMeta(
                path=path_str,
                data_type=data_type,
                year=year,
                quarter=quarter_code,
                quarter_label=quarter_label,
            )
        )
    return sorted(files, key=lambda meta: meta.sort_key)


@st.cache_data(show_spinner=False)
def discover_local_files(data_dir_str: str) -> List[FileMeta]:
    data_dir = Path(data_dir_str)
    if not data_dir.exists():
        return []

    csv_paths = [str(path) for path in sorted(data_dir.glob("*.csv"))]
    return _build_file_meta(csv_paths)


@st.cache_resource(show_spinner=False)
def _get_gcs_filesystem(service_account_json: Optional[str]):
    if gcsfs is None:  # pragma: no cover - surfaced via UI
        raise RuntimeError("gcsfs is not installed. Run `pip install gcsfs`.")

    token = json.loads(service_account_json) if service_account_json else None
    return gcsfs.GCSFileSystem(token=token)


@st.cache_data(show_spinner=True)
def discover_gcs_years(
    bucket: str, prefix: str, service_account_json: Optional[str]
) -> Tuple[List[int], bool]:
    bucket = bucket.strip()
    if bucket.startswith("gs://"):
        bucket = bucket[5:]
    if not bucket:
        return [], False

    prefix = prefix.strip().lstrip("/")
    search_root = f"{bucket}/{prefix}" if prefix else bucket

    try:
        fs = _get_gcs_filesystem(service_account_json)
    except Exception as exc:  # pragma: no cover - surfaced via UI
        st.sidebar.error(f"GCS authentication failed while listing years: {exc}")
        return [], False

    try:
        entries = fs.ls(search_root, detail=False)
    except FileNotFoundError:
        st.sidebar.error(f"GCS path not found: {search_root}")
        return [], False
    except Exception as exc:  # pragma: no cover
        st.sidebar.error(f"Failed to inspect gs://{search_root}: {exc}")
        return [], False

    years: Set[int] = set()
    dir_years: Set[int] = set()
    for entry in entries:
        normalized = entry.rstrip("/")
        name = normalized.split("/")[-1]
        inferred = infer_year(name)
        if inferred is not None:
            years.add(inferred)
        if entry.endswith("/") and re.fullmatch(r"\d{4}", name):
            dir_years.add(int(name))

    use_year_dirs = len(dir_years) > 0
    return sorted(years), use_year_dirs


@st.cache_data(show_spinner=True)
def discover_gcs_files(
    bucket: str,
    prefix: str,
    service_account_json: Optional[str],
    years: Optional[Tuple[int, ...]] = None,
    use_year_dirs: bool = False,
) -> List[FileMeta]:
    bucket = bucket.strip()
    if bucket.startswith("gs://"):
        bucket = bucket[5:]
    if not bucket:
        return []

    prefix = prefix.strip().lstrip("/")
    search_root = f"{bucket}/{prefix}" if prefix else bucket

    try:
        fs = _get_gcs_filesystem(service_account_json)
    except Exception as exc:  # pragma: no cover - surfaced via UI
        st.sidebar.error(f"GCS authentication failed: {exc}")
        return []

    year_filters: Tuple[int, ...] = tuple(sorted(set(years))) if years else ()

    csv_paths: List[str] = []
    if year_filters and use_year_dirs:
        normalized_root = search_root.rstrip("/")
        search_roots = [f"{normalized_root}/{year}".lstrip("/") for year in year_filters]
        for root in search_roots:
            try:
                objects = fs.find(root)
            except FileNotFoundError:
                continue
            except Exception as exc:  # pragma: no cover
                st.sidebar.error(f"Failed to list gs://{root}: {exc}")
                continue
            csv_paths.extend(f"gs://{obj}" for obj in objects if obj.lower().endswith(".csv"))
        if not csv_paths and year_filters:
            st.sidebar.info("Falling back to prefix-wide scan; year folders not found.")
    if not csv_paths:
        try:
            objects = fs.find(search_root)
        except FileNotFoundError:
            return []
        except Exception as exc:  # pragma: no cover
            st.sidebar.error(f"Failed to list gs://{search_root}: {exc}")
            return []
        csv_paths = [f"gs://{obj}" for obj in objects if obj.lower().endswith(".csv")]

    if year_filters:
        csv_paths = filter_paths_by_year(csv_paths, year_filters)

    csv_paths = sorted(csv_paths)
    if not csv_paths:
        return []
    return _build_file_meta(csv_paths)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    """Attempt to read a CSV trying multiple encodings commonly used in SOD files."""

    encodings = ("utf-8", "cp1252", "latin-1")
    last_error: Optional[Exception] = None
    read_kwargs = {"low_memory": False}
    for encoding in encodings:
        try:
            return pd.read_csv(csv_path, encoding=encoding, encoding_errors="ignore", **read_kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise last_error or UnicodeDecodeError(
        "utf-8", b"", 0, 1, "Failed to decode CSV with fallback encodings."
    )


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse date strings using the most common SOD formats before falling back."""

    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        if not parsed.isna().all():
            return parsed
    return pd.to_datetime(series, errors="coerce")


def extract_person_name(value: object) -> Optional[str]:
    """Extract the member name (if present) from an organization label."""

    if not isinstance(value, str):
        return None

    text = value.strip()
    text = re.sub(r"^\d{4}\s+", "", text)  # drop leading year tokens
    text = text.strip(" ,.-")
    if not text:
        return None

    match = PERSON_PREFIX_PATTERN.match(text)
    if not match:
        return None

    text = PERSON_PREFIX_PATTERN.sub("", text)
    text = text.strip(" ,.-")
    if not text:
        return None

    # Remove extra descriptors following punctuation.
    for splitter in (",", " - ", "\u2013", "("):
        if splitter in text:
            text = text.split(splitter, 1)[0].strip()

    if not text:
        return None

    name = " ".join(text.split())
    name = name.upper()
    # Filter out overly short strings (e.g., accidental single tokens).
    if len(name) < 3 or " " not in name:
        return None
    return name


@st.cache_data(show_spinner=True)
def load_data(file_paths: Tuple[str, ...], data_type: str) -> pd.DataFrame:
    if not file_paths:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for path_str in file_paths:
        if not _path_exists(path_str):
            continue
        df = _read_csv_with_fallback(path_str)
        df.columns = [col.strip().upper() for col in df.columns]
        df["SOURCE_FILE"] = _extract_filename(path_str)
        if "ORGANIZATION" in df.columns and "PERSON" not in df.columns:
            df["PERSON"] = df["ORGANIZATION"].apply(extract_person_name)

        if data_type == "Detail" and "AMOUNT" in df.columns:
            df["AMOUNT"] = _to_numeric(df["AMOUNT"])
            for date_col in ("TRANSACTION DATE", "PERFORM START DT", "PERFORM END DT"):
                if date_col in df.columns:
                    df[date_col] = _parse_dates(df[date_col])
        else:
            for value_col in ("QTD AMOUNT", "YTD AMOUNT"):
                if value_col in df.columns:
                    df[value_col] = _to_numeric(df[value_col])

        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def sidebar_dataset_picker(files: List[FileMeta]) -> Tuple[str, List[FileMeta], List[Optional[int]]]:
    st.sidebar.header("Dataset")

    data_type = st.sidebar.selectbox("Data type", options=["Summary", "Detail"], index=0)
    filtered = [meta for meta in files if meta.data_type == data_type]
    if not filtered:
        st.sidebar.info(f"No {data_type.lower()} files found in the data folder.")
        return data_type, [], []

    year_groups: Dict[Optional[int], List[FileMeta]] = {}
    for meta in filtered:
        year_groups.setdefault(meta.year, []).append(meta)

    def year_label(year: Optional[int]) -> str:
        return str(year) if year is not None else "Unknown Year"

    def sort_key(year: Optional[int]) -> Tuple[int, int]:
        # Known years first, ascending; Unknown at the end.
        return (1 if year is None else 0, year or 0)

    sorted_years = sorted(year_groups.keys(), key=sort_key)
    label_to_files: Dict[str, List[FileMeta]] = {}
    option_labels: List[str] = []
    for year in sorted_years:
        metas = year_groups[year]
        label = f"{year_label(year)} ({len(metas)} file{'s' if len(metas) != 1 else ''})"
        option_labels.append(label)
        label_to_files[label] = metas

    default_selection: List[str] = []
    known_years = [year for year in sorted_years if year is not None]
    if known_years:
        latest_year = max(known_years)
        label_prefix = year_label(latest_year)
        for label in option_labels:
            if label.startswith(label_prefix):
                default_selection = [label]
                break
    elif option_labels:
        default_selection = [option_labels[-1]]

    selected_labels = st.sidebar.multiselect(
        "Reporting years",
        options=option_labels,
        default=default_selection,
    )
    selected_files = [meta for label in selected_labels for meta in label_to_files.get(label, [])]
    selected_years_set = {
        meta.year
        for label in selected_labels
        for meta in label_to_files.get(label, [])
    }
    selected_years = sorted(
        selected_years_set,
        key=lambda year: (1 if year is None else 0, year or 0),
    )

    st.sidebar.caption("Tip: limit selections if you run into memory constraints.")
    return data_type, selected_files, selected_years


def build_filter_controls(df: pd.DataFrame, data_type: str) -> Dict[str, object]:
    filters: Dict[str, object] = {}
    if df.empty:
        return filters

    st.sidebar.header("Filters")

    if "PERSON" in df.columns and df["PERSON"].notna().any():
        person_options = sorted(df["PERSON"].dropna().unique())
        if len(person_options) > SIDEBAR_OPTION_LIMIT:
            filters["person_query"] = st.sidebar.text_input("Person contains")
        else:
            filters["person"] = st.sidebar.multiselect("Person", options=person_options)

    for column_label, filter_key in (
        ("ORGANIZATION", "organization"),
        ("PROGRAM", "program"),
        ("BUDGET OBJECT CLASS", "boc"),
        ("BUDGET OBJECT CODE", "boc_code"),
    ):
        if column_label in df.columns:
            options = sorted(df[column_label].dropna().unique())
            if len(options) > SIDEBAR_OPTION_LIMIT:
                filters[f"{filter_key}_query"] = st.sidebar.text_input(f"{column_label.title()} contains")
            else:
                filters[filter_key] = st.sidebar.multiselect(column_label.title(), options=options)

    if data_type == "Detail" and "VENDOR NAME" in df.columns:
        filters["vendor_query"] = st.sidebar.text_input("Vendor contains")

    amount_column = "AMOUNT" if data_type == "Detail" else None
    if data_type == "Summary":
        available_value_cols = [col for col in ("QTD AMOUNT", "YTD AMOUNT") if col in df.columns]
        if available_value_cols:
            amount_column = st.sidebar.selectbox(
                "Value column",
                options=available_value_cols,
                format_func=lambda col: "Quarter-to-date" if col.startswith("QTD") else "Year-to-date",
            )

    filters["amount_column"] = amount_column
    if amount_column and amount_column in df.columns:
        cleaned = df[amount_column].dropna()
        if not cleaned.empty:
            min_val = float(cleaned.min())
            max_val = float(cleaned.max())
            if min_val == max_val:
                max_val = min_val + 1.0
            filters["amount_range"] = st.sidebar.slider(
                "Amount range",
                min_value=float(min_val),
                max_value=float(max_val),
                value=(float(min_val), float(max_val)),
                step=float(max(1.0, (max_val - min_val) / 100)),
            )

    if data_type == "Detail" and "TRANSACTION DATE" in df.columns:
        date_series = df["TRANSACTION DATE"].dropna()
        if not date_series.empty:
            min_date = date_series.min().date()
            max_date = date_series.max().date()
            filters["transaction_dates"] = st.sidebar.date_input(
                "Transaction date range",
                value=(min_date, max_date),
            )

    return filters


def apply_filters(df: pd.DataFrame, filters: Dict[str, object], data_type: str) -> pd.DataFrame:
    filtered = df.copy()

    column_map = {
        "organization": "ORGANIZATION",
        "program": "PROGRAM",
        "boc": "BUDGET OBJECT CLASS",
        "boc_code": "BUDGET OBJECT CODE",
        "person": "PERSON",
    }
    for key, column in column_map.items():
        if column not in filtered.columns:
            continue
        values = filters.get(key)
        if values:
            filtered = filtered[filtered[column].isin(values)]
        query_value = filters.get(f"{key}_query")
        if query_value:
            filtered = filtered[
                filtered[column].astype(str).str.contains(str(query_value), case=False, na=False)
            ]

    vendor_query = filters.get("vendor_query")
    if vendor_query and "VENDOR NAME" in filtered.columns:
        filtered = filtered[
            filtered["VENDOR NAME"].str.contains(vendor_query, case=False, na=False)
        ]

    amount_column = filters.get("amount_column")
    amount_range = filters.get("amount_range")
    if amount_column and amount_range and amount_column in filtered.columns:
        min_amount, max_amount = amount_range
        filtered = filtered[
            (filtered[amount_column] >= float(min_amount))
            & (filtered[amount_column] <= float(max_amount))
        ]

    if data_type == "Detail" and "TRANSACTION DATE" in filtered.columns:
        date_range = filters.get("transaction_dates")
        if date_range and len(date_range) == 2:
            start_date, end_date = date_range
            start_ts = pd.to_datetime(start_date)
            end_ts = pd.to_datetime(end_date)
            filtered = filtered[
                (filtered["TRANSACTION DATE"] >= start_ts)
                & (filtered["TRANSACTION DATE"] <= end_ts)
            ]

    return filtered


def render_detail_view(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No detail records match the current filters.")
        return

    total_amount = df["AMOUNT"].sum()
    median_amount = df["AMOUNT"].median()
    unique_vendors = df.get("VENDOR NAME", pd.Series(dtype=str)).nunique(dropna=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total spend (filtered)", f"${total_amount:,.0f}")
    c2.metric("Median transaction", f"${median_amount:,.0f}")
    c3.metric("Unique vendors", f"{unique_vendors:,}")

    st.subheader("Top organizations")
    org_totals = (
        df.groupby("ORGANIZATION")["AMOUNT"].sum().sort_values(ascending=False).head(10)
    )
    st.bar_chart(org_totals)

    if "VENDOR NAME" in df.columns:
        st.subheader("Top vendors")
        vendor_totals = (
            df.groupby("VENDOR NAME")["AMOUNT"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .rename("Total Amount")
        )
        st.dataframe(vendor_totals)


def render_summary_view(df: pd.DataFrame, amount_column: Optional[str]) -> None:
    if df.empty:
        st.info("No summary rows match the current filters.")
        return

    amount_column = amount_column or "QTD AMOUNT"
    if amount_column not in df.columns:
        amount_column = df.columns[-1]

    total_value = df[amount_column].sum()
    avg_value = df[amount_column].mean()
    unique_programs = df.get("PROGRAM", pd.Series(dtype=str)).nunique(dropna=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Aggregate value", f"${total_value:,.0f}")
    c2.metric("Mean row value", f"${avg_value:,.0f}")
    c3.metric("Programs represented", f"{unique_programs:,}")

    st.subheader("Top descriptions")
    top_desc = (
        df.groupby("DESCRIPTION")[amount_column]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .rename("Total")
    )
    st.dataframe(top_desc)

    st.subheader("Organizations by value")
    org_chart = (
        df.groupby("ORGANIZATION")[amount_column]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    st.bar_chart(org_chart)


def render_data_preview(
    df: pd.DataFrame, data_type: str, selected_paths: Tuple[str, ...], filters: Dict[str, object]
) -> None:
    st.subheader("Filtered rows")
    st.caption(f"{len(df):,} rows displayed (showing first 1,000).")
    st.dataframe(df.head(1000))

    if data_type == "Detail":
        st.caption("Downloading detail rows uses the streaming pipeline to avoid memory spikes.")
        if st.button("Prepare filtered CSV", key="detail_preview_prepare"):
            detail_filters = dict(filters)
            detail_filters["amount_column"] = "AMOUNT"
            try:
                csv_bytes = stream_filtered_detail_csv(list(selected_paths), detail_filters)
            except ValueError as exc:
                st.info(str(exc))
                return

            st.download_button(
                "Download filtered CSV",
                data=csv_bytes,
                file_name="sod_detail_filtered.csv",
                mime="text/csv",
                key="detail_preview_download",
            )
    else:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered CSV",
            data=csv_bytes,
            file_name="sod_filtered.csv",
            mime="text/csv",
        )


def stream_filtered_detail_csv(
    csv_paths: List[str], filters: Dict[str, object], chunk_size: int = 100_000
) -> bytes:
    """Read detail CSVs in chunks, apply filters, and return the concatenated CSV bytes."""

    output_buffer = io.StringIO()
    header_written = False
    total_rows = 0

    for path_str in csv_paths:
        if not _path_exists(path_str):
            continue

        for chunk in _iter_csv_chunks(path_str, chunk_size):
            chunk.columns = [col.strip().upper() for col in chunk.columns]
            chunk["SOURCE_FILE"] = _extract_filename(path_str)

            if "PERSON" not in chunk.columns and "ORGANIZATION" in chunk.columns:
                chunk["PERSON"] = chunk["ORGANIZATION"].apply(extract_person_name)

            if "AMOUNT" in chunk.columns:
                chunk["AMOUNT"] = _to_numeric(chunk["AMOUNT"])
            if "TRANSACTION DATE" in chunk.columns:
                chunk["TRANSACTION DATE"] = _parse_dates(chunk["TRANSACTION DATE"])

            subset = apply_filters(chunk, filters, "Detail")
            if subset.empty:
                continue

            subset.to_csv(
                output_buffer,
                index=False,
                header=not header_written,
                mode="a",
            )
            header_written = True
            total_rows += len(subset)

    if total_rows == 0:
        raise ValueError("No detail rows match the current filters.")

    return output_buffer.getvalue().encode("utf-8")


def _iter_csv_chunks(csv_path: str, chunk_size: int) -> Iterable[pd.DataFrame]:
    """Yield DataFrame chunks using the same encoding fallback as bulk loading."""

    encodings = ("utf-8", "cp1252", "latin-1")
    last_error: Optional[Exception] = None
    for encoding in encodings:
        try:
            chunk_iter = pd.read_csv(
                csv_path,
                encoding=encoding,
                encoding_errors="ignore",
                low_memory=False,
                chunksize=chunk_size,
            )
            for chunk in chunk_iter:
                yield chunk
            return
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    raise last_error or UnicodeDecodeError(
        "utf-8", b"", 0, 1, f"Failed to decode {_extract_filename(csv_path)} with fallback encodings."
    )


def render_detail_download_sidebar(
    all_files: List[FileMeta], selected_years: List[Optional[int]], filters: Dict[str, object]
) -> None:
    expander = st.sidebar.expander("Detailed summary download", expanded=False)
    with expander:
        if not selected_years:
            st.info("Select at least one reporting year to enable detailed downloads.")
            return

        detail_files = [
            meta for meta in all_files if meta.data_type == "Detail" and meta.year in selected_years
        ]
        if not detail_files:
            st.info("No detail files available for the selected years.")
            return

        if not st.button("Prepare detailed summary CSV", key="detail_download_prepare"):
            st.caption("Click to generate a CSV of detail records matching your filters.")
            return

        detail_paths = [str(meta.path) for meta in detail_files]
        detail_filters = dict(filters)
        detail_filters["amount_column"] = "AMOUNT"

        try:
            csv_bytes = stream_filtered_detail_csv(detail_paths, detail_filters)
        except ValueError as exc:
            st.info(str(exc))
            return

        st.download_button(
            "Download detailed summary",
            data=csv_bytes,
            file_name="sod_detail_filtered.csv",
            mime="text/csv",
            key="detail_download_link",
        )


def ensure_data_directory(path_str: str) -> Optional[Path]:
    """Ensure a directory with CSVs exists, extracting from a zip if necessary."""

    path = Path(path_str).expanduser()
    if path.is_dir():
        return path

    candidates = []
    if path.suffix.lower() == ".zip" and path.is_file():
        candidates.append((path, path.parent / path.stem))
    else:
        zip_candidate = path.with_suffix(".zip")
        if zip_candidate.is_file():
            candidates.append((zip_candidate, path))

    for zip_path, dest_dir in candidates:
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            with ZipFile(zip_path) as zf:
                zf.extractall(dest_dir)
            st.sidebar.success(f"Extracted {zip_path.name} into {dest_dir}")
            return dest_dir
        except Exception as exc:  # pragma: no cover - surface to the UI
            st.sidebar.error(f"Failed to extract {zip_path.name}: {exc}")
            return None

    return path if path.is_dir() else None


def main() -> None:
    st.set_page_config(page_title="SOD Explorer", layout="wide")
    st.title("Statement of Disbursements Explorer")
    st.caption("Interactively analyze SOD summary/detail CSV files with custom filters.")

    data_source = st.sidebar.radio("Data source", options=["Local folder", "GCS bucket"], index=0)
    files: List[FileMeta] = []

    if data_source == "Local folder":
        data_dir_input = st.sidebar.text_input(
            "SOD data folder",
            value=str(DEFAULT_DATA_DIR),
            help="Path containing the raw CSV files.",
        )
        data_dir = ensure_data_directory(data_dir_input)
        if not data_dir:
            st.error("Unable to locate or extract the data directory. Check the path or zip file.")
            st.stop()
        files = discover_local_files(str(data_dir))
    else:
        profile = get_preloaded_gcs_profile()
        gcs_bucket = (profile.get("bucket") or "").strip()
        gcs_prefix = (profile.get("prefix") or "").strip()
        service_account_json = resolve_service_account_json(profile.get("service_account_json"))

        if not gcs_bucket:
            st.error(
                "No preloaded GCS bucket configured. Update DEFAULT_GCS_PROFILE or add an entry "
                "under [gcs_sources] in Streamlit secrets."
            )
            st.stop()

        prefix_label = gcs_prefix or "(entire bucket)"
        st.sidebar.success(f"Using GCS bucket: {gcs_bucket}\nPrefix: {prefix_label}")
        if service_account_json:
            st.sidebar.caption("Service account credentials loaded automatically.")
        else:
            st.sidebar.warning("No service account JSON detected; relying on default credentials.")

        available_years, use_year_dirs = discover_gcs_years(
            gcs_bucket, gcs_prefix, service_account_json
        )
        selected_year_filters: Tuple[int, ...] = ()
        if available_years:
            default_year = max(available_years)
            selected_year_filters = tuple(
                st.sidebar.multiselect(
                    "GCS reporting years",
                    options=available_years,
                    default=[default_year],
                    format_func=lambda year: str(year),
                    help="Only the chosen years will be scanned within the bucket.",
                )
            )
            if not selected_year_filters:
                st.warning("Select at least one year to begin exploring the GCS data.")
                st.stop()
        else:
            manual_text = st.sidebar.text_input(
                "GCS reporting years",
                placeholder="e.g. 2023, 2024",
                help="Enter one or more four-digit years separated by commas or spaces.",
            )
            manual_years = parse_year_tokens(manual_text)
            if manual_years:
                selected_year_filters = tuple(manual_years)
            else:
                st.warning(
                    "Unable to auto-detect year folders. Provide at least one year above "
                    "or configure a narrower prefix."
                )
                st.stop()

        files = discover_gcs_files(
            gcs_bucket,
            gcs_prefix,
            service_account_json,
            years=selected_year_filters,
            use_year_dirs=use_year_dirs,
        )

    if not files:
        if data_source == "Local folder":
            st.error("No CSV files detected. Confirm the folder path and retry.")
        else:
            st.error("No CSV files detected in the specified bucket/prefix.")
        st.stop()

    data_type, selected_files, selected_years = sidebar_dataset_picker(files)
    if not selected_files:
        st.warning("Pick at least one reporting period to begin exploring the data.")
        st.stop()

    selected_paths = tuple(str(meta.path) for meta in selected_files)
    with st.spinner("Loading CSV files…"):
        df = load_data(selected_paths, data_type)

    if df.empty:
        st.error("The selected files could not be loaded or contain no rows.")
        st.stop()

    filters = build_filter_controls(df, data_type)
    filtered_df = apply_filters(df, filters, data_type)

    amount_column = filters.get("amount_column")
    if data_type == "Detail":
        render_detail_view(filtered_df)
    else:
        render_summary_view(filtered_df, amount_column)

    render_data_preview(filtered_df, data_type, selected_paths, filters)
    render_detail_download_sidebar(files, selected_years, filters)


if __name__ == "__main__":
    main()
