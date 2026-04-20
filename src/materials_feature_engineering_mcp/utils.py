"""
Utility functions for Materials Feature Engineering MCP Tool
Common helper functions for data loading, validation, etc.
"""

import os
import re
import uuid
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional

import pandas as pd


DATA_DIR = Path("data")
_SAFE_USER_ID_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


def _is_url(path: str) -> bool:
    """Determine whether the given path is a URL."""
    try:
        parsed = urlparse(path)
        return parsed.scheme in ("http", "https", "ftp", "s3", "gs", "file")
    except Exception:
        return False


def _validate_data_path(data_path: str) -> None:
    """Perform existence validation for local paths, skip local validation for URLs."""
    if _is_url(data_path):
        return
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file does not exist: {data_path}")


def _read_tabular_data(data_path: str) -> pd.DataFrame:
    """Read CSV or Excel data from a local path or URL into a DataFrame."""
    local_path = _load_data_safe(data_path)
    parsed_path = urlparse(local_path).path.lower()

    if parsed_path.endswith(".csv"):
        return pd.read_csv(local_path)
    if parsed_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(local_path)

    try:
        return pd.read_csv(local_path)
    except Exception:
        return pd.read_excel(local_path)


def _sanitize_user_id(user_id: Optional[str]) -> str:
    """Normalize user identifiers so they are safe for filesystem paths."""
    if not user_id or not user_id.strip():
        return "default"

    cleaned = _SAFE_USER_ID_PATTERN.sub("_", user_id.strip()).strip("._-")
    return cleaned or "default"


def _resolve_path_within_data_dir(path: str | Path) -> Path:
    """Resolve a path and ensure it stays inside the managed data directory."""
    data_root = DATA_DIR.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if candidate != data_root and data_root not in candidate.parents:
        raise ValueError(f"Path escapes data directory: {candidate}")

    return candidate


def _relative_path_within_data_dir(path: str | Path) -> Path:
    """Return a relative path rooted at the managed data directory."""
    return _resolve_path_within_data_dir(path).relative_to(DATA_DIR.resolve())


def _validate_target_dims(total_columns: int, target_dims: int) -> None:
    """Validate that at least one feature column remains after splitting targets."""
    if target_dims <= 0:
        raise ValueError("target_dims must be a positive integer")
    if total_columns <= 1:
        raise ValueError("Dataset must contain at least one feature column and one target column")
    if target_dims >= total_columns:
        raise ValueError(
            f"target_dims must be between 1 and {total_columns - 1}, current value: {target_dims}"
        )


def _download_file_from_url(url: str, max_retries: int = 3, timeout: int = 30) -> str:
    """
    Download file from URL to temporary directory with retry mechanism.

    Args:
        url: URL to download from
        max_retries: Maximum number of retry attempts (default: 3)
        timeout: Timeout in seconds for each attempt (default: 30)

    Returns:
        Path to the downloaded temporary file

    Raises:
        ValueError: If download fails after all retries
    """
    import urllib.request
    import urllib.error

    # Extract filename from URL
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename:
        filename = "downloaded_data.csv"

    # Create temporary file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"mcp_download_{uuid.uuid4()}_{filename}")

    print(f"📥 Downloading from URL: {url}")
    print(f"   Temporary location: {temp_path}")

    for attempt in range(1, max_retries + 1):
        try:
            print(f"   Attempt {attempt}/{max_retries}...")

            # Create request with headers
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

            # Download file
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content = response.read()

            # Save to temporary file
            with open(temp_path, 'wb') as f:
                f.write(content)

            file_size = len(content) / (1024 * 1024)  # MB
            print(f"✓ Download successful! Size: {file_size:.2f} MB")
            return temp_path

        except urllib.error.URLError as e:
            print(f"✗ Attempt {attempt} failed: URL error - {str(e)}")
            if attempt == max_retries:
                raise ValueError(f"Failed to download from URL after {max_retries} attempts: {str(e)}. Please check if the URL is accessible.")

        except Exception as e:
            error_msg = str(e).lower()
            if "timed out" in error_msg or "timeout" in error_msg:
                print(f"✗ Attempt {attempt} failed: Timeout ({timeout}s)")
            else:
                print(f"✗ Attempt {attempt} failed: {str(e)}")

            if attempt == max_retries:
                raise ValueError(f"Failed to download from URL after {max_retries} attempts: {str(e)}. Please check your network connection or try a local file.")

        # Wait before retry (exponential backoff)
        if attempt < max_retries:
            wait_time = 2 ** attempt  # 2, 4, 8 seconds
            print(f"   Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    raise ValueError(f"Failed to download from URL after {max_retries} attempts")


def _try_convert_url_to_local_path(url: str) -> Optional[str]:
    """
    Try to convert URL to local file path if the file exists locally.

    Supports patterns like:
    - http://localhost:8180/download/file/10004/uuid/filename.csv
    - http://domain.com/download/file/user_id/uuid/filename.csv

    Args:
        url: URL to check

    Returns:
        Local file path if exists, None otherwise
    """
    try:
        # Parse URL to extract path components
        from urllib.parse import urlparse, unquote
        parsed = urlparse(url)

        path_parts = [unquote(part) for part in parsed.path.strip("/").split("/") if part]
        relative_parts: list[str] = []

        if len(path_parts) >= 3 and path_parts[0] == "download" and path_parts[1] == "file":
            relative_parts = path_parts[2:]
        elif len(path_parts) >= 2 and path_parts[0] == "static":
            if path_parts[1] in {"models", "model", "reports", "file"}:
                relative_parts = path_parts[2:]
            else:
                relative_parts = path_parts[1:]

        if relative_parts:
            local_path = DATA_DIR.joinpath(*relative_parts)
            try:
                resolved_path = _resolve_path_within_data_dir(local_path)
            except ValueError:
                return None

            if resolved_path.exists():
                print(f"✓ Found local file: {resolved_path}")
                print("  Skipping HTTP download from URL")
                return str(resolved_path)

    except Exception:
        # If parsing fails, just return None and fall back to HTTP download
        pass

    return None


def _load_data_safe(data_path: str, max_retries: int = 3, timeout: int = 60) -> str:
    """
    Safely load data from either local path or URL.

    Args:
        data_path: Local file path or URL
        max_retries: Maximum number of retry attempts for URL downloads (default: 3)
        timeout: Timeout in seconds for each download attempt (default: 60)

    Returns:
        Local file path (original or downloaded temporary file)
    """
    if _is_url(data_path):
        # First, try to convert URL to local path if file exists locally
        local_path = _try_convert_url_to_local_path(data_path)
        if local_path:
            return local_path

        # If not found locally, download from URL
        # For localhost URLs, use longer timeout as they might be from local server
        if 'localhost' in data_path or '127.0.0.1' in data_path:
            timeout = max(timeout, 30)  # At least 2 minutes for localhost
            print(f"💡 Detected localhost URL, using extended timeout: {timeout}s")

        return _download_file_from_url(data_path, max_retries=max_retries, timeout=timeout)
    else:
        return data_path
