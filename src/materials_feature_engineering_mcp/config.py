MCP_PORT = 8180
# BASE_URL = "http://localhost:8180"
BASE_URL = "https://www.matterai.cn/fe"
from pathlib import Path

from .utils import _relative_path_within_data_dir

DATA_ROOT = Path("data")
DOWNLOAD_PREFIX = "/download/file"
STATIC_PREFIX = "/static"


def _relative_url_path(path: str) -> str:
    return _relative_path_within_data_dir(path).as_posix()


def get_download_url(path: str):
    return f"{BASE_URL}{DOWNLOAD_PREFIX}/{_relative_url_path(path)}"


def get_static_url(path: str):
    return f"{BASE_URL}{STATIC_PREFIX}/{_relative_url_path(path)}"
