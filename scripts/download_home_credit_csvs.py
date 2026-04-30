"""
Download Home Credit Default Risk CSVs into data/raw/ using the Kaggle CLI.

Prerequisites:
  1. Join the competition and accept the rules:
     https://www.kaggle.com/c/home-credit-default-risk
  2. Authenticate (new Kaggle API tokens look like KGAT_...):
       - Recommended: copy config/access_token.example to config/access_token
         and paste your token as a single line (no quotes). Gitignored.
       - Or set env KAGGLE_API_TOKEN to the token string, or to a file path.
       - Or put the token in ~/.kaggle/access_token (one line).
     Legacy (username + secret key, not KGAT_):
       - config/kaggle.json (see kaggle.json.example), or ~/.kaggle/kaggle.json,
         or KAGGLE_USERNAME + KAGGLE_KEY.
     Optional: KAGGLE_CONFIG_DIR = folder that contains kaggle.json (legacy only).
  3. pip install kaggle

Usage (from project root):
    python scripts/download_home_credit_csvs.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
COMPETITION = "home-credit-default-risk"

PROJECT_KAGGLE_DIR = PROJECT_ROOT / "config"
PROJECT_KAGGLE_JSON = PROJECT_KAGGLE_DIR / "kaggle.json"
PROJECT_ACCESS_TOKEN = PROJECT_KAGGLE_DIR / "access_token"
PROJECT_ACCESS_TOKEN_TXT = PROJECT_KAGGLE_DIR / "access_token.txt"
HOME_KAGGLE_DIR = Path.home() / ".kaggle"
HOME_KAGGLE_JSON = HOME_KAGGLE_DIR / "kaggle.json"
HOME_ACCESS_TOKEN = HOME_KAGGLE_DIR / "access_token"
HOME_ACCESS_TOKEN_TXT = HOME_KAGGLE_DIR / "access_token.txt"


def _env_key_auth() -> bool:
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))


def _explicit_config_json() -> Path | None:
    d = os.environ.get("KAGGLE_CONFIG_DIR", "").strip()
    if not d:
        return None
    p = Path(d).expanduser() / "kaggle.json"
    return p if p.is_file() else None


def _access_token_text_ok(value: str) -> bool:
    v = value.strip()
    if len(v) < 15:
        return False
    upper = v.upper()
    if upper.startswith("PASTE_") or "PLACEHOLDER" in upper:
        return False
    return True


def _access_token_file_ok(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return _access_token_text_ok(path.read_text(encoding="utf-8-sig"))
    except OSError:
        return False


def _env_kaggle_api_token_literal() -> bool:
    t = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not t:
        return False
    if Path(t).expanduser().exists():
        return False
    return _access_token_text_ok(t)


def _env_kaggle_api_token_path_ok() -> bool:
    t = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not t:
        return False
    p = Path(t).expanduser()
    if not p.is_file():
        return False
    return _access_token_file_ok(p)


def _first_access_token_file() -> Path | None:
    for p in (
        PROJECT_ACCESS_TOKEN,
        PROJECT_ACCESS_TOKEN_TXT,
        HOME_ACCESS_TOKEN,
        HOME_ACCESS_TOKEN_TXT,
    ):
        if _access_token_file_ok(p):
            return p
    return None


def _credentials_ok() -> bool:
    if _env_key_auth():
        return True
    if _env_kaggle_api_token_literal() or _env_kaggle_api_token_path_ok():
        return True
    if _first_access_token_file() is not None:
        return True
    if _explicit_config_json() is not None:
        return True
    if PROJECT_KAGGLE_JSON.is_file():
        return True
    return HOME_KAGGLE_JSON.is_file()


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    if not env.get("KAGGLE_API_TOKEN", "").strip():
        at = _first_access_token_file()
        if at is not None:
            env["KAGGLE_API_TOKEN"] = str(at.resolve())
    if _env_key_auth():
        return env
    if os.environ.get("KAGGLE_CONFIG_DIR", "").strip():
        return env
    if PROJECT_KAGGLE_JSON.is_file():
        env["KAGGLE_CONFIG_DIR"] = str(PROJECT_KAGGLE_DIR.resolve())
    return env


def _print_credentials_help() -> None:
    print(
        "Kaggle API credentials are missing.\n\n"
        "New API tokens (Generate New Token on Kaggle) look like KGAT_...\n"
        "They are NOT pasted into kaggle.json. Use one of these:\n\n"
        "  1) Project file (recommended)\n"
        f"      Copy:  config/access_token.example  ->  config/access_token\n"
        f"      Edit:  {PROJECT_ACCESS_TOKEN}\n"
        "      Put the entire token on one line, no quotes, save.\n\n"
        "  2) Environment variable\n"
        "      set KAGGLE_API_TOKEN=KGAT_...   (PowerShell: $env:KAGGLE_API_TOKEN='KGAT_...')\n\n"
        "  3) User folder\n"
        f"      Save token one line in: {HOME_ACCESS_TOKEN}\n\n"
        "Legacy username + key (Create Legacy API Key / old kaggle.json):\n"
        f"      {PROJECT_KAGGLE_JSON}  or  {HOME_KAGGLE_JSON}\n\n"
        "Then accept the competition rules at:\n"
        f"  https://www.kaggle.com/c/{COMPETITION}\n\n"
        "Docs: https://www.kaggle.com/docs/api\n",
        file=sys.stderr,
    )


def _kaggle_exe() -> str | None:
    return shutil.which("kaggle")


def _using_project_token_file() -> bool:
    if _env_key_auth():
        return False
    if (
        _env_kaggle_api_token_literal()
        or _env_kaggle_api_token_path_ok()
        or _first_access_token_file() is not None
    ):
        return False
    explicit = os.environ.get("KAGGLE_CONFIG_DIR", "").strip()
    if explicit:
        try:
            return Path(explicit).expanduser().resolve() == PROJECT_KAGGLE_DIR.resolve()
        except OSError:
            return False
    return PROJECT_KAGGLE_JSON.is_file()


def _validate_project_kaggle_json() -> str | None:
    if not PROJECT_KAGGLE_JSON.is_file():
        return None
    try:
        raw = PROJECT_KAGGLE_JSON.read_text(encoding="utf-8-sig")
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return f"config/kaggle.json is not valid JSON ({e}). Fix or delete the file."
    u = str(data.get("username", "")).strip()
    k = str(data.get("key", "")).strip()
    if not u or not k:
        return "config/kaggle.json must contain non-empty 'username' and 'key'."
    if k.startswith("KGAT_"):
        return (
            "config/kaggle.json 'key' looks like a new Kaggle API token (KGAT_...). "
            "Those do not go in kaggle.json. Copy config/access_token.example to "
            "config/access_token, paste the token as a single line, save, and remove "
            "or fix kaggle.json."
        )
    if u in ("your_kaggle_username", "YOUR_KAGGLE_USERNAME"):
        return "config/kaggle.json still has the example username; paste your real Kaggle username."
    if k in ("your_kaggle_api_key", "YOUR_KAGGLE_API_KEY"):
        return (
            "config/kaggle.json still has the example key. For a new KGAT token use "
            "config/access_token instead; for legacy keys use the long key from Kaggle settings."
        )
    return None


def _print_401_help() -> None:
    print(
        "\n--- Kaggle returned 401 Unauthorized ---\n"
        "Common fixes:\n"
        "  1. Accept the competition rules:\n"
        f"     https://www.kaggle.com/c/{COMPETITION}\n"
        "  2. If you use a KGAT_ token: put it in config/access_token (one line) or\n"
        "     KAGGLE_API_TOKEN — not in kaggle.json 'key'.\n"
        "  3. Regenerate token: https://www.kaggle.com/settings/api\n"
        "  4. Legacy kaggle.json: username = profile slug (kaggle.com/<slug>); key = legacy secret.\n",
        file=sys.stderr,
    )


def _extract_zips(folder: Path, remove_after: bool) -> None:
    for zp in sorted(folder.glob("*.zip")):
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(folder)
        if remove_after:
            zp.unlink()


def main() -> int:
    exe = _kaggle_exe()
    if not exe:
        print(
            "The `kaggle` command was not found. Install with: pip install kaggle",
            file=sys.stderr,
        )
        return 1

    if not _credentials_ok():
        PROJECT_KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
        HOME_KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
        _print_credentials_help()
        return 1

    if _using_project_token_file():
        err = _validate_project_kaggle_json()
        if err:
            print(err, file=sys.stderr)
            return 1

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    cmd = [
        exe,
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-p",
        str(DATA_RAW),
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=_subprocess_env(),
        text=True,
        capture_output=True,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        combined = (proc.stdout or "") + (proc.stderr or "")
        if "401" in combined:
            _print_401_help()
        return proc.returncode

    _extract_zips(DATA_RAW, remove_after=True)
    print("Done. CSV files should now be in data/raw/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
