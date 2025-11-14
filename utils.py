import logging
from pathlib import Path
from typing import Optional, Dict, Any

import cloudscraper
import requests
from requests.adapters import HTTPAdapter
from requests.cookies import RequestsCookieJar
from requests.models import Response as RequestsResponse
from urllib3.util.retry import Retry

# --------------------------------
# Logging configuration
# --------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler("scraper.log"), logging.StreamHandler()],
)
logger = logging.getLogger("fetcher")

LOCAL_FILE_HEADER = "X-Local-File"


def fetch(
        url: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        cookies: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = None,
        filename: Optional[str] = None,
        refresh: bool = False,
        valid_rules: Optional[Dict[str, Any]] = None,
        session: Optional[requests.Session] = None,
        timeout: float = 10.0,
        backoff_factor: float = 0.1,
        max_retries: int = 0,
) -> Optional[RequestsResponse]:
    session = session or cloudscraper.create_scraper()
    file_path = Path(save_dir) / filename if save_dir and filename else None

    # Load from cache
    if file_path and file_path.exists() and not refresh:
        logger.info(f"Loading cached response for {url} from {file_path.resolve().as_uri()}")
        try:
            content = file_path.read_text(encoding="utf-8")
            response = RequestsResponse()
            response.status_code = 200
            response._content = content.encode("utf-8")
            response.url = url
            response.reason = "OK"
            response.headers["Content-Type"] = "text/html; charset=utf-8"
            response.headers[LOCAL_FILE_HEADER] = f"file://{file_path.resolve().as_posix()}"
            response.cookies = RequestsCookieJar()
            return response
        except Exception as e:
            logger.error(f"Error reading cache file {file_path.resolve().as_uri()}: {e}")

    # Configure retries
    if max_retries > 0:
        retries = Retry(
            total=max_retries,
            read=max_retries,
            connect=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(500, 502, 503, 504, 403, 429),
            allowed_methods={"HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"},
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

    # Perform request
    try:
        logger.info(f"Requesting {method} {url}")
        response = session.request(
            method=method.upper(),
            url=url,
            params=params,
            headers=headers,
            data=data,
            json=json,
            cookies=cookies,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error for {url}: {e.response.status_code} - {e.response.reason}")
        return None
    except (requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException) as e:
        logger.error(f"Request failed for {url}: {e}")
        if isinstance(e, requests.exceptions.RequestException):
            raise
        return None

    # Cache response
    if file_path:
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(response.text, encoding="utf-8")
            response.headers[LOCAL_FILE_HEADER] = f"{file_path.resolve().as_uri()}"
            logger.info(f"Saved response for {url} to {file_path.resolve().as_uri()}")

        except IOError as e:
            logger.error(f"Failed to save response to {file_path}: {e}")

    return response


import re
import time
from typing import Optional, List, Dict, Any, Tuple


def validate_text(
        text: str,
        rules: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Validates text based on simple rule categories:

    Supported rule keys:
        - 'required': list of required substrings
        - 'forbidden': list of substrings not allowed
        - 'regex': list of regex patterns text must match
        - 'expiration': Unix timestamp; text considered invalid if now > expiration
        - 'must_start_with': string prefix that text must start with
        - 'must_end_with': string suffix that text must end with
        - 'case_sensitive': boolean (default: True)

    Returns:
        (text, None) if valid
        (None, [errors]) if invalid
    """

    errors = []
    rules = rules or {}

    # --- BASIC: must be a non-empty string ---
    if not isinstance(text, str):
        return None, [f"Invalid input type: {type(text).__name__}. Expected string"]

    if not text:
        return None, ["Text is empty"]

    # --- Case handling ---
    case_sensitive = rules.get("case_sensitive", True)
    compare_text = text if case_sensitive else text.lower()

    # --- Required keywords ---
    if "required" in rules:
        for item in rules["required"]:
            req = item if case_sensitive else item.lower()
            if req not in compare_text:
                errors.append(f"Required keyword missing: '{item}'")

    # --- Forbidden keywords ---
    if "forbidden" in rules:
        for item in rules["forbidden"]:
            forb = item if case_sensitive else item.lower()
            if forb in compare_text:
                errors.append(f"Forbidden keyword found: '{item}'")

    # --- Regex validation ---
    if "regex" in rules:
        for pattern in rules["regex"]:
            try:
                if not re.search(pattern, text):
                    errors.append(f"Regex not satisfied: {pattern}")
            except re.error as e:
                errors.append(f"Invalid regex '{pattern}': {e}")

    # --- Expiration check ---
    if "expiration" in rules:
        now = time.time()
        if now > rules["expiration"]:
            errors.append("Text has expired based on expiration timestamp")

    # --- Developer-suggested useful rules ---
    if "must_start_with" in rules:
        if not text.startswith(rules["must_start_with"]):
            errors.append(
                f"Text must start with: '{rules['must_start_with']}'"
            )

    if "must_end_with" in rules:
        if not text.endswith(rules["must_end_with"]):
            errors.append(
                f"Text must end with: '{rules['must_end_with']}'"
            )

    return (text, None) if not errors else (None, errors)
