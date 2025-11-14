import logging
import os
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
        proxies=None,
        refresh: bool = False,
        valid_rules: Optional[Dict[str, Any]] = None,
        session: Optional[requests.Session] = None,
        timeout: float = 10.0,
        backoff_factor: float = 0.1,
        max_retries: int = 0,
) -> Any:
    try:
        session = session or cloudscraper.session()
        file_path = Path(save_dir) / filename if save_dir and filename else None

        # Determine if this is likely a PDF based on filename
        is_pdf = filename and filename.lower().endswith('.pdf') if filename else False

        # Load from cache
        if file_path and file_path.exists() and not refresh:
            logger.info(f"Loading cached response for {filename} from {file_path.resolve().as_uri()}")
            try:
                if is_pdf:
                    # Read binary content for PDF
                    content = file_path.read_bytes()
                    content_type = "application/pdf"
                else:
                    # Read text content for HTML/text
                    content = file_path.read_text(encoding="utf-8")
                    content_type = "text/html; charset=utf-8"

                    if valid_rules:
                        response_text, errors = validate_text(content, valid_rules)
                        if not response_text and errors:
                            e = '\n'.join(errors)
                            if any(word in e for word in valid_rules['forbidden']):
                                os.remove(file_path)
                                print('🗑️ Forbbiden repsponse deleting')
                            return Exception(e)
                        content = response_text

                response = RequestsResponse()
                response.status_code = 200
                response._content = content if isinstance(content, bytes) else content.encode("utf-8")
                response.url = url
                response.reason = "OK"
                response.headers["Content-Type"] = content_type
                response.headers[LOCAL_FILE_HEADER] = f"file://{file_path.resolve().as_posix()}"
                response.cookies = RequestsCookieJar()
                return response
            except Exception as e:
                return ValueError(f"Error reading cache file {file_path.resolve().as_uri()}: {e}")

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
            logger.info(f"Requesting for {filename} [{method}] : {url}")
            response = session.request(
                method=method.upper(),
                url=url,
                params=params,
                headers=headers,
                data=data,
                json=json,
                cookies=cookies,
                timeout=timeout,
                proxies=proxies
            )
            response.raise_for_status()

            # Check if response is PDF based on Content-Type header
            content_type = response.headers.get('Content-Type', '').lower()
            is_pdf_response = 'application/pdf' in content_type or is_pdf

            if is_pdf_response:
                # Cache PDF binary content
                if file_path:
                    try:
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        file_path.write_bytes(response.content)
                        response.headers[LOCAL_FILE_HEADER] = f"{file_path.resolve().as_uri()}"
                        logger.info(f"Saved PDF response for {filename} to {file_path.resolve().as_uri()}")
                    except IOError as e:
                        logger.error(f"Failed to save response to {file_path}: {e}")
                return response
            else:
                # Handle text content with validation
                if valid_rules:
                    response_text, errors = validate_text(response.text, valid_rules)
                    if not response_text and errors:
                        e = '\n'.join(errors)
                        return ValueError(e)
                    # Update response content with validated text
                    response._content = response_text.encode("utf-8")

                # Cache text response
                if file_path:
                    try:
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        file_path.write_text(response.text, encoding="utf-8")
                        response.headers[LOCAL_FILE_HEADER] = f"{file_path.resolve().as_uri()}"
                        logger.info(f"Saved response for {filename} to {file_path.resolve().as_uri()}")
                    except IOError as e:
                        logger.error(f"Failed to save response to {file_path}: {e}")
                return response

        except requests.exceptions.HTTPError as e:
            return e
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.RequestException) as e:
            return e
    except Exception as outer_e:
        return outer_e


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
