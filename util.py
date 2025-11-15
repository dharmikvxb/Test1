"""
Professional utility module for web scraping and MongoDB operations.

Modules:
    - Record: MongoDB document wrapper with state tracking
    - fetch: HTTP client with caching and validation
    - validate_text: Response validation engine
"""
import logging
import os
import re
import time
from pathlib import Path
from typing import  List, Tuple

import cloudscraper
import requests
from requests.adapters import HTTPAdapter
from requests.cookies import RequestsCookieJar
from requests.models import Response as RequestsResponse
from urllib3.util.retry import Retry

# ==============================================================================
# Logging Configuration
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ],
)

logger = logging.getLogger(__name__)

# ==============================================================================
# Constants
# ==============================================================================

LOCAL_FILE_HEADER = "X-Local-File"


# ==============================================================================
# MongoDB Record Wrapper
# ==============================================================================

from typing import Optional, Dict, Any



# ==============================================================================
# Text Validation
# ==============================================================================

def validate_text(
    text: str,
    rules: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Validates text based on configurable rules.

    Supported rule keys:
        - 'required': list of required substrings
        - 'forbidden': list of substrings not allowed
        - 'regex': list of regex patterns text must match
        - 'expiration': Unix timestamp; text invalid if now > expiration
        - 'must_start_with': string prefix that text must start with
        - 'must_end_with': string suffix that text must end with
        - 'case_sensitive': boolean (default: True)

    Args:
        text: Text to validate
        rules: Dictionary of validation rules

    Returns:
        Tuple of (text, None) if valid, or (None, [errors]) if invalid
    """
    errors = []
    rules = rules or {}

    # Check input type
    if not isinstance(text, str):
        return None, [f"Invalid input type: {type(text).__name__}. Expected string"]

    if not text:
        return None, ["Text is empty"]

    # Case handling
    case_sensitive = rules.get("case_sensitive", True)
    compare_text = text if case_sensitive else text.lower()

    # Required keywords
    if "required" in rules:
        for item in rules["required"]:
            req = item if case_sensitive else item.lower()
            if req not in compare_text:
                errors.append(f"Required keyword missing: '{item}'")

    # Forbidden keywords
    if "forbidden" in rules:
        for item in rules["forbidden"]:
            forb = item if case_sensitive else item.lower()
            if forb in compare_text:
                errors.append(f"Forbidden keyword found: '{item}'")

    # Regex validation
    if "regex" in rules:
        for pattern in rules["regex"]:
            try:
                if not re.search(pattern, text):
                    errors.append(f"Regex not satisfied: {pattern}")
            except re.error as e:
                errors.append(f"Invalid regex '{pattern}': {e}")

    # Expiration check
    if "expiration" in rules:
        now = time.time()
        if now > rules["expiration"]:
            errors.append("Text has expired based on expiration timestamp")

    # Prefix check
    if "must_start_with" in rules:
        if not text.startswith(rules["must_start_with"]):
            errors.append(f"Text must start with: '{rules['must_start_with']}'")

    # Suffix check
    if "must_end_with" in rules:
        if not text.endswith(rules["must_end_with"]):
            errors.append(f"Text must end with: '{rules['must_end_with']}'")

    return (text, None) if not errors else (None, errors)


# ==============================================================================
# HTTP Fetch with Caching
# ==============================================================================

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
    proxies: Optional[Dict[str, str]] = None,
    refresh: bool = False,
    valid_rules: Optional[Dict[str, Any]] = None,
    session: Optional[requests.Session] = None,
    timeout: float = 10.0,
    backoff_factor: float = 0.1,
    max_retries: int = 0,
) -> Any:
    """
    Fetch HTTP resource with caching, validation, and retry logic.

    Args:
        url: Target URL
        method: HTTP method (GET, POST, etc.)
        params: URL parameters
        cookies: Request cookies
        headers: Request headers
        data: Form data for POST requests
        json: JSON data for POST requests
        save_dir: Directory to cache responses
        filename: Cache filename
        proxies: Proxy configuration
        refresh: Force refresh cache
        valid_rules: Text validation rules (see validate_text)
        session: Existing requests session
        timeout: Request timeout in seconds
        backoff_factor: Retry backoff factor
        max_retries: Maximum retry attempts

    Returns:
        Response object on success, Exception on error

    """
    try:
        session = session or cloudscraper.session()
        file_path = Path(save_dir) / filename if save_dir and filename else None

        # Determine content type from filename
        is_pdf = filename and filename.lower().endswith('.pdf') if filename else False

        # --------------------------------------------------------------------------
        # Load from cache
        # --------------------------------------------------------------------------
        if file_path and file_path.exists() and not refresh:
            logger.info(f"Loading cached response: {filename} from {file_path.resolve().as_uri()}")
            try:
                if is_pdf:
                    # Binary content (PDF)
                    content = file_path.read_bytes()
                    content_type = "application/pdf"
                else:
                    # Text content (HTML)
                    content = file_path.read_text(encoding="utf-8")
                    content_type = "text/html; charset=utf-8"

                    # Validate cached text
                    if valid_rules:
                        response_text, errors = validate_text(content, valid_rules)
                        if not response_text and errors:
                            error_msg = '\n'.join(errors)
                            # Delete cache if forbidden content found
                            if any(word in error_msg for word in valid_rules.get('forbidden', [])):
                                os.remove(file_path)
                                logger.warning('🗑️ Forbidden response deleted from cache')
                            return Exception(error_msg)
                        content = response_text

                # Build mock response
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

        # --------------------------------------------------------------------------
        # Configure retries
        # --------------------------------------------------------------------------
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

        # --------------------------------------------------------------------------
        # Perform HTTP request
        # --------------------------------------------------------------------------
        try:
            logger.info(f"Requesting: {filename} [{method}] {url}")
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

            # Check content type
            content_type = response.headers.get('Content-Type', '').lower()
            is_pdf_response = 'application/pdf' in content_type or is_pdf

            # Handle PDF response
            if is_pdf_response:
                if file_path:
                    try:
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        file_path.write_bytes(response.content)
                        response.headers[LOCAL_FILE_HEADER] = f"{file_path.resolve().as_uri()}"
                        logger.info(f"Saved PDF: {filename} to {file_path.resolve().as_uri()}")
                    except IOError as e:
                        logger.error(f"Failed to save PDF to {file_path}: {e}")
                return response

            # Handle text response with validation
            else:
                if valid_rules:
                    response_text, errors = validate_text(response.text, valid_rules)
                    if not response_text and errors:
                        error_msg = '\n'.join(errors)
                        return ValueError(error_msg)
                    # Update response content with validated text
                    response._content = response_text.encode("utf-8")

                # Cache text response
                if file_path:
                    try:
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        file_path.write_text(response.text, encoding="utf-8")
                        response.headers[LOCAL_FILE_HEADER] = f"{file_path.resolve().as_uri()}"
                        logger.info(f"Saved response: {filename} to {file_path.resolve().as_uri()}")
                    except IOError as e:
                        logger.error(f"Failed to save response to {file_path}: {e}")
                return response

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {url}: {e}")
            return e
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException
        ) as e:
            logger.error(f"Request error for {url}: {e}")
            return e

    except Exception as outer_e:
        logger.error(f"Unexpected error in fetch: {outer_e}")
        return outer_e


from copy import deepcopy
from datetime import datetime
from typing import Optional, Dict, Any


class Record:
    """
    Professional MongoDB document wrapper with state tracking.

    Features:
        - Dot notation access: rec.field_name
        - Dictionary methods: rec.get(), rec.update()
        - State management: rec.mark_done(), rec.mark_fail(), rec.mark_reset()
        - Timestamp tracking in format: YYYYMMDDHHMMSS
        - State format: T|20241115143022 or F|20241115143022 (clean, no metadata)
        - Metadata stored separately in meta dict: {state_name: {key: val}}
        - Thread-safe operations with proper error handling

    Structure:
        {
            "_id": ...,
            "stats": {
                "scraping": "T|20241115143022",
                "processing": "F|20241116120000"
            },
            "meta": {
                "scraping": {"url": "...", "items": 42},
                "processing": {"error": "timeout", "retry": 2}
            }
        }


    """

    __slots__ = ('_doc', '_col', '_id')

    def __init__(self, doc: Dict[str, Any], collection):
        """
        Initialize Record wrapper.

        Args:
            doc: MongoDB document (will be deep copied)
            collection: PyMongo collection reference
        """
        object.__setattr__(self, '_doc', deepcopy(doc))
        object.__setattr__(self, '_col', collection)
        object.__setattr__(self, '_id', doc.get('_id'))

        # Ensure stats and meta fields exist
        if 'stats' not in self._doc or not isinstance(self._doc['stats'], dict):
            self._doc['stats'] = {}
        if 'meta' not in self._doc or not isinstance(self._doc['meta'], dict):
            self._doc['meta'] = {}

    def __repr__(self) -> str:
        """Clean, informative representation."""
        doc_preview = {k: v for k, v in self._doc.items() if k not in ('stats', 'meta')}
        stats_count = len(self._doc.get('stats', {}))
        return f"<Record(_id={self._id}, fields={list(doc_preview.keys())}, stats={stats_count})>"

    def __str__(self) -> str:
        """User-friendly string representation."""
        return f"Record(id={self._id})"

    # --------------------------------------------------------------------------
    # Attribute Access
    # --------------------------------------------------------------------------

    def __getattr__(self, key: str) -> Any:
        """Enable dot notation: rec.field_name"""
        if key in self._doc:
            return self._doc[key]
        raise AttributeError(f"Record has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        """Enable dot notation assignment: rec.field_name = value"""
        if key in self.__slots__:
            object.__setattr__(self, key, value)
        else:
            self._doc[key] = value

    # --------------------------------------------------------------------------
    # Dictionary Methods
    # --------------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Get field value with optional default."""
        return self._doc.get(key, default)

    def keys(self):
        """Return document keys."""
        return self._doc.keys()

    def values(self):
        """Return document values."""
        return self._doc.values()

    def items(self):
        """Return document items."""
        return self._doc.items()

    def to_dict(self) -> Dict[str, Any]:
        """Return clean deep copy of document."""
        return deepcopy(self._doc)

    # --------------------------------------------------------------------------
    # Database Operations
    # --------------------------------------------------------------------------

    def update(self, **fields) -> bool:
        """
        Update multiple fields at once.

        Args:
            **fields: Key-value pairs to update

        Returns:
            bool: True if update was successful
        """
        if not fields:
            return False

        try:
            self._doc.update(fields)
            if self._id:
                result = self._col.update_one(
                    {'_id': self._id},
                    {'$set': fields}
                )
                return result.modified_count > 0
            return False
        except Exception as e:
            print(f"Failed to update record {self._id}: {e}")
            return False

    def reload(self) -> bool:
        """
        Reload document from database.

        Returns:
            bool: True if reload was successful
        """
        try:
            fresh_doc = self._col.find_one({'_id': self._id})
            if fresh_doc:
                object.__setattr__(self, '_doc', fresh_doc)
                return True
            return False
        except Exception as e:
            print(f"Failed to reload record {self._id}: {e}")
            return False

    # --------------------------------------------------------------------------
    # State Management - Internal Helpers
    # --------------------------------------------------------------------------

    @staticmethod
    def _timestamp() -> str:
        """Generate timestamp in YYYYMMDDHHMMSS format."""
        return datetime.now().strftime('%Y%m%d%H%M%S')

    @staticmethod
    def _parse_timestamp(ts_str: str) -> Optional[datetime]:
        """Parse timestamp string back to datetime."""
        try:
            return datetime.strptime(ts_str, '%Y%m%d%H%M%S')
        except (ValueError, TypeError):
            return None

    def _encode_state(self, success: bool) -> str:
        """
        Encode state with success flag and timestamp only.

        Format: "T|20241115143022" (success)
                "F|20241115143022" (failure)

        This clean format allows:
            - Easy regex queries: {'stats.scraping': {'$regex': '^T'}}
            - Simple timestamp extraction: substr(field, 2, 14)
            - Better indexing performance (smaller field size)
        """
        flag = 'T' if success else 'F'
        ts = self._timestamp()
        return f"{flag}|{ts}"

    def _decode_state(self, encoded: str, state_name: str) -> Optional[Dict[str, Any]]:
        """
        Decode state string and merge with metadata.

        Returns:
            {
                'success': bool,
                'ts': str,
                'timestamp': datetime,
                'meta': dict
            }
        """
        if not encoded:
            return None

        try:
            parts = encoded.split('|')
            if len(parts) < 2:
                return None

            flag, ts = parts[0], parts[1]
            result = {
                'success': flag == 'T',
                'ts': ts,
                'timestamp': self._parse_timestamp(ts),
                'meta': self._doc['meta'].get(state_name, {})
            }

            return result
        except Exception as e:
            print(f"Failed to decode state '{encoded}': {e}")
            return None

    # --------------------------------------------------------------------------
    # State Management - Public API
    # --------------------------------------------------------------------------

    def mark_done(self, state: str, **meta) -> bool:
        """
        Mark state as successfully completed.

        Args:
            state: State name (e.g., 'scraping', 'processing')
            **meta: Additional metadata stored separately (e.g., url='...', count=10)

        Returns:
            bool: True if mark was successful
        """
        encoded = self._encode_state(True)
        update_doc = {f'stats.{state}': encoded}

        # Store metadata separately if provided
        if meta:
            self._doc['meta'][state] = meta
            update_doc[f'meta.{state}'] = meta

        try:
            self._doc['stats'][state] = encoded
            result = self._col.update_one(
                {'_id': self._id},
                {'$set': update_doc}
            )

            if meta:
                print(f"Record {self._id}:✅ [{state}] done | {meta}")
            else:
                print(f"Record {self._id}:✅ [{state}] done")

            return result.modified_count > 0
        except Exception as e:
            print(f"Failed to mark_done {state} for {self._id}: {e}")
            return False

    def mark_fail(self, state: str, **meta) -> bool:
        """
        Mark state as failed.

        Args:
            state: State name
            **meta: Additional metadata (e.g., error='...', retry_count=3)

        Returns:
            bool: True if mark was successful


        """
        encoded = self._encode_state(False)
        update_doc = {f'stats.{state}': encoded}

        # Store metadata separately if provided
        if meta:
            self._doc['meta'][state] = meta
            update_doc[f'meta.{state}'] = meta

        try:
            self._doc['stats'][state] = encoded
            result = self._col.update_one(
                {'_id': self._id},
                {'$set': update_doc}
            )

            if meta:
                print(f"Record {self._id}:⛔ [{state}] failed | {meta}")
            else:
                print(f"Record {self._id}:⛔ [{state}] failed")

            return result.modified_count > 0
        except Exception as e:
            print(f"Failed to mark_fail {state} for {self._id}: {e}")
            return False

    def mark_reset(self, state: str) -> bool:
        """
        Reset/clear a state and its metadata.

        Args:
            state: State name to reset

        Returns:
            bool: True if reset was successful
        """
        try:
            self._doc['stats'][state] = None
            if state in self._doc['meta']:
                del self._doc['meta'][state]

            result = self._col.update_one(
                {'_id': self._id},
                {
                    '$unset': {
                        f'stats.{state}': '',
                        f'meta.{state}': ''
                    }
                }
            )
            print(f"Record {self._id}:⏺️ [{state}] reset")
            return result.modified_count > 0
        except Exception as e:
            print(f"Failed to reset {state} for {self._id}: {e}")
            return False

    def get_sv(self, state: str) -> Optional[Dict[str, Any]]:
        """
        Get state value (sv = state value) with metadata.

        Args:
            state: State name

        Returns:
            None if state doesn't exist, otherwise dict with:
                - success: bool (True if 'T', False if 'F')
                - ts: str (timestamp as YYYYMMDDHHMMSS)
                - timestamp: datetime object
                - meta: dict of additional metadata from meta field
        """
        raw = self._doc['stats'].get(state)
        if raw is None:
            return None
        return self._decode_state(raw, state)

    def get_meta(self, state: str) -> Dict[str, Any]:
        """Get metadata for a specific state."""
        return self._doc['meta'].get(state, {})

    def set_meta(self, state: str, **meta) -> bool:
        """
        Update metadata for a state without changing the state itself.

        Args:
            state: State name
            **meta: Metadata key-value pairs

        Returns:
            bool: True if update was successful
        """
        if not meta:
            return False

        try:
            if state not in self._doc['meta']:
                self._doc['meta'][state] = {}

            self._doc['meta'][state].update(meta)

            result = self._col.update_one(
                {'_id': self._id},
                {'$set': {f'meta.{state}': self._doc['meta'][state]}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Failed to set_meta for {state} on {self._id}: {e}")
            return False

    def has_state(self, state: str) -> bool:
        """Check if a state exists (regardless of success/fail)."""
        return state in self._doc['stats'] and self._doc['stats'][state] is not None

    def is_state_success(self, state: str) -> bool:
        """Check if a state exists and was successful."""
        sv = self.get_sv(state)
        return sv is not None and sv['success']

    def is_state_failed(self, state: str) -> bool:
        """Check if a state exists and failed."""
        sv = self.get_sv(state)
        return sv is not None and not sv['success']

    def get_all_states(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """Get all states with their decoded values and metadata."""
        return {
            state: self._decode_state(value, state) if value else None
            for state, value in self._doc['stats'].items()
        }


# ==============================================================================
# Query Helper Functions
# ==============================================================================

def query_stats(
        field: str,
        success: Optional[bool] = None,
        ts_after: Optional[str] = None,
        ts_before: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build query for state field with success and timestamp filters.

    Args:
        field: State field name
        success: True for T, False for F, None for untouched
        ts_after: Timestamp (YYYYMMDDHHMMSS) - records after this
        ts_before: Timestamp (YYYYMMDDHHMMSS) - records before this

    Returns:
        MongoDB query dict

    Examples:
        >>> query_stats('scraping', success=True)
        {'stats.scraping': {'$regex': '^T\\|'}}

        >>> query_stats('scraping', success=True, ts_after='20241115000000')
        {'stats.scraping': {'$regex': '^T\\|'}, '$expr': {...}}
    """
    q = {}
    f = f"stats.{field}"

    # Success / Fail / Untouched filter
    if success is True:
        q[f] = {"$regex": "^T\\|"}
    elif success is False:
        q[f] = {"$regex": "^F\\|"}
    elif success is None:
        q[f] = None

    # Timestamp filters
    if ts_after or ts_before:
        conditions = []

        if ts_after:
            conditions.append({
                "$gte": [
                    {"$substr": [f"${f}", 2, 14]},
                    ts_after
                ]
            })

        if ts_before:
            conditions.append({
                "$lte": [
                    {"$substr": [f"${f}", 2, 14]},
                    ts_before
                ]
            })

        if len(conditions) == 1:
            q["$expr"] = conditions[0]
        else:
            q["$expr"] = {"$and": conditions}

    return q


def query_success(
        state_name: str,
        ts_after: Optional[str] = None,
        ts_before: Optional[str] = None
) -> Dict[str, Any]:
    """Query for successful state (T prefix)."""
    return query_stats(state_name, success=True, ts_after=ts_after, ts_before=ts_before)


def query_failed(
        state_name: str,
        ts_after: Optional[str] = None,
        ts_before: Optional[str] = None
) -> Dict[str, Any]:
    """Query for failed state (F prefix)."""
    return query_stats(state_name, success=False, ts_after=ts_after, ts_before=ts_before)


def query_unprocessed(state_name: str) -> Dict[str, Any]:
    """Query for unprocessed records (state is None or missing)."""
    return {f"stats.{state_name}": None}


def query_with_meta(state_name: str, **meta_filters) -> Dict[str, Any]:
    """
    Query for records with specific metadata values.

    Args:
        state_name: State field name
        **meta_filters: Metadata key-value pairs to match

    Returns:
        MongoDB query dict

    Example:
        >>> query_with_meta('scraping', url='https://example.com')
        {'meta.scraping.url': 'https://example.com'}

        >>> query_with_meta('processing', retry={'$gte': 3})
        {'meta.processing.retry': {'$gte': 3}}
    """
    return {
        f"meta.{state_name}.{key}": value
        for key, value in meta_filters.items()
    }


def query_success_with_meta(
        state_name: str,
        ts_after: Optional[str] = None,
        ts_before: Optional[str] = None,
        **meta_filters
) -> Dict[str, Any]:
    """
    Combine success query with metadata filters.

    Example:
        >>> query_success_with_meta('scraping',
        ...                         ts_after='20241115000000',
        ...                         url='https://example.com')
    """
    q = query_success(state_name, ts_after=ts_after, ts_before=ts_before)
    q.update(query_with_meta(state_name, **meta_filters))
    return q


# ==============================================================================
# Module Exports
# ==============================================================================

__all__ = [
    'Record',
    'fetch',
    'validate_text',
    'LOCAL_FILE_HEADER',
    'query_stats'
]
