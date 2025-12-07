"""Storage backend abstraction for pluggable cloud and local storage.

This module provides a backend system that allows memora to transparently
use different storage backends (local SQLite, cloud-synced SQLite, etc.) while
keeping the same API surface.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tempfile
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import filelock
except ImportError:
    filelock = None

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    boto3 = None
    ClientError = None
    NoCredentialsError = None

logger = logging.getLogger(__name__)


class ConflictError(Exception):
    """Raised when a cloud sync conflict is detected (concurrent modification)."""
    pass


class StorageBackend(ABC):
    """Abstract base class for storage backends.

    Backends are responsible for:
    1. Providing a SQLite connection via connect()
    2. Syncing state before use (download from cloud, etc.)
    3. Syncing state after writes (upload to cloud, etc.)
    """

    @abstractmethod
    def connect(self, *, check_same_thread: bool = True) -> sqlite3.Connection:
        """Return a SQLite connection ready for use.

        For cloud backends, this may involve syncing from remote first.

        Args:
            check_same_thread: SQLite connection parameter

        Returns:
            sqlite3.Connection ready for queries
        """
        pass

    @abstractmethod
    def sync_before_use(self) -> None:
        """Sync state before using the database (e.g., download from cloud)."""
        pass

    @abstractmethod
    def sync_after_write(self) -> None:
        """Sync state after modifying the database (e.g., upload to cloud)."""
        pass

    @abstractmethod
    def get_info(self) -> dict:
        """Return diagnostic information about the backend."""
        pass


class LocalSQLiteBackend(StorageBackend):
    """Local file-based SQLite backend (original behavior)."""

    def __init__(self, db_path: Path):
        """Initialize local SQLite backend.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self._ensure_parent_dir()

    def _ensure_parent_dir(self) -> None:
        """Ensure parent directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self, *, check_same_thread: bool = True) -> sqlite3.Connection:
        """Return a connection to the local SQLite database."""
        conn = sqlite3.connect(self.db_path, check_same_thread=check_same_thread)
        conn.row_factory = sqlite3.Row
        return conn

    def sync_before_use(self) -> None:
        """No-op for local backend."""
        pass

    def sync_after_write(self) -> None:
        """No-op for local backend."""
        pass

    def get_info(self) -> dict:
        """Return backend information."""
        return {
            "backend_type": "local_sqlite",
            "db_path": str(self.db_path),
            "exists": self.db_path.exists(),
            "size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
        }


class CloudSQLiteBackend(StorageBackend):
    """Cloud-backed SQLite using local cache with sync to/from S3-compatible storage.

    This backend:
    - Downloads the SQLite file from cloud storage to a local cache
    - Serves all queries from the local cache (fast)
    - Uploads changes back to cloud storage after writes
    - Uses file locking to prevent concurrent corruption
    - Tracks dirty state to avoid unnecessary uploads
    """

    def __init__(
        self,
        cloud_url: str,
        cache_dir: Optional[Path] = None,
        encrypt: bool = False,
        compress: bool = False,
        auto_sync: bool = True,
    ):
        """Initialize cloud SQLite backend.

        Args:
            cloud_url: S3 URL (e.g., s3://bucket/path/to/db.sqlite)
            cache_dir: Local cache directory (default: ~/.cache/memora)
            encrypt: Enable server-side encryption on upload
            compress: Compress database before upload
            auto_sync: Automatically sync before/after operations
        """
        if boto3 is None:
            raise ImportError(
                "boto3 is required for cloud storage. "
                "Install with: pip install boto3"
            )

        if filelock is None:
            raise ImportError(
                "filelock is required for cloud storage. "
                "Install with: pip install filelock"
            )

        self.cloud_url = cloud_url
        self.encrypt = encrypt
        self.compress = compress
        self.auto_sync = auto_sync

        # Parse S3 URL
        self.bucket, self.key = self._parse_s3_url(cloud_url)

        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "memora"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create a unique cache path based on bucket + key
        cache_key = hashlib.sha256(f"{self.bucket}/{self.key}".encode()).hexdigest()[:16]
        self.cache_path = self.cache_dir / cache_key / "memories.db"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Lock file to prevent concurrent access
        self.lock_path = self.cache_path.parent / "sync.lock"
        self.lock = filelock.FileLock(self.lock_path, timeout=30)

        # Metadata file to track sync state
        self.meta_path = self.cache_path.parent / "metadata.json"

        # S3 client
        self.s3_client = boto3.client("s3")

        # Dirty tracking
        self._is_dirty = False
        self._last_hash = None

        logger.info(f"Initialized CloudSQLiteBackend: {cloud_url} -> {self.cache_path}")

    def _parse_s3_url(self, url: str) -> tuple[str, str]:
        """Parse S3 URL into bucket and key.

        Args:
            url: S3 URL like s3://bucket/path/to/file.db

        Returns:
            (bucket, key) tuple
        """
        if not url.startswith("s3://"):
            raise ValueError(f"Cloud URL must start with s3://, got: {url}")

        parts = url[5:].split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URL format: {url}")

        bucket, key = parts
        return bucket, key

    def _compute_hash(self) -> Optional[str]:
        """Compute hash of the local database file.

        Returns:
            SHA256 hash of the file, or None if file doesn't exist
        """
        if not self.cache_path.exists():
            return None

        sha256 = hashlib.sha256()
        with open(self.cache_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _load_metadata(self) -> dict:
        """Load sync metadata from cache."""
        if self.meta_path.exists():
            try:
                with open(self.meta_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        return {}

    def _save_metadata(self, metadata: dict) -> None:
        """Save sync metadata to cache."""
        try:
            with open(self.meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")

    def sync_before_use(self) -> None:
        """Download database from S3 if needed."""
        if not self.auto_sync:
            return

        with self.lock:
            try:
                # Check if remote object exists and get metadata
                try:
                    head_response = self.s3_client.head_object(
                        Bucket=self.bucket,
                        Key=self.key
                    )
                    remote_etag = head_response.get("ETag", "").strip('"')
                    remote_modified = head_response.get("LastModified")
                except ClientError as e:
                    if e.response["Error"]["Code"] == "404":
                        # Remote object doesn't exist yet - create empty local DB
                        logger.info(f"Remote database not found, will create new one")
                        return
                    raise

                # Load local metadata
                metadata = self._load_metadata()
                local_etag = metadata.get("etag")

                # Skip download if local cache is up to date
                if local_etag == remote_etag and self.cache_path.exists():
                    logger.debug(f"Local cache is up to date (ETag: {remote_etag})")
                    self._last_hash = self._compute_hash()
                    return

                # Download from S3
                logger.info(f"Downloading {self.bucket}/{self.key} to {self.cache_path}")
                start_time = time.time()

                # Download to temporary file first
                temp_path = self.cache_path.parent / f"{self.cache_path.name}.tmp"
                self.s3_client.download_file(self.bucket, self.key, str(temp_path))

                # Move to final location
                shutil.move(str(temp_path), str(self.cache_path))

                duration = time.time() - start_time
                size_mb = self.cache_path.stat().st_size / (1024 * 1024)
                logger.info(f"Downloaded {size_mb:.2f} MB in {duration:.2f}s")

                # Update metadata
                metadata["etag"] = remote_etag
                metadata["last_sync"] = datetime.now().isoformat()
                metadata["remote_modified"] = remote_modified.isoformat() if remote_modified else None
                self._save_metadata(metadata)

                # Update hash
                self._last_hash = self._compute_hash()
                self._is_dirty = False

            except NoCredentialsError:
                logger.error("AWS credentials not found")
                raise
            except Exception as e:
                logger.error(f"Failed to sync from cloud: {e}")
                raise

    def sync_after_write(self) -> None:
        """Upload database to S3 if dirty."""
        if not self.auto_sync:
            return

        # Fast path: check dirty flag first (avoids expensive hashing)
        if not self._is_dirty:
            logger.debug("Database not dirty, skipping sync")
            return

        with self.lock:
            try:
                # Double-check dirty flag under lock
                if not self._is_dirty:
                    logger.debug("Database not dirty (checked under lock), skipping sync")
                    return

                if not self.cache_path.exists():
                    logger.warning("Cache file doesn't exist, nothing to upload")
                    return

                # Compute hash to detect changes (only when dirty flag is set)
                current_hash = self._compute_hash()
                if current_hash == self._last_hash:
                    # False positive - dirty flag was set but content unchanged
                    logger.debug("Database unchanged after hashing, skipping upload")
                    self._is_dirty = False
                    return

                # Check for conflicts before uploading
                # Load the last known remote ETag
                metadata = self._load_metadata()
                last_known_etag = metadata.get("etag")

                # Verify remote hasn't changed since our last sync
                if last_known_etag:
                    try:
                        current_remote = self.s3_client.head_object(
                            Bucket=self.bucket,
                            Key=self.key
                        )
                        current_remote_etag = current_remote.get("ETag", "").strip('"')

                        if current_remote_etag != last_known_etag:
                            # Conflict detected: remote was modified by another writer
                            logger.error(
                                f"Conflict detected! Remote object changed since last sync. "
                                f"Expected ETag: {last_known_etag}, "
                                f"Current ETag: {current_remote_etag}"
                            )
                            raise ConflictError(
                                f"Database was modified by another process. "
                                f"Run 'memora-server sync-pull' to get latest changes."
                            )
                    except ClientError as e:
                        if e.response["Error"]["Code"] != "404":
                            raise

                # Upload to S3
                logger.info(f"Uploading {self.cache_path} to {self.bucket}/{self.key}")
                start_time = time.time()

                extra_args = {}
                if self.encrypt:
                    extra_args["ServerSideEncryption"] = "AES256"

                self.s3_client.upload_file(
                    str(self.cache_path),
                    self.bucket,
                    self.key,
                    ExtraArgs=extra_args if extra_args else None
                )

                duration = time.time() - start_time
                size_mb = self.cache_path.stat().st_size / (1024 * 1024)
                logger.info(f"Uploaded {size_mb:.2f} MB in {duration:.2f}s")

                # Update metadata with new remote state
                head_response = self.s3_client.head_object(
                    Bucket=self.bucket,
                    Key=self.key
                )
                metadata = {
                    "etag": head_response.get("ETag", "").strip('"'),
                    "last_sync": datetime.now().isoformat(),
                    "remote_modified": head_response.get("LastModified").isoformat() if head_response.get("LastModified") else None,
                }
                self._save_metadata(metadata)

                # Update tracking
                self._last_hash = current_hash
                self._is_dirty = False

            except ConflictError:
                # Re-raise conflict errors without wrapping
                raise
            except Exception as e:
                logger.error(f"Failed to sync to cloud: {e}")
                raise

    def connect(self, *, check_same_thread: bool = True) -> sqlite3.Connection:
        """Return a connection to the cached SQLite database.

        This will sync from cloud if needed before returning the connection.
        """
        # Sync from cloud before use
        self.sync_before_use()

        # Create connection to local cache
        conn = sqlite3.connect(self.cache_path, check_same_thread=check_same_thread)
        conn.row_factory = sqlite3.Row

        # Mark backend as dirty when connection commits
        # We use a wrapper class to intercept commits since in Python 3.13+
        # sqlite3.Connection methods are read-only
        class TrackedConnection:
            def __init__(self, conn, backend):
                self._conn = conn
                self._backend = backend

            def __getattr__(self, name):
                attr = getattr(self._conn, name)
                if name == 'commit':
                    def wrapped_commit(*args, **kwargs):
                        result = attr(*args, **kwargs)
                        self._backend._is_dirty = True
                        logger.debug("Database marked as dirty after commit")
                        return result
                    return wrapped_commit
                return attr

            def __enter__(self):
                return self._conn.__enter__()

            def __exit__(self, *args):
                return self._conn.__exit__(*args)

        return TrackedConnection(conn, self)

    def get_info(self) -> dict:
        """Return backend information."""
        metadata = self._load_metadata()
        return {
            "backend_type": "cloud_sqlite",
            "cloud_url": self.cloud_url,
            "bucket": self.bucket,
            "key": self.key,
            "cache_path": str(self.cache_path),
            "cache_exists": self.cache_path.exists(),
            "cache_size_bytes": self.cache_path.stat().st_size if self.cache_path.exists() else 0,
            "is_dirty": self._is_dirty,
            "last_etag": metadata.get("etag"),
            "last_sync": metadata.get("last_sync"),
            "auto_sync": self.auto_sync,
            "encrypt": self.encrypt,
        }

    def force_sync_pull(self) -> None:
        """Force download from cloud, ignoring local state."""
        with self.lock:
            logger.info("Forcing sync pull from cloud")
            # Clear metadata to force download
            if self.meta_path.exists():
                self.meta_path.unlink()
            self.sync_before_use()

    def force_sync_push(self) -> None:
        """Force upload to cloud, even if not dirty."""
        with self.lock:
            logger.info("Forcing sync push to cloud")
            self._is_dirty = True
            self._last_hash = None  # Force hash mismatch
            self.sync_after_write()


def parse_backend_uri(uri: str) -> StorageBackend:
    """Parse a storage URI and return the appropriate backend.

    Supported URI formats:
    - file:///path/to/db.sqlite (local SQLite)
    - /path/to/db.sqlite (local SQLite)
    - s3://bucket/path/to/db.sqlite (S3-compatible cloud storage)

    Args:
        uri: Storage URI string

    Returns:
        StorageBackend instance
    """
    if uri.startswith("s3://"):
        # Parse cloud storage options from environment
        encrypt = os.getenv("MEMORA_CLOUD_ENCRYPT", "").lower() in ("1", "true", "yes")
        compress = os.getenv("MEMORA_CLOUD_COMPRESS", "").lower() in ("1", "true", "yes")
        cache_dir_env = os.getenv("MEMORA_CACHE_DIR")
        cache_dir = Path(cache_dir_env) if cache_dir_env else None

        return CloudSQLiteBackend(
            cloud_url=uri,
            cache_dir=cache_dir,
            encrypt=encrypt,
            compress=compress,
        )

    elif uri.startswith("file://"):
        # file:// URI
        path = uri[7:]  # Remove file://
        return LocalSQLiteBackend(Path(path))

    else:
        # Assume local path
        return LocalSQLiteBackend(Path(uri))
