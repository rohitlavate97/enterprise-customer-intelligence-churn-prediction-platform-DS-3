"""Dataset hashing, metadata versioning, and provenance tracking."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("data.versioning")


class DatasetVersionManager:
    """Computes SHA256 hashes and manages versioned data manifests for reproducibility."""

    @staticmethod
    def compute_sha256(filepath: Path) -> str:
        """Compute SHA256 checksum of a file on disk."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @classmethod
    def register_version(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        output_dir: Path,
        metadata: dict | None = None,
    ) -> tuple[Path, str]:
        """Save DataFrame and generate a version manifest with SHA256 hash and metadata."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_{timestamp}.csv"
        filepath = output_dir / filename

        df.to_csv(filepath, index=False)
        checksum = self.compute_sha256(filepath)

        manifest = {
            "dataset_name": dataset_name,
            "filename": filename,
            "filepath": str(filepath.resolve()),
            "timestamp_utc": timestamp,
            "sha256_checksum": checksum,
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "custom_metadata": metadata or {},
        }

        manifest_path = output_dir / f"{dataset_name}_{timestamp}_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Registered dataset version '{dataset_name}' [SHA256: {checksum[:12]}...] at {filepath}")
        return filepath, checksum
