"""Model Registry for Versioned Model Binary Storage, Champion Promotion, and Automated Rollback."""

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any
import joblib
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("training.model_registry")


class ModelRegistry:
    """Manages versioned model binaries, metadata manifests, Champion promotion, and rollback."""

    def __init__(self, registry_dir: Path | None = None) -> None:
        self.registry_dir = registry_dir or settings.models_dir / "registry"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.registry_dir / "registry_manifest.json"
        self._init_manifest()

    def _init_manifest(self) -> None:
        """Initialize registry manifest JSON if missing."""
        if not self.manifest_path.exists():
            initial_data = {
                "active_champion": None,
                "previous_champion": None,
                "history": [],
            }
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, indent=2)

    def load_manifest(self) -> dict[str, Any]:
        """Load registry manifest dict."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_manifest(self, manifest_data: dict[str, Any]) -> None:
        """Save registry manifest dict."""
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

    def register_version(
        self,
        model_obj: Any,
        version_tag: str,
        metrics: dict[str, Any],
        description: str = "Candidate model version",
    ) -> Path:
        """Save versioned model binary and record metadata entry."""
        model_filename = f"model_{version_tag}.joblib"
        artifact_path = self.registry_dir / model_filename
        joblib.dump(model_obj, artifact_path)

        with open(artifact_path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()

        entry = {
            "version_tag": version_tag,
            "filename": model_filename,
            "filepath": str(artifact_path),
            "sha256": sha256,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": metrics,
            "description": description,
        }

        manifest = self.load_manifest()
        manifest["history"].append(entry)
        self.save_manifest(manifest)

        logger.info(f"Registered model version '{version_tag}' in registry (SHA256: {sha256[:10]}...).")
        return artifact_path

    def promote_to_champion(self, version_tag: str) -> None:
        """Promote a registered model version to active Champion and update production artifact."""
        manifest = self.load_manifest()
        entry = next((e for e in manifest["history"] if e["version_tag"] == version_tag), None)

        if entry is None:
            raise ValueError(f"Version '{version_tag}' not found in model registry!")

        source_path = Path(entry["filepath"])
        # Use a stable, model-type-agnostic champion artifact name
        prod_target_path = settings.artifacts_dir / "champion_model.joblib"

        # Copy to active production artifact
        shutil.copy2(source_path, prod_target_path)

        # Keep full rollback chain: push current champion into previous_champion
        current_champion = manifest.get("active_champion")
        manifest["previous_champion"] = current_champion
        manifest["active_champion"] = entry
        self.save_manifest(manifest)

        logger.info(f"PROMOTED model version '{version_tag}' to Active Champion (Copied to {prod_target_path.name}).")  

    def rollback_champion(self) -> bool:
        """Rollback active Champion to previous Champion version.

        After rollback the demoted version is stored in ``previous_champion`` so
        a second rollback call can re-apply it if needed.
        """
        manifest = self.load_manifest()
        prev = manifest.get("previous_champion")

        if prev is None:
            logger.warning("Rollback failed: No previous Champion found in registry!")
            return False

        prev_path = Path(prev["filepath"])
        prod_target_path = settings.artifacts_dir / "champion_model.joblib"

        shutil.copy2(prev_path, prod_target_path)

        # Swap champion/previous so the demoted version is still reachable
        demoted = manifest["active_champion"]
        manifest["active_champion"] = prev
        manifest["previous_champion"] = demoted
        self.save_manifest(manifest)

        logger.warning(f"ROLLBACK EXECUTED: Restored previous Champion version '{prev['version_tag']}'.")
        return True
