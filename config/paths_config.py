#!/usr/bin/env python3
"""
Minimale Pfad-Konfiguration für bl-churn (Standalone-Repo)
Verwendet OUTBOX_ROOT (ENV) oder fallback auf project_root/dynamic_system_outputs/outbox.
"""

import os
from pathlib import Path
from typing import List


class ProjectPaths:
    """
    Minimaler Satz an Pfaden/Utilities, kompatibel zu Sandbox-Imports.
    """

    _project_root: Path = None

    @classmethod
    def _initialize_root(cls) -> Path:
        if cls._project_root is None:
            current_file = Path(__file__).resolve()
            cls._project_root = current_file.parent.parent
        return cls._project_root

    @classmethod
    def project_root(cls) -> Path:
        return cls._initialize_root()

    # Haupt-Verzeichnisse
    @classmethod
    def models_directory(cls) -> Path:
        return cls.project_root() / "models"

    @classmethod
    def input_data_directory(cls) -> Path:
        return cls.project_root() / "input_data"

    @classmethod
    def config_directory(cls) -> Path:
        # Shared-Config als Standard (Submodule unter config/shared/config)
        shared = cls.project_root() / "config" / "shared" / "config"
        return shared if shared.exists() else cls.project_root() / "config"

    @classmethod
    def dynamic_system_outputs_directory(cls) -> Path:
        return cls.project_root() / "dynamic_system_outputs"

    # OUTBOX
    @classmethod
    def outbox_directory(cls) -> Path:
        env_root = os.environ.get("OUTBOX_ROOT")
        if env_root:
            try:
                p = Path(env_root).resolve()
                p.mkdir(parents=True, exist_ok=True)
                return p
            except Exception:
                pass
        return cls.dynamic_system_outputs_directory() / "outbox"

    @classmethod
    def outbox_churn_directory(cls) -> Path:
        return cls.outbox_directory() / "churn"

    @classmethod
    def outbox_cox_directory(cls) -> Path:
        return cls.outbox_directory() / "cox"

    @classmethod
    def outbox_counterfactuals_directory(cls) -> Path:
        return cls.outbox_directory() / "counterfactuals"

    @classmethod
    def outbox_churn_experiment_directory(cls, experiment_id: int) -> Path:
        return cls.outbox_churn_directory() / f"experiment_{int(experiment_id)}"

    @classmethod
    def outbox_cox_experiment_directory(cls, experiment_id: int) -> Path:
        return cls.outbox_cox_directory() / f"experiment_{int(experiment_id)}"

    # Konfigurationsdateien
    @classmethod
    def data_dictionary_file(cls) -> Path:
        return cls.config_directory() / "data_dictionary_optimized.json"

    @classmethod
    def feature_mapping_file(cls) -> Path:
        return cls.config_directory() / "feature_mapping.json"

    @classmethod
    def cf_cost_policy_file(cls) -> Path:
        return cls.config_directory() / "cf_cost_policy.json"

    # Convenience/Backwards-Compatibility
    @classmethod
    def get_input_data_path(cls) -> Path:
        return cls.input_data_directory() / "churn_Data_cleaned.csv"

    @classmethod
    def get_models_directory(cls) -> Path:
        return cls.models_directory()

    @classmethod
    def get_data_dictionary_file(cls) -> Path:
        return cls.data_dictionary_file()

    @classmethod
    def ensure_directory_exists(cls, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    # Utility (optional)
    @classmethod
    def get_latest_file_by_pattern(cls, directory: Path, pattern: str) -> Path:
        if not directory.exists():
            return None
        matches = list(directory.glob(pattern))
        if not matches:
            return None
        return max(matches, key=lambda p: p.stat().st_mtime)


