#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .paths_config import ProjectPaths


@dataclass
class DataAccess:
    input_csv_path: Path
    models_dir: Path


def get_data_access() -> DataAccess:
    return DataAccess(
        input_csv_path=ProjectPaths.get_input_data_path(),
        models_dir=ProjectPaths.models_directory(),
    )


