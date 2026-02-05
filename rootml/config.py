import yaml
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ExportConfig:
    input_files: List[str]
    tree: str
    features: List[str]
    label: str
    weight: Optional[str]
    event_id: List[str]
    selection: Optional[str]
    chunk_size: int = 100_000


def load_export_config(path: str) -> ExportConfig:

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    required = ["input_files", "tree", "features", "label", "event_id"]

    for key in required:
        if key not in data:
            raise ValueError(f"Missing required config field: {key}")

    return ExportConfig(
        input_files=data["input_files"],
        tree=data["tree"],
        features=data["features"],
        label=data["label"],
        weight=data.get("weight"),
        event_id=data["event_id"],
        selection=data.get("selection"),
        chunk_size=data.get("chunk_size", 100_000),
    )
