from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ModelInfo:
    weight_url: str
    config_url: str
    meta: Dict[str, Any]


class PretrainedModels:
    mlp_wacsf_v1 = ModelInfo(
        weight_url="https://github.com/shelllbw/xanesnet-pretrained-models/raw/refs/heads/master/mlp_wacsf_v1/weights.pth",
        config_url="https://github.com/shelllbw/xanesnet-pretrained-models/raw/refs/heads/master/mlp_wacsf_v1/metadata.yaml",
        meta={"num_params": 123456, "_docs": """Example pretrained model"""},
    )

    gnn_pdos_v1 = ModelInfo(
        weight_url="https://github.com/shelllbw/xanesnet-models/raw/refs/heads/master/mlp_wacsf_v1/weights.pth",
        config_url="https://github.com/shelllbw/xanesnet-models/raw/refs/heads/master/mlp_wacsf_v1/metadata.yaml",
        meta={"num_params": 123456, "_docs": """Example pretrained model"""},
    )
