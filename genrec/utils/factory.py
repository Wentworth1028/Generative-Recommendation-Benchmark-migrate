from __future__ import annotations

from genrec.data.collators.generative.ghost_collator import GhostDataCollator
from genrec.data.collators.generative.dros_collator import DROSTigerDataCollator
from genrec.data.collators.generative.tiger_collator import TigerDataCollator
from genrec.data.datasets.generative.dros_dataset import DROSTigerDataset
from genrec.data.datasets.generative.ghost_dataset import GhostDataset
from genrec.data.datasets.generative.tiger_dataset import TigerDataset
from genrec.quantization.pipelines.ghost_pipeline import GhostSKTTrainingPipeline
from genrec.quantization.pipelines.rqvae_pipeline import RQVAETrainingPipeline
from genrec.quantization.pipelines.rqvae_pipeline_letter import LETTERRQVAETrainingPipeline
from genrec.quantization.tokenizers.ghost_tokenizer import GhostSKTTokenizer
from genrec.quantization.tokenizers.letter_tokenizer import LETTERRQVAETokenizer
from genrec.quantization.tokenizers.rqvae_tokenizer import RQVAETokenizer
from genrec.utils.models_setup.care_setup import create_care_letter_model, create_care_tiger_model
from genrec.utils.models_setup.ghost_setup import create_ghost_letter_model, create_ghost_tiger_model
from genrec.utils.models_setup.letter_setup import create_letter_model
from genrec.utils.models_setup.tiger_setup import create_tiger_model
from genrec.utils.models_setup.twmtl_setup import create_twmtl_letter_model, create_twmtl_tiger_model


MODEL_FACTORY = {
    "tiger": create_tiger_model,
    "tiger_resn": create_tiger_model,
    "tiger_logdet": create_tiger_model,
    "tiger_dros": create_tiger_model,
    "letter": create_letter_model,
    "letter_resn": create_letter_model,
    "letter_logdet": create_letter_model,
    "letter_dros": create_letter_model,
    "tiger_twmtl": create_twmtl_tiger_model,
    "letter_twmtl": create_twmtl_letter_model,
    "tiger_care": create_care_tiger_model,
    "letter_care": create_care_letter_model,
    "tiger_crab": create_tiger_model,
    "letter_crab": create_letter_model,
    "ghost_tiger": create_ghost_tiger_model,
    "ghost_letter": create_ghost_letter_model,
    "ghost_tiger_skt_only": create_ghost_tiger_model,
    "ghost_letter_skt_only": create_ghost_letter_model,
}

DATASET_MAP = {
    "tiger": TigerDataset,
    "tiger_resn": TigerDataset,
    "tiger_logdet": TigerDataset,
    "tiger_dros": DROSTigerDataset,
    "letter": TigerDataset,
    "letter_resn": TigerDataset,
    "letter_logdet": TigerDataset,
    "letter_dros": DROSTigerDataset,
    "tiger_twmtl": TigerDataset,
    "letter_twmtl": TigerDataset,
    "tiger_care": TigerDataset,
    "letter_care": TigerDataset,
    "tiger_crab": TigerDataset,
    "letter_crab": TigerDataset,
    "ghost_tiger": GhostDataset,
    "ghost_letter": GhostDataset,
    "ghost_tiger_skt_only": GhostDataset,
    "ghost_letter_skt_only": GhostDataset,
}

COLLATOR_MAP = {
    "tiger": TigerDataCollator,
    "tiger_resn": TigerDataCollator,
    "tiger_logdet": TigerDataCollator,
    "tiger_dros": DROSTigerDataCollator,
    "letter": TigerDataCollator,
    "letter_resn": TigerDataCollator,
    "letter_logdet": TigerDataCollator,
    "letter_dros": DROSTigerDataCollator,
    "tiger_twmtl": TigerDataCollator,
    "letter_twmtl": TigerDataCollator,
    "tiger_care": TigerDataCollator,
    "letter_care": TigerDataCollator,
    "tiger_crab": TigerDataCollator,
    "letter_crab": TigerDataCollator,
    "ghost_tiger": GhostDataCollator,
    "ghost_letter": GhostDataCollator,
    "ghost_tiger_skt_only": GhostDataCollator,
    "ghost_letter_skt_only": GhostDataCollator,
}

PIPELINE_MAP = {
    "tiger": RQVAETrainingPipeline,
    "tiger_resn": RQVAETrainingPipeline,
    "tiger_logdet": RQVAETrainingPipeline,
    "tiger_dros": RQVAETrainingPipeline,
    "letter": LETTERRQVAETrainingPipeline,
    "letter_resn": LETTERRQVAETrainingPipeline,
    "letter_logdet": LETTERRQVAETrainingPipeline,
    "letter_dros": LETTERRQVAETrainingPipeline,
    "ghost": GhostSKTTrainingPipeline,
    "ghost_tiger": GhostSKTTrainingPipeline,
    "ghost_letter": GhostSKTTrainingPipeline,
    "ghost_tiger_skt_only": GhostSKTTrainingPipeline,
    "ghost_letter_skt_only": GhostSKTTrainingPipeline,
}

TOKENIZER_CLASS_MAP = {
    "tiger": RQVAETokenizer,
    "letter": LETTERRQVAETokenizer,
    "ghost": GhostSKTTokenizer,
}


def get_model_factory(name: str):
    if name not in MODEL_FACTORY:
        raise ValueError(f"Unknown generative type: '{name}', options: {list(MODEL_FACTORY.keys())}")
    return MODEL_FACTORY[name]


def get_dataset_class(name: str):
    if name not in DATASET_MAP:
        raise ValueError(f"Unknown generative type: '{name}', options: {list(DATASET_MAP.keys())}")
    return DATASET_MAP[name]


def get_collator_class(name: str):
    if name not in COLLATOR_MAP:
        raise ValueError(f"Unknown generative type: '{name}', options: {list(COLLATOR_MAP.keys())}")
    return COLLATOR_MAP[name]


def get_pipeline_class(name: str):
    if name not in PIPELINE_MAP:
        raise ValueError(f"Unknown tokenizer type: '{name}', options: {list(PIPELINE_MAP.keys())}")
    return PIPELINE_MAP[name]


def get_tokenizer_class(name: str):
    if name not in TOKENIZER_CLASS_MAP:
        raise ValueError(f"Unknown tokenizer kind: '{name}', options: {list(TOKENIZER_CLASS_MAP.keys())}")
    return TOKENIZER_CLASS_MAP[name]
