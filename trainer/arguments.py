from dataclasses import dataclass, field
from typing import List, Tuple
from transformers import TrainingArguments
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from typing import Dict, Optional, Sequence, List


@dataclass
class EnhancedTrainingArguments(TrainingArguments):
    temperature: float = field(
        default=0.05,
        metadata={
            "help": "Temperature parameter for controlling randomness in sampling."
        },
    )

    use_fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use mixed precision training."},
    )

    def __post_init__(self):
        if self.use_fp16:
            self.fp16 = True
            self.fp16_backend = "amp"

        self.log_level = "logging.INFO"
        self.log_level_replica = "logging.WARNING"


@dataclass
class ModelDataArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    data_dir: str = field(metadata={"help": "Directory containing the dataset files"})
    cache_dir: str = field(
        default=None, metadata={"help": "Directory to store the preprocessed datasets"}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not"},
    )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Used to init model class, format is XXXXForCausalLM. \
            e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"
        },
    )
    mm_tunable_parts: Optional[str] = field(
        default=None,
        metadata={
            "help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model"'
        },
    )
    # deciding which part of the multimodal model to tune, will overwrite other previous settings

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(
        default=None
    )  # default to the last layer


@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={
            "help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"
        },
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_crop_resolution: Optional[int] = field(default=None)
