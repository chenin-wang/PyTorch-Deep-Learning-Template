from dataclasses import dataclass, field
from typing import Dict, List,Sequence, Tuple, Union, Any, Callable, Optional
from typing import Dict, List,Sequence, Tuple, Union, Any, Callable, Optional
from transformers import TrainingArguments
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
import yaml
import logging
from loggers.logging_colors import get_logger

logger = get_logger()

class ConfigurableBase:
    @classmethod
    def from_yaml(cls, file_path: str):
        """Load configuration from a YAML file."""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = OmegaConf.create(config_dict)
        default_config = OmegaConf.structured(cls)
        merged_config = OmegaConf.merge(default_config, config)
        
        return instantiate(merged_config)

    def to_yaml(self, file_path: str):
        """Save the configuration to a YAML file."""
        config_dict = OmegaConf.to_container(OmegaConf.structured(self), resolve=True)
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {file_path}")


@dataclass
class EnhancedTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="output",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    temperature: float = field(
        default=0.05,
        metadata={"help": "Temperature parameter for controlling randomness in sampling."}
    )
    use_fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use mixed precision training."}
    )

    # Override the lr_scheduler_kwargs field
    lr_scheduler_kwargs: Optional[Dict[str, Union[str, float, int]]] = field(
        default=None,
        metadata={"help": "Additional arguments for the learning rate scheduler."}
    )

    # Override the lr_scheduler_kwargs field
    lr_scheduler_kwargs: Optional[Dict[str, Union[str, float, int]]] = field(
        default=None,
        metadata={"help": "Additional arguments for the learning rate scheduler."}
    )

    def __post_init__(self):
        super().__post_init__()
        if self.use_fp16:
            self.fp16 = True
            self.fp16_backend = "amp"
        self.log_level = logging.INFO
        self.log_level_replica = logging.WARNING

    def to_yaml(self, file_path: str):
        """Save the configuration to a YAML file."""
        config_dict = self.to_dict()
        
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {file_path}")
        self.log_level = logging.INFO
        self.log_level_replica = logging.WARNING

@dataclass
class ModelDataArguments(ConfigurableBase):
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    data_dir: str = field(metadata={"help": "Directory containing the dataset files"})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory to store the preprocessed datasets"}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not"}
    )

@dataclass
class ModelArguments(ConfigurableBase):
    model_name_or_path: str = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Used to init model class, format is XXXXForCausalLM. "
                    "e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"
        }
    )
    mm_tunable_parts: Optional[str] = field(
        default=None,
        metadata={
            "help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model"'
        }
    )
    version: str = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)


@dataclass
class DataArguments(ConfigurableBase):
    data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the training data, in llava's instruction.json format. "
                    "Supporting multiple json files via /path/to/{a,b,c}.json"
        }
    )
    lazy_preprocess: bool = field(default=False)
    is_multimodal: bool = field(default=False)
    image_crop_resolution: Optional[int] = field(default=None)


if __name__ == "__main__":
    print(compose("config"))