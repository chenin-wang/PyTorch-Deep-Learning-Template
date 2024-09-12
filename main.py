import os
import debugpy

try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

import pathlib
import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple, Union
from data.DataLoader import get_dataloaders
from data.Dataset import train_ds, test_ds
from models.model import CustomModel
from trainer.arguments import EnhancedTrainingArguments,ModelArguments, DataArguments
from trainer.trainer import EnhancedTrainer
from transformers import HfArgumentParser, PreTrainedModel,PretrainedConfig,PreTrainedTokenizerBase, TrainingArguments, TrainerCallback, DataCollator, EvalPrediction
from torch.utils.data import Dataset, IterableDataset
from .loggers.logging_colors import get_logger

logger = get_logger(__name__)

def main():
    # model
    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
    config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    ignore_mismatched_sizes: bool = False,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
    revision: str = "main",
    use_safetensors: bool = None,
    # train
    model: Union[PreTrainedModel, nn.Module] = None,
    training_args: TrainingArguments = None,
    data_collator: Optional[DataCollator] = None,
    train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
    eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    model_init: Optional[Callable[[], PreTrainedModel]] = None,
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    callbacks: Optional[List[TrainerCallback]] = None,
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,

    parser = HfArgumentParser((ModelArguments, DataArguments, EnhancedTrainingArguments))
    model_args,data_args,training_args = parser.parse_args_into_dataclasses()

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model Data parameters %s", model_args)

    model = CustomModel.from_pretrained(pretrained_model_name_or_path,
                                        model_args,
                                        config = config, 
                                        cache_dir = cache_dir,
                                        ignore_mismatched_sizes = ignore_mismatched_sizes,
                                        use_safetensors = use_safetensors)

    trainer = EnhancedTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        model_init=model_init,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        optimizers=optimizers,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    main()