import unittest
import tempfile
import os
from transformers import HfArgumentParser
from trainer.arguments import EnhancedTrainingArguments, ModelDataArguments, ModelArguments, DataArguments


class TestArguments(unittest.TestCase):

    def test_enhanced_training_arguments(self):
        args = EnhancedTrainingArguments(
            output_dir="./output",
            temperature=0.1,
            use_fp16=True,
            lr_scheduler_kwargs={"warmup_steps": 100}
        )
        self.assertEqual(args.temperature, 0.1)
        self.assertTrue(args.use_fp16)
        self.assertTrue(args.fp16)
        self.assertEqual(args.fp16_backend, "amp")
        self.assertEqual(args.lr_scheduler_kwargs, {"warmup_steps": 100})

    def test_model_data_arguments(self):
        args = ModelDataArguments(
            model_name_or_path="bert-base-uncased",
            data_dir="./data",
            cache_dir="./cache",
            overwrite_cache=True
        )
        self.assertEqual(args.model_name_or_path, "bert-base-uncased")
        self.assertEqual(args.data_dir, "./data")
        self.assertEqual(args.cache_dir, "./cache")
        self.assertTrue(args.overwrite_cache)

    def test_model_arguments(self):
        args = ModelArguments(
            model_name_or_path="facebook/opt-350m",
            model_class_name="LlavaLlama",
            mm_tunable_parts="mm_mlp_adapter,mm_vision_resampler",
            freeze_backbone=True,
            vision_tower="openai/clip-vit-base-patch32",
            vision_tower_pretrained="openai/clip-vit-base-patch32"
        )
        self.assertEqual(args.model_name_or_path, "facebook/opt-350m")
        self.assertEqual(args.model_class_name, "LlavaLlama")
        self.assertEqual(args.mm_tunable_parts, "mm_mlp_adapter,mm_vision_resampler")
        self.assertTrue(args.freeze_backbone)
        self.assertEqual(args.vision_tower, "openai/clip-vit-base-patch32")
        self.assertEqual(args.vision_tower_pretrained, "openai/clip-vit-base-patch32")

    def test_data_arguments(self):
        args = DataArguments(
            data_path="./data/instruction.json",
            lazy_preprocess=True,
            is_multimodal=True,
            image_crop_resolution=224
        )
        self.assertEqual(args.data_path, "./data/instruction.json")
        self.assertTrue(args.lazy_preprocess)
        self.assertTrue(args.is_multimodal)
        self.assertEqual(args.image_crop_resolution, 224)

    def test_yaml_serialization(self):
        args = EnhancedTrainingArguments(output_dir="./output", temperature=0.1)
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tmp:
            args.to_yaml(tmp.name)
            parser = HfArgumentParser((EnhancedTrainingArguments,))
            train_args = parser.parse_yaml_file(yaml_file=tmp.name)[0]

            self.assertEqual(train_args.output_dir, "./output")
            self.assertEqual(train_args.temperature, 0.1)
        
        self.assertEqual(args.output_dir, train_args.output_dir)
        self.assertEqual(args.temperature, train_args.temperature)
        
        os.unlink(tmp.name)


if __name__ == '__main__':
    # python -m unittest test.test_args
    unittest.main()
