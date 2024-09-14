# PyTorch Deep Learning Template
development is ongoing. Please stay tuned for updates.

A clean and modular template to kickstart your next deep learning project 🚀🚀

## Key Features

- **Modularity**: Logical components separated into different Python submodules
- **Ready to Go**: Uses [transformers](https://github.com/huggingface/transformers) and [accelerate](https://github.com/huggingface/accelerate) to eliminate boilerplate code
- **Customizable**: Easily swap models, loss functions, and optimizers
- **Logging**: Utilizes Python's [logging](https://docs.python.org/3/library/logging.html) module 
- **Experiment Tracking**: Integrates [Weights & Biases](https://www.wandb.ai) for comprehensive experiment monitoring
- **Metrics**: Uses [torchmetrics](https://github.com/Lightning-AI/metrics) for efficient metric computation and [evaluate](https://github.com/huggingface/evaluate) for multi-metric model evaluation
- **Playground**: Jupyter notebook for quick experimentation and prototyping

### Motivation
Let's face it, usually data scientists are not software engineers and they usually end up with spaghetti code, most of the time on a big unusable Jupiter-notebook. With this repo, I have proposed a clean example of how your code should be split and modularized to make scalability and sharability possible. In this example, we will try to classify Darth Vader and Luke Skywalker. We have 100 images per class gathered using google images. The dataset is [here](https://drive.google.com/open?id=1LyHJxUVjOgDIgGJL4MnDhA10xjejWuw7). You just have to extract it in this folder and run main.py. We are fine-tuning resnet18 and it should be able to reach > 90% accuracy in 5/10 epochs.
## Structure
The template is inside `./template`.
```bash
.
│  .gitignore
│  main.py # main script to run the project
│  playground.ipynb # a notebook to play around with the code
│  README.md
│  requirements.txt
│  test.py
│  train.sh
│
├─callbacks # Callbacks for training and logging
│      CometCallback.py
│      __init__.py
│
├─configs # Config files
│      config.yaml
│      ds_zero2_no_offload.json
│
├─data # Data module
│  │  DataLoader.py
│  │  Dataset.py
│  │  __init__.py
│  │
│  └─transformations
│          transforms.py
│          __init__.py
│
├─loggers # Logging module
│  │  logging_colors.py
│
├─losses # Losses module
│      loss.py
│      __init__.py
│
├─metrics # Metrics module
│      metric.py
│      __init__.py
│
├─models # Models module
│  │  modelutils.py
│  │  __init__.py
│  │
│  ├─HFModel # HuggingFace models
│  │      configuration_hfmodel.py
│  │      convert_hfmodel_original_pytorch_to_hf.py
│  │      feature_extraction_hfmodel.py # audio processing
│  │      image_processing_hfmodel.py # Image processing
│  │      modeling_hfmodel.py # Modeling
│  │      processing_hfmodel.py # mutimodal processing
│  │      tokenization_hfmodel.py # Tokenization
│  │      tokenization_hfmodel_fast.py
│  │      __init__.py
│  │
│  └─TorchModel # Torch models
│          modeling_torch.py
│          utils.py
│          __init__.py
│
├─onnx # ONNX module
│      converter2onnx.py
│
├─trainer # Trainer module 
│      acclerate.py
│      arguments.py
│      evaluater.py
│      inference.py
│      trainer.py
│      __init__.py
│
└─utils
        constants.py
        profiler.py # Profiling module
        utils.py

```
## Key Components

### Project Structure
- Maintain a clean and modular structure
- Define paths and constants in a central location (e.g. `Project.py`)
- Use `pathlib.Path` for cross-platform compatibility

### Data Processing
- Implement custom datasets by subclassing `torch.utils.data.Dataset`
- Define data transformations in `data/transformations/`
- Use `get_dataloaders()` to configure train/val/test loaders

### Modeling
- Define models in the `models/` directory
- Implement custom architectures or modify existing ones as needed

### Training and Evaluation  
- Utilize `main.py` for training/evaluation logic
- Leverage libraries like Accelerate for distributed training
- Implement useful callbacks:
  - Learning rate scheduling
  - Model checkpointing
  - Early stopping

### Logging and Experiment Tracking
- Use Python's `logging` module for consistent logging
- Integrate experiment tracking (e.g. Weights & Biases, MLflow)

### Utilities
- Implement helper functions for visualization, profiling, etc.
- Store in `utils/` directory

## Best Practices

- Avoid hardcoding paths - use a centralized configuration
- Modularize code for reusability and maintainability  
- Leverage existing libraries and tools when possible
- Document code and maintain a clear project structure
- Use version control and create reproducible experiments

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Modify `Project.py` with your paths/constants
4. Implement your custom dataset/model as needed
5. Run training: `python main.py`

This template addresses the common challenge of unstructured and hard-to-maintain code in data science projects. It provides a clean, modular structure that promotes scalability and shareability. The example project demonstrates image classification using a fine-tuned ResNet18 model on a Star Wars character dataset.

## Project Structure
The project is structured in a modular way, with separate folders for data processing, modeling, training and utilities. The `Project` class in `Project.py` stores paths and constants that are used throughout the codebase.

## Data Processing

Data processing is handled by the `get_dataloaders()` function in `data/datasets.py`. It takes in the dataset name and splits it into train/val/test sets using a predefined split ratio. Transforms can be applied to each set as needed.

## Modeling

Models are defined in `models/modeling_torch.py`. This file contains the implementation of a simple CNN architecture for image classification. You can modify or add your own models here.
## Training

Training is handled by the `train()` function in `train.py`. It takes in the model, dataloaders, and training parameters, and trains the model using the specified optimizer and loss function.

## Utilities

Utilities such as logging, saving, and loading models are handled by the `utils.py` file. This file contains functions for saving and loading models, as well as logging training progress.

## Example Usage

To train the model, run the following command:

```bash
python main.py
```
This will train the model using the specified parameters and save the trained model to the output directory.
To load and evaluate a pre-trained model, run the following command:
```
python main.py --evaluate --model_path /path/to/pretrained/model
```
This will load the pre-trained model and evaluate its performance on the test set.
```bash
python main.py --evaluate --model_path /path/to/pretrained/model
```
```bash
.
│  .gitignore
│  main.py # main script to run the project
│  playground.ipynb # a notebook to play around with the code
│  README.md
│  requirements.txt
│  test.py
│  train.sh
│
├─callbacks # Callbacks for training and logging
│      CometCallback.py
│      __init__.py
│
├─configs # Config files
│      config.yaml
│      ds_zero2_no_offload.json
│
├─data # Data module
│  │  DataLoader.py
│  │  Dataset.py
│  │  __init__.py
│  │
│  └─transformations
│          transforms.py
│          __init__.py
│
├─loggers # Logging module
│  │  logging_colors.py
│
├─losses # Losses module
│      loss.py
│      __init__.py
│
├─metrics # Metrics module
│      metric.py
│      __init__.py
│
├─models # Models module
│  │  modelutils.py
│  │  __init__.py
│  │
│  ├─HFModel # HuggingFace models
│  │      configuration_hfmodel.py
│  │      convert_hfmodel_original_pytorch_to_hf.py
│  │      feature_extraction_hfmodel.py # audio processing
│  │      image_processing_hfmodel.py # Image processing
│  │      modeling_hfmodel.py # Modeling
│  │      processing_hfmodel.py # mutimodal processing
│  │      tokenization_hfmodel.py # Tokenization
│  │      tokenization_hfmodel_fast.py
│  │      __init__.py
│  │
│  └─TorchModel # Torch models
│          modeling_torch.py
│          utils.py
│          __init__.py
│
├─onnx # ONNX module
│      converter2onnx.py
│
├─trainer # Trainer module 
│      acclerate.py
│      arguments.py
│      evaluater.py
│      inference.py
│      trainer.py
│      __init__.py
│
└─utils
        constants.py
        profiler.py # Profiling module
        utils.py

```
## Key Components

### Project Structure
- Maintain a clean and modular structure
- Define paths and constants in a central location (e.g. `Project.py`)
- Use `pathlib.Path` for cross-platform compatibility

### Data Processing
- Implement custom datasets by subclassing `torch.utils.data.Dataset`
- Define data transformations in `data/transformations/`
- Use `get_dataloaders()` to configure train/val/test loaders

### Modeling
- Define models in the `models/` directory
- Implement custom architectures or modify existing ones as needed

### Training and Evaluation  
- Utilize `main.py` for training/evaluation logic
- Leverage libraries like Accelerate for distributed training
- Implement useful callbacks:
  - Learning rate scheduling
  - Model checkpointing
  - Early stopping

### Logging and Experiment Tracking
- Use Python's `logging` module for consistent logging
- Integrate experiment tracking (e.g. Weights & Biases, MLflow)

### Utilities
- Implement helper functions for visualization, profiling, etc.
- Store in `utils/` directory

## Best Practices

- Avoid hardcoding paths - use a centralized configuration
- Modularize code for reusability and maintainability  
- Leverage existing libraries and tools when possible
- Document code and maintain a clear project structure
- Use version control and create reproducible experiments

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Modify `Project.py` with your paths/constants
4. Implement your custom dataset/model as needed
5. Run training: `python main.py`

This template addresses the common challenge of unstructured and hard-to-maintain code in data science projects. It provides a clean, modular structure that promotes scalability and shareability. The example project demonstrates image classification using a fine-tuned ResNet18 model on a Star Wars character dataset.

## Project Structure
The project is structured in a modular way, with separate folders for data processing, modeling, training and utilities. The `Project` class in `Project.py` stores paths and constants that are used throughout the codebase.

## Data Processing

Data processing is handled by the `get_dataloaders()` function in `data/datasets.py`. It takes in the dataset name and splits it into train/val/test sets using a predefined split ratio. Transforms can be applied to each set as needed.

## Modeling

Models are defined in `models/modeling_torch.py`. This file contains the implementation of a simple CNN architecture for image classification. You can modify or add your own models here.
## Training

Training is handled by the `train()` function in `train.py`. It takes in the model, dataloaders, and training parameters, and trains the model using the specified optimizer and loss function.

## Utilities

Utilities such as logging, saving, and loading models are handled by the `utils.py` file. This file contains functions for saving and loading models, as well as logging training progress.

## Example Usage

To train the model, run the following command:

```bash
python main.py
```
This will train the model using the specified parameters and save the trained model to the output directory.
To load and evaluate a pre-trained model, run the following command:
```
python main.py --evaluate --model_path /path/to/pretrained/model
```
This will load the pre-trained model and evaluate its performance on the test set.
```bash
python main.py --evaluate --model_path /path/to/pretrained/model
```
## Models

The models used in this project are defined in the `models.py` file. This file contains the implementation of the ResNet50 model, which is a popular architecture for image classification. You can modify or add your own models here.
## Training
Training is handled by the `train.py` file. This file contains the implementation of the training loop, which trains the model on the provided dataset using the specified parameters and saves the trained model to the output directory.





The models used in this project are defined in the `models.py` file. This file contains the implementation of the ResNet50 model, which is a popular architecture for image classification. You can modify or add your own models here.
## Training
Training is handled by the `train.py` file. This file contains the implementation of the training loop, which trains the model on the provided dataset using the specified parameters and saves the trained model to the output directory.




