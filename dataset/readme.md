# Dataset Loading and Processing Module

This module provides utilities for loading, processing, and creating PyTorch `DataLoader` objects for training deep learning models. It supports loading data from the following sources:

- **Custom Local Datasets:** Using the `CustomTorchDataset` class.
- **Standard Hugging Face Datasets:** Using the `CustomHuggingFaceDataset` class.
- **Custom Hugging Face Datasets:** Using the `CustomHuggingFaceDataset` class with your own dataset scripts.
- **Datasets from Image/Label Paths:** Using the `create_from_paths` static method of the `CustomHuggingFaceDataset` class.

## Key Features:

- **Modularity:** Clear separation of dataset loading and data loader creation logic.
- **Flexibility:** Supports various data sources and data loading scenarios.
- **Distributed Training Compatibility:** Seamless integration with PyTorch's `DistributedSampler` for efficient multi-GPU training.
- **Data Transformations:** Easily apply transformations to your datasets.

## Module Structure:

- `dataset_builder.py`: Contains the dataset classes for loading and processing data:
    - `CustomTorchDataset`: For loading custom image data organized in a list of dictionaries.
    - `CustomHuggingFaceDataset`: A wrapper class for loading standard and custom Hugging Face Datasets. It provides methods for:
        - Loading standard datasets by name (e.g., 'cifar10').
        - Loading custom datasets from your own scripts.
        - Creating datasets directly from lists of image and label paths.
- `data_loader.py`: Provides the `get_dataloaders` function, which handles:
    - Dataset splitting (train/validation/test).
    - `DistributedSampler` setup (if `distributed=True`).
    - Creation of PyTorch `DataLoader` objects for each split.

## How to Use:

1. **Data Preparation:**
   - **For `CustomTorchDataset`:** Organize your data as a list of dictionaries, where each dictionary represents a data sample and has keys corresponding to the data you want to load (e.g., 'image', 'label').
   - **For `CustomHuggingFaceDataset` (using a custom dataset script):** Ensure your custom dataset script is correctly set up following the Hugging Face Datasets guidelines. See the `dataset/hf_dataset_sample` directory for an example.
   - **For `CustomHuggingFaceDataset` (creating from paths):** Prepare your lists of image paths and corresponding label paths.

2. **Dataset Instantiation:**
   - Import the necessary classes from `dataset_builder`:
     ```python
     from dataset import dataset_builder
     ```
   - Create an instance of the appropriate dataset class:
     - **For custom local datasets:**
       ```python
       data = [{"image": "path/to/image1.jpg", "label": 0}, ...]  # Your data
       dataset = dataset_builder.CustomTorchDataset(data, transforms=your_transforms)
       ```
     - **For standard Hugging Face datasets:**
       ```python
       dataset = dataset_builder.CustomHuggingFaceDataset(dataset_name='cifar10', transforms=your_transforms)
       ```
     - **For your custom Hugging Face datasets:**
       ```python
       dataset = dataset_builder.CustomHuggingFaceDataset(
           dataset_name='your_custom_dataset_name',
           dataset_script_path="path/to/your/dataset_script.py",
           transforms=your_transforms
       )
       ```
     - **For creating a dataset from image and label paths:**
       ```python
       dataset = dataset_builder.CustomHuggingFaceDataset.create_from_paths(
           image_paths=["path/to/image1.jpg", ...],
           label_paths=["path/to/label1.png", ...],
           transforms=your_transforms
       )
       ```
   - **Important:** Register your custom Hugging Face dataset once at the beginning of your script:
     ```python
     import datasets
     datasets.load_dataset(
         "path/to/your/dataset_script.py",
         name="default",  # Or the appropriate config name
         trust_remote_code=True 
     )
     ```

3. **Data Loading (for `CustomHuggingFaceDataset`):**
   - Load the desired data splits (e.g., 'train', 'validation', 'test'):
     ```python
     train_dataset = dataset.load('train')
     val_dataset = dataset.load('validation')
     test_dataset = dataset.load('test')
     ```

4. **DataLoader Creation:**
   - Import the `get_dataloaders` function from `data_loader`:
     ```python
     from dataset import data_loader
     ```
   - Create DataLoaders for each split:
     ```python
     train_loader, val_loader, test_loader = data_loader.get_dataloaders(
         train_dataset, 
         batch_size=32, 
         num_workers=4, 
         distributed=True  # Set to True for distributed training
     )
     ```

Now you have your `DataLoader` objects ready for training your PyTorch models!
