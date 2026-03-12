# AI Art vs Human Art

Streamlit application and research workspace for detecting whether an image is AI-generated or a real photograph, with a second-stage classifier that identifies the likely image generator when AI content is detected.

Live app: `https://aivsreal-fbgngja2ai8fsxfhqdau96.streamlit.app/`

## Overview

This repository combines three parts of the project in one workspace:

- A production-style Streamlit interface in `app.py`
- Training and experimentation notebooks in `notebooks/`
- Local datasets and legacy model artifacts used during development

The current application uses Vision Transformer (ViT) models hosted on Hugging Face for inference. The app first performs binary classification (`AI Generated` vs `Real Image`) and, when the image is classified as AI-generated, runs a multiclass generator classifier.

## Key Features of the Streamlit App

- Single-image analysis from local upload or image URL
- Side-by-side comparison mode for two images
- Batch analysis for multiple uploaded files
- Generator family prediction for AI images
- Confidence scores and per-generator probability breakdown
- In-session history of recent analyses
- Responsive Streamlit UI with custom styling

## Model Pipeline

### Binary classifier

- Purpose: detect whether an image is AI-generated or a real photograph
- Hugging Face model: `gechen98/AI_image_classification`

### Generator classifier

- Purpose: identify the likely source model for AI-generated images
- Hugging Face model: `gechen98/AI_image_generator_classification`
- Supported labels:
  - `adm`
  - `biggan`
  - `glide`
  - `midjourney`
  - `sdv5`
  - `vqdm`
  - `wukong`

### Inference stack

- Base processor: `google/vit-base-patch16-224`
- Runtime libraries: PyTorch, Transformers, TorchVision, Streamlit

## Repository Contents

```text
AI_Art_vs_Human_Art/
|-- app.py                         # Current Streamlit app
|-- app_original.py                # Earlier TensorFlow/Streamlit prototype
|-- requirements.txt               # Python dependencies
|-- notebooks/
|   |-- baseline_efficientb3.ipynb
|   |-- binary_classification_vit.ipynb
|   `-- multiclass_classification_vit.ipynb
|-- merged_data/                   # Binary classification dataset
|   |-- train/
|   `-- val/
|-- dataset_multiclass/            # Generator classification dataset
|   |-- train/
|   `-- val/
|-- raw_data/                      # Source generator-specific image folders
|-- models/                        # Legacy local Keras model artifacts
|-- checkpoints/                   # Saved training checkpoints
|-- logs/                          # Training logs
`-- batch_predict/                 # Local sample files for manual batch runs
```

## Dataset Summary

### Binary dataset: `merged_data/`

- Training split:
  - `ai`: 13,999 images
  - `nature`: 14,001 images
- Validation split:
  - `ai`: 3,501 images
  - `nature`: 3,501 images

### Multiclass dataset: `dataset_multiclass/`

- Training split:
  - `adm`: 2,000
  - `biggan`: 2,000
  - `glide`: 1,999
  - `midjourney`: 2,000
  - `sdv5`: 2,000
  - `vqdm`: 2,000
  - `wukong`: 2,000
- Validation split:
  - `adm`: 500
  - `biggan`: 500
  - `glide`: 500
  - `midjourney`: 500
  - `sdv5`: 500
  - `vqdm`: 500
  - `wukong`: 500

## Installation

### Prerequisites

- Python 3.10 or newer recommended
- `pip`
- Internet access on first run to download Hugging Face model weights

### Setup

```bash
git clone git@github.com:Gechen989898/AI_Art_vs_Human_Art.git
cd AI_Art_vs_Human_Art

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Running the App

Start the Streamlit UI:

```bash
streamlit run app.py
```

Streamlit will print a local URL, typically:

```text
http://localhost:8501
```

## Live Deployment

The project is also deployed on Streamlit Community Cloud:

`https://aivsreal-fbgngja2ai8fsxfhqdau96.streamlit.app/`

## How the App Works

### Single mode

Upload an image file or provide an image URL. The app returns:

- Binary classification result
- Confidence score
- Generator prediction if the image is classified as AI-generated

### Compare mode

Upload two images and view both predictions side by side.

### Batch mode

Upload multiple image files and process them in one run, with a summary of AI vs real counts.

## Notebooks

- `notebooks/baseline_efficientb3.ipynb`: baseline CNN/EfficientNet experiments
- `notebooks/binary_classification_vit.ipynb`: binary ViT training and evaluation
- `notebooks/multiclass_classification_vit.ipynb`: multiclass ViT training and evaluation

These notebooks appear to represent the research and model-development side of the project, while `app.py` is the active inference interface.

## Legacy Artifacts

The repository still contains earlier TensorFlow/Keras components:

- `app_original.py`
- `models/basic_cnn.keras`
- `models/CNN_augmentaiton.keras`
- checkpoint files in `checkpoints/`

These are useful for project history and experimentation, but the current Streamlit app is centered on the Hugging Face ViT models.

## Known Limitations

- First-run startup depends on downloading transformer model weights.
- Prediction confidence is not a guarantee of correctness, especially for edited, compressed, or out-of-distribution images.
- There is no automated test suite in the current repository.

## Team

| Name | GitHub |
|------|--------|
| Gechen Ma | [@Gechen989898](https://github.com/Gechen989898) |
| Didier Peran Ganthier | [@didierganthier](https://github.com/didierganthier) |
| Alexis Kipiani | [@Alex-gitacc](https://github.com/Alex-gitacc) |
| Mame | [@kharitsama](https://github.com/kharitsama) |

## License

This repository is presented for educational and research purposes.
