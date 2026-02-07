# AI Art vs Human Art Classification

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3106/)

> A deep learning project to classify and distinguish AI-generated artwork from human-created artwork.

---

## 👥 Team Members

| Name | GitHub |
|------|--------|
| Gechen Ma | [@Gechen989898](https://github.com/Gechen989898) |
| Didier Peran Ganthier | [@didierganthier](https://github.com/didierganthier) |
| Alexis Kipiani | [@Alex-gitacc](https://github.com/Alex-gitacc) |
| Mame | [@kharitsama](https://github.com/kharitsama) |

---

## 📋 Project Overview

With the rise of AI image generation tools (DALL-E, Midjourney, Stable Diffusion), distinguishing between AI-generated and human-created art has become increasingly challenging. This project aims to build a robust classification model that can accurately identify the origin of artwork.

### Objectives
- Build and compare multiple deep learning architectures
- Achieve high accuracy in classifying AI vs Human art
- Deploy a functional API for real-time predictions
- Create an interactive demo interface

---

## 📊 Dataset

**Tiny GenImage** - A lightweight version of the GenImage dataset featuring images from multiple modern AI generators.

| Dataset | Size | Link |
|---------|------|------|
| **Tiny GenImage** | 8.36 GB | [Kaggle](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage) |

### Dataset Structure

The dataset is pre-organized with `train` and `val` splits, containing images labeled as `ai` (generated) vs `nature` (real).

**AI Generators Included:**
| Generator | Folder | Type |
|-----------|--------|------|
| BigGAN | `imagenet_ai_0419_biggan` | GAN-based |
| VQDM | `imagenet_ai_0419_vqdm` | Diffusion |
| Stable Diffusion v5 | `imagenet_ai_0424_sdv5` | Diffusion |
| Wukong | `imagenet_ai_0424_wukon` | Diffusion |
| ADM | `imagenet_ai_0508_adm` | Diffusion |
| GLIDE | `imagenet_glide` | Diffusion |
| Midjourney | `imagenet_midjourney` | Diffusion |

```
tiny_genimage/
├── imagenet_ai_0419_biggan/
│   ├── train/
│   │   ├── ai/
│   │   └── nature/
│   └── val/
│       ├── ai/
│       └── nature/
├── imagenet_ai_0419_vqdm/
│   └── ...
├── imagenet_ai_0424_sdv5/
│   └── ...
├── imagenet_ai_0424_wukon/
│   └── ...
├── imagenet_ai_0508_adm/
│   └── ...
├── imagenet_glide/
│   └── ...
└── imagenet_midjourney/
    └── ...
```

> 💡 **Note**: Having multiple AI generators allows us to test model generalization across different generation techniques (GAN vs Diffusion models).

---

## 🏗️ Project Architecture

```
AI_Art_vs_Human_Art/
├── README.md
├── Makefile
├── requirements.txt
├── setup.py
├── Dockerfile
├── docker-compose.yml
├── .env.sample
├── .gitignore
├── raw_data/
│   └── tiny_genimage/
│       ├── imagenet_ai_0419_biggan/
│       ├── imagenet_ai_0419_vqdm/
│       ├── imagenet_ai_0424_sdv5/
│       ├── imagenet_ai_0424_wukon/
│       ├── imagenet_ai_0508_adm/
│       ├── imagenet_glide/
│       └── imagenet_midjourney/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_comparison.ipynb
├── ai_art_classifier/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── efficientnet.py
│   │   ├── xception.py
│   │   └── vision_transformer.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── callbacks.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── api/
│       ├── __init__.py
│       ├── fast_api.py
│       └── database.py
├── streamlit_app/
│   └── app.py
├── models/
│   └── (saved model weights: .h5, ONNX, TFLite)
└── tests/
    └── (unit & integration tests)
```

---

## 🧠 Models

We will train and compare multiple models using transfer learning:

| Model | Expected Accuracy | Notes |
|-------|-------------------|-------|
| **EfficientNetB3** (Baseline) | 78-82% | Good balance of speed and accuracy |
| **EfficientNetB4** | 80-84% | Improved accuracy |
| **Xception** | 85-90% | Gold standard for AI/deepfake detection - excels at texture & noise patterns |
| **Vision Transformer (ViT-B)** | 83-87% | Transformer-based approach |

> 💡 **Why Xception?** Xception uses "depthwise separable convolutions" which are exceptionally good at detecting *texture* and *noise* patterns rather than just shapes. It's the architecture behind most successful DeepFake detectors and is particularly effective at spotting the "glossy" or "smooth" texture that AI models like Midjourney often produce.

Final production model target: **~87-92% accuracy**

---

## 📅 Project Timeline (6 Weeks)

### Week 1: Data Preparation & Model Baseline

**Monday-Wednesday: Data Setup**
- Download Tiny GenImage dataset from Kaggle
- Explore dataset structure and document findings
- Create 70/15/15 train/val/test split with stratification
- Analyze class distribution and data quality
- Set up data pipeline with augmentation
- Initialize GitHub repository with project structure
- Set up MLflow experiment tracking
- Configure Python environment and requirements.txt

**Thursday-Friday: Train Baseline Model**
- Create and train EfficientNetB3 baseline model
- Expected accuracy: 78-82%
- Evaluate on validation set
- Save model and document results
- Create initial model comparison notebook

> 🎯 **Friday EOD Deliverable:** Baseline model with ~80% accuracy

### Week 2: Model Comparison & Selection

**Monday-Wednesday: Train Additional Models (Parallel)**
- Train Vision Transformer (ViT-B) model • Expected accuracy: 83-87%
- Train EfficientNetB4 model • Expected accuracy: 80-84%
- Document training logs and hyperparameters for each model
- Track all experiments in MLflow

**Thursday: Model Evaluation & Comparison**
- Evaluate all 3 models on test set
- Calculate accuracy, precision, recall, F1-score, ROC-AUC
- Create confusion matrices for each model
- Generate performance comparison table
- Create ROC curves overlay visualization

**Friday: Model Selection & Decision**
- Decide on best single model or ensemble approach
- Document decision-making rationale
- Save selected model(s)

> 🎯 **Friday EOD Deliverable:** Model comparison report and selected model(s)

### Week 3: Hyperparameter Tuning & Final Model

**Monday-Wednesday: Systematic Hyperparameter Tuning**
- Test different learning rates (1e-5, 5e-5, 1e-4, 5e-4, 1e-3)
- Test different batch sizes (16, 32, 64)
- Test different training schedules and optimizers
- Track all configurations in MLflow
- Identify best hyperparameter combination

**Thursday: Create Ensemble Model (Optional)**
- Combine best performing models
- Implement ensemble voting/averaging
- Evaluate ensemble performance
- Expected accuracy: 85-90%

**Friday: Finalize Model**
- Train final model with best hyperparameters
- Evaluate on test set
- Save final model in multiple formats (.h5, ONNX, TFLite)
- Document all hyperparameters and training details
- Create final model report

> 🎯 **Friday EOD Deliverable:** Production-ready model with ~87-90% accuracy, saved in multiple formats

### Week 4: REST API & Web Interface

**Monday-Tuesday: Build REST API (FastAPI)**
- Design API endpoints and request/response schemas
- Implement `POST /predict` endpoint for single image
- Implement `POST /batch-predict` endpoint for multiple images
- Implement `GET /health` endpoint for health checks
- Add comprehensive error handling
- Create API documentation (auto-generated with Swagger/OpenAPI)
- Test all API endpoints locally

**Wednesday-Thursday: Build Web Interface (Streamlit)**
- Create Streamlit app with page layout
- Implement "Single Image" mode with upload and predictions
- Implement "Batch Upload" mode for multiple images
- Implement "From URL" mode for image URLs
- Add confidence visualizations and gauges
- Add results export to CSV
- Style interface with custom CSS

**Friday: Testing & Integration**
- Test API endpoints with curl and Python requests
- Test web interface with various image types
- Verify predictions match between API and web interface
- Test error handling for invalid inputs
- Document API usage examples

> 🎯 **Friday EOD Deliverable:** Working API + web interface, tested and ready

### Week 5: Database, Docker & Monitoring

**Monday-Tuesday: Database Setup**
- Set up PostgreSQL database
- Create Prediction table schema
- Implement database models with SQLAlchemy
- Add prediction saving to API endpoint
- Create database migrations
- Test database operations

**Wednesday: Docker Containerization**
- Create Dockerfile for API service
- Create docker-compose.yml for multi-container setup
- Build and test Docker images locally
- Verify API and web interface work in containers
- Set up environment variables and secrets management

**Thursday: Logging & Monitoring**
- Implement structured logging throughout application
- Set up Prometheus metrics (prediction counter, latency, accuracy)
- Create logging configuration with rotating file handlers
- Add health check endpoint with database connectivity check
- Set up log file rotation

**Friday: Testing**
- Write unit tests for API endpoints
- Write integration tests with database
- Write tests for data processing functions
- Achieve >80% code coverage
- Run full test suite

> 🎯 **Friday EOD Deliverable:** Dockerized system with database, logging, monitoring, and comprehensive tests

### Week 6: Cloud Deployment & Documentation

**Monday-Tuesday: Cloud Deployment**
- Choose cloud platform (AWS, GCP, or Azure)
- Set up cloud infrastructure (container registry, compute resources)
- Push Docker images to container registry
- Deploy API service to cloud platform
- Configure load balancing and auto-scaling
- Set up SSL/TLS certificates
- Test deployed system with real traffic

**Wednesday: Final Testing & Optimization**
- Test API performance in production
- Test web interface against production API
- Benchmark inference time and throughput
- Monitor logs and metrics in production
- Optimize performance if needed
- Document any deployment-specific configurations

**Thursday-Friday: Comprehensive Documentation**
- Write detailed README with features and quick start
- Document API endpoints and usage examples
- Create deployment guides for AWS, GCP, and Azure
- Write developer setup guide
- Create troubleshooting guide
- Document model performance metrics
- Create architecture diagram
- Write future maintenance guide

**Friday: Final Review & Presentation**
- Review all code for quality and consistency
- Ensure all tests pass
- Verify documentation is complete and accurate
- Test entire system end-to-end
- Prepare final project summary
- Create project demo and walkthrough

> 🎯 **Friday EOD Deliverable:** Production-deployed system with complete documentation

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10.6
- pyenv (recommended)

### Installation

```bash
# Clone the repository
git clone git@github.com:Gechen989898/AI_Art_vs_Human_Art.git
cd AI_Art_vs_Human_Art

# Create and activate virtual environment
pyenv virtualenv 3.10.6 AI_Art_vs_Human_Art
pyenv activate AI_Art_vs_Human_Art

# Install dependencies
pip install -r requirements.txt
```

### Download Data

```bash
# Using Kaggle CLI
kaggle datasets download -d yangsangtai/tiny-genimage
unzip tiny-genimage.zip -d raw_data/
```

### Training

```bash
# Train EfficientNetB3 baseline
make train_efficientnet_b3

# Train all models
make train_all

# Run hyperparameter tuning
make tune
```

### Running the API

```bash
# Start FastAPI server
make run_api
```

---

## 📈 Expected Results

| Model | Target Accuracy | Training Time |
|-------|-----------------|---------------|
| EfficientNetB3 (Baseline) | 78-82% | ~30 min |
| EfficientNetB4 | 80-84% | ~45 min |
| Xception | 85-90% | ~45 min |
| Vision Transformer (ViT-B) | 83-87% | ~1 hour |
| Final Model (Tuned/Ensemble) | 87-92% | ~1-2 hours |

---

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow / PyTorch
- **Models**: EfficientNet, Xception, Vision Transformer (ViT)
- **Data Processing**: NumPy, Pandas, OpenCV
- **Visualization**: Matplotlib, Seaborn
- **API**: FastAPI
- **Database**: PostgreSQL, SQLAlchemy
- **Deployment**: Docker, docker-compose, AWS/GCP/Azure
- **Demo**: Streamlit
- **Experiment Tracking**: MLflow
- **Monitoring**: Prometheus
- **Testing**: pytest

---

## 📚 Resources

- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Xception Paper - Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

---

## 📝 License

This project is for educational purposes.

---

## 🤝 Contributing

1. Create a feature branch from `master`
2. Make your changes
3. Submit a Pull Request
4. Request review from at least one team member

**Branch naming convention**: `feature/<your-name>/<feature-description>`

---

*Project started: February 2026*
