# Vietnamese-Health-Chat-LoRA ⚕️

[![GitHub Stars](https://img.shields.io/github/stars/danhtran2mind/Vietnamese-Health-Chat-LoRA?style=social&label=Repo%20Stars)](https://github.com/danhtran2mind/Vietnamese-Health-Chat-LoRA/stargazers)
![Repo Views](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fdanhtran2mind%2FVietnamese-Health-Chat-LoRA&label=Repo+Views&icon=github&color=%236f42c1&message=&style=social&tz=UTC)

[![huggingface-hub](https://img.shields.io/badge/huggingface--hub-blue.svg?logo=huggingface)](https://huggingface.co/docs/hub)
[![torch](https://img.shields.io/badge/torch-blue.svg?logo=pytorch)](https://pytorch.org/)
[![transformers](https://img.shields.io/badge/transformers-blue.svg?logo=huggingface)](https://huggingface.co/docs/transformers)
[![gradio](https://img.shields.io/badge/gradio-blue.svg?logo=gradio)](https://gradio.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-trl-blue?style=flat&logo=github)](https://github.com/huggingface/trl)
[![HuggingFace Hub](https://img.shields.io/badge/HuggingFace-Unsloth%20AI-13b989?style=flat&logo=huggingface)](https://huggingface.co/unsloth)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Introduction
Vietnamese-Health-Chat-LoRA is an advanced conversational AI project designed for healthcare queries in Vietnamese. Utilizing LoRA fine-tuning, it enhances leading language models to deliver precise, contextually relevant medical responses. 🚀

## Key Features
- 🩺 **Specialized Medical Responses**: Tailored for accurate healthcare conversations in Vietnamese.
- 🤖 **Diverse Base Models**: Supports Gemma, Llama, and Qwen for versatile performance.
- 📱 **User-Friendly Interface**: Gradio-powered GUI for effortless interaction.
- ⚡ **Optimized Training**: Efficient LoRA configurations for resource-friendly fine-tuning.

## Notebooks
Dive into the training and inference processes:

- **gemma-3-1b-grpo-vi-medical-lora**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danhtran2mind/Vietnamese-Health-Chat-LoRA/blob/main/notebooks/gemma-3-1b-grpo-vi-medical-lora.ipynb)  
- **gemma-3-1b-it-vi-medical-lora**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danhtran2mind/Vietnamese-Health-Chat-LoRA/blob/main/notebooks/gemma-3-1b-it-vi-medical-lora.ipynb)  
- **llama-3-2-1b-it-vi-medical-lora**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danhtran2mind/Vietnamese-Health-Chat-LoRA/blob/main/notebooks/llama-3-2-1b-it-vi-medical-lora.ipynb)  
- **llama-3-2-3b-it-vi-medical-lora**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danhtran2mind/Vietnamese-Health-Chat-LoRA/blob/main/notebooks/llama-3-2-3b-it-vi-medical-lora.ipynb)  
- **llama-3-2-3b-reasoning-vi-medical-lora-training**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danhtran2mind/Vietnamese-Health-Chat-LoRA/blob/main/notebooks/llama-3-2-3b-reasoning-vi-medical-lora-training.ipynb)  
- **qwen-3-0-6b-it-vi-medical-lora**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danhtran2mind/Vietnamese-Health-Chat-LoRA/blob/main/notebooks/qwen-3-0-6b-it-vi-medical-lora.ipynb)  
- **qwen-3-0-6b-reasoning-vi-medical-lora**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danhtran2mind/Vietnamese-Health-Chat-LoRA/blob/main/notebooks/qwen-3-0-6b-reasoning-vi-medical-lora.ipynb)  

## Dataset
Trained on the **ViMedAQA** dataset:  
[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-tmnam20%2FViMedAQA-yellow?style=flat&logo=huggingface)](https://huggingface.co/datasets/tmnam20/ViMedAQA)

## Base Model
- [Gemma-3-1B-Instruct](https://huggingface.co/google/gemma-3-1b-it)
- [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [Qwen-3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

## Demonstration
Discover the capabilities of Vietnamese-Health-Chat-LoRA:  
- **HuggingFace Space**: [![HuggingFace Space Demo](https://img.shields.io/badge/HuggingFace-danhtran2mind%2FText2Video--Ghibli--style-yellow?style=flat&logo=huggingface)](https://huggingface.co/spaces/danhtran2mind/Text2Video-Ghibli-style)  
- **Demo GUI**:  
  ![Gradio Demo](./assets/gradio_app_demo.jpg)

Run locally:  
```bash
python apps/gradio_app.py
```

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/danhtran2mind/Vietnamese-Health-Chat-LoRA
cd Vietnamese-Health-Chat-LoRA
```

### Step 2: Install Dependencies
```bash
pip install -r requirements/requirements.txt
```

## Usage
Start the Gradio app:  
```bash
python apps/gradio_app.py
```

## Training
Check the [Notebooks](#notebooks) section for training details of each LoRA model.

## Environment
- **Python**: 3.10+  
- **Libraries**: See [requirements_compatible.txt](requirements/requirements_compatible.txt) for versions.

## Contact
For inquiries or issues, visit the [GitHub Issues tab](https://github.com/danhtran2mind/Vietnamese-Health-Chat-LoRA/issues).