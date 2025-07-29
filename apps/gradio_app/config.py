import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LoRA configurations
LORA_CONFIGS = {
    "Gemma-3-1B-Instruct-Vi-Medical-LoRA": {
        "base_model": "unsloth/gemma-3-1b-it",
        "lora_adapter": "danhtran2mind/Gemma-3-1B-Instruct-Vi-Medical-LoRA"
    },
    "Gemma-3-1B-GRPO-Vi-Medical-LoRA": {
        "base_model": "unsloth/gemma-3-1b-it",
        "lora_adapter": "danhtran2mind/Gemma-3-1B-GRPO-Vi-Medical-LoRA"
    },
    "Llama-3.2-3B-Instruct-Vi-Medical-LoRA": {
        "base_model": "unsloth/Llama-3.2-3B-Instruct",
        "lora_adapter": "danhtran2mind/Llama-3.2-3B-Instruct-Vi-Medical-LoRA"
    },
    "Llama-3.2-1B-Instruct-Vi-Medical-LoRA": {
        "base_model": "unsloth/Llama-3.2-1B-Instruct",
        "lora_adapter": "danhtran2mind/Llama-3.2-1B-Instruct-Vi-Medical-LoRA"
    },
    "Llama-3.2-3B-Reasoning-Vi-Medical-LoRA": {
        "base_model": "unsloth/Llama-3.2-3B-Instruct",
        "lora_adapter": "danhtran2mind/Llama-3.2-3B-Reasoning-Vi-Medical-LoRA"
    },
    "Qwen-3-0.6B-Instruct-Vi-Medical-LoRA": {
        "base_model": "Qwen/Qwen3-0.6B",
        "lora_adapter": "danhtran2mind/Qwen-3-0.6B-Instruct-Vi-Medical-LoRA"
    },
    "Qwen-3-0.6B-Reasoning-Vi-Medical-LoRA": {
        "base_model": "Qwen/Qwen3-0.6B",
        "lora_adapter": "danhtran2mind/Qwen-3-0.6B-Reasoning-Vi-Medical-LoRA"
    }
}

# LORA_CONFIGS = {
#     "Gemma-3-1B-Instruct-Vi-Medical-LoRA": {
#         "base_model": "google/gemma-3-1b-it",
#         "lora_adapter": "danhtran2mind/Gemma-3-1B-Instruct-Vi-Medical-LoRA"
#     },
#     "Gemma-3-1B-GRPO-Vi-Medical-LoRA": {
#         "base_model": "google/gemma-3-1b-it",
#         "lora_adapter": "danhtran2mind/Gemma-3-1B-GRPO-Vi-Medical-LoRA"
#     },
#     "Llama-3.2-3B-Instruct-Vi-Medical-LoRA": {
#         "base_model": "meta-llama/Llama-3.2-3B-Instruct",
#         "lora_adapter": "danhtran2mind/Llama-3.2-3B-Instruct-Vi-Medical-LoRA"
#     },
#     "Llama-3.2-1B-Instruct-Vi-Medical-LoRA": {
#         "base_model": "meta-llama/Llama-3.2-1B-Instruct",
#         "lora_adapter": "danhtran2mind/Llama-3.2-1B-Instruct-Vi-Medical-LoRA"
#     },
#     "Llama-3.2-3B-Reasoning-Vi-Medical-LoRA": {
#         "base_model": "meta-llama/Llama-3.2-3B-Instruct",
#         "lora_adapter": "danhtran2mind/Llama-3.2-3B-Reasoning-Vi-Medical-LoRA"
#     },
#     "Qwen-3-0.6B-Instruct-Vi-Medical-LoRA": {
#         "base_model": "Qwen/Qwen3-0.6B",
#         "lora_adapter": "danhtran2mind/Qwen-3-0.6B-Instruct-Vi-Medical-LoRA"
#     },
#     "Qwen-3-0.6B-Reasoning-Vi-Medical-LoRA": {
#         "base_model": "Qwen/Qwen3-0.6B",
#         "lora_adapter": "danhtran2mind/Qwen-3-0.6B-Reasoning-Vi-Medical-LoRA"
#     }
# }

# Model settings
MAX_INPUT_TOKEN_LENGTH = 4096
DEFAULT_MAX_NEW_TOKENS = 512
MAX_MAX_NEW_TOKENS = 2048

MODEL_IDS = list(LORA_CONFIGS.keys())