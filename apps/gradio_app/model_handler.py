# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel
# import gc
# from config import logger, LORA_CONFIGS

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login
import gc
from config import logger, LORA_CONFIGS

# Check for Hugging Face API token
if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
    logger.error("Hugging Face API token is not set. Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")
    raise ValueError("Hugging Face API token is not set. Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")

# Set the Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# Initialize API
login(os.environ.get("HUGGINGFACEHUB_API_TOKEN"))

class ModelHandler:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_id = None

    def load_model(self, model_id, chatbot_state):
        """Load the model, tokenizer, and apply LoRA adapter for the given model ID."""
        try:
            logger.info(f"Loading model: {model_id}")
            print(f"Changing to model: {model_id}")
            self.clear_model()

            if model_id not in LORA_CONFIGS:
                raise ValueError(f"Invalid model ID: {model_id}")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            base_model_name = LORA_CONFIGS[model_id]["base_model"]
            lora_adapter_name = LORA_CONFIGS[model_id]["lora_adapter"]

            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True
            )
            self.tokenizer.use_default_system_prompt = False

            if self.tokenizer.pad_token is None or self.tokenizer.pad_token == self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.unk_token or "<pad>"
                logger.info(f"Set pad_token to {self.tokenizer.pad_token}")

            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True
            )

            self.model = PeftModel.from_pretrained(self.model, lora_adapter_name)
            self.model.eval()
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

            self.current_model_id = model_id
            chatbot_state = []
            return f"Successfully loaded model: {model_id} with LoRA adapter {lora_adapter_name}", chatbot_state
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {str(e)}")
            return f"Error: Failed to load model {model_id}: {str(e)}", chatbot_state

    def clear_model(self):
        """Clear the current model and tokenizer from memory."""
        if self.model is not None:
            print("Clearing previous model from RAM/VRAM...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print("Memory cleared successfully.")