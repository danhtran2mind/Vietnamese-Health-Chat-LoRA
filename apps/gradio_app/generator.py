import copy
import re
import torch
from threading import Thread
from transformers import TextIteratorStreamer
from config import logger, MAX_INPUT_TOKEN_LENGTH
from prompts import PROMPT_FUNCTIONS
from response_parser import ParserState, parse_response, format_response, remove_tags
from utils import merge_conversation

def generate_response(model_handler, history, temperature, top_p, top_k, max_tokens, seed, active_gen, model_id, auto_clear):
    raw_history = copy.deepcopy(history)
    
    # Clean history by removing tags from assistant responses
    history = [[item[0], remove_tags(item[1]) if item[1] else None] for item in history]
    
    try:
        # Validate history
        if not isinstance(history, list) or not history:
            logger.error("History is empty or not a list")
            history = [[None, "Error: Conversation history is empty or invalid"]]
            yield history
            return
        # Validate last history entry
        if not isinstance(history[-1], (list, tuple)) or len(history[-1]) < 1 or not history[-1][0]:
            logger.error("Last history entry is invalid or missing user message")
            history = raw_history
            history[-1][1] = "Error: No valid user message provided"
            yield history
            return
            
        # Load model if necessary
        if model_handler.model is None or model_handler.tokenizer is None or model_id != model_handler.current_model_id:
            status, _ = model_handler.load_model(model_id, history)
            if "Error" in status:
                logger.error(status)
                history[-1][1] = status
                yield history
                return
        
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))

        # Validate prompt function
        if model_id not in PROMPT_FUNCTIONS:
            logger.error(f"No prompt function defined for model_id: {model_id}")
            history[-1][1] = f"Error: No prompt function defined for model {model_id}"
            yield history
            return
        prompt_fn = PROMPT_FUNCTIONS[model_id]

        # Handle specific model prompt formatting
        if model_id in [
            "Llama-3.2-3B-Reasoning-Vi-Medical-LoRA",
            "Qwen-3-0.6B-Reasoning-Vi-Medical-LoRA"
        ]:
            if auto_clear:                    
                text = prompt_fn(model_handler.tokenizer, history[-1][0])
            else:
                text = prompt_fn(model_handler.tokenizer, merge_conversation(history))
                
            inputs = model_handler.tokenizer(
                [text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_INPUT_TOKEN_LENGTH
            )
        else:
            # Build conversation for other models
            conversation = []
            for msg in history:
                if msg[0]:
                    conversation.append({"role": "user", "content": msg[0]})
                if msg[1]:
                    clean_text = ' '.join(line for line in msg[1].split('\n') if not line.startswith('✅ Thought for')).strip()
                    conversation.append({"role": "assistant", "content": clean_text})
                elif msg[0] and not msg[1]:
                    conversation.append({"role": "assistant", "content": ""})
            
            # Ensure at least one user message
            if not any(msg["role"] == "user" for msg in conversation):
                logger.error("No valid user messages in conversation history")
                history = raw_history
                history[-1][1] = "Error: No valid user messages in conversation history"
                yield history
                return
            
            # Apply auto_clear logic
            if auto_clear:
                # Keep only the last user message and add an empty assistant response
                user_msgs = [msg for msg in conversation if msg["role"] == "user"]
                if user_msgs:
                    conversation = [{"role": "user", "content": user_msgs[-1]["content"]}, {"role": "assistant", "content": ""}]
                else:
                    logger.error("No user messages found after filtering")
                    history = raw_history
                    history[-1][1] = "Error: No user messages found in conversation history"
                    yield history
                    return
            else:
                # Ensure the conversation ends with an assistant placeholder if the last message is from user
                if conversation and conversation[-1]["role"] == "user":
                    conversation.append({"role": "assistant", "content": ""})

            text = prompt_fn(model_handler.tokenizer, conversation)
            tokenizer_kwargs = {
                "return_tensors": "pt",
                "padding": True,
                "truncation": True,
                "max_length": MAX_INPUT_TOKEN_LENGTH
            }

            inputs = model_handler.tokenizer(text, **tokenizer_kwargs)

        if inputs is None or "input_ids" not in inputs:
            logger.error("Tokenizer returned invalid or None output")
            history = raw_history
            history[-1][1] = "Error: Failed to tokenize input"
            yield history
            return

        input_ids = inputs["input_ids"].to(model_handler.model.device)
        attention_mask = inputs.get("attention_mask").to(model_handler.model.device) if "attention_mask" in inputs else None
        
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "pad_token_id": model_handler.tokenizer.pad_token_id,
            "eos_token_id": model_handler.tokenizer.eos_token_id,
            "use_cache": True,
            "cache_implementation": "dynamic",
        }

        streamer = TextIteratorStreamer(model_handler.tokenizer, timeout=360.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs["streamer"] = streamer

        def run_generation():
            try:
                model_handler.model.generate(**generate_kwargs)
            except Exception as e:
                logger.error(f"Generation failed: {str(e)}")
                raise

        thread = Thread(target=run_generation)
        thread.start()

        state = ParserState()
        if model_id in [
            "Llama-3.2-3B-Reasoning-Vi-Medical-LoRA",
            "Qwen-3-0.6B-Reasoning-Vi-Medical-LoRA"
        ]:
            full_response = "<think>"
        else:
            full_response = ""
        
        for text in streamer:
            if not active_gen[0]:
                logger.info("Generation stopped by user")
                break
                
            if text:
                logger.debug(f"Raw streamer output: {text}")
                text = re.sub(r'<\|\w+\|>', '', text)
                full_response += text
                state, elapsed = parse_response(full_response, state)
                
                collapsible, answer_part = format_response(state, elapsed)
                history = raw_history
                history[-1][1] = "\n\n".join(collapsible + [answer_part])
                yield history
            else:
                logger.debug("Streamer returned empty text")
        
        thread.join()
        thread = None
        state, elapsed = parse_response(full_response, state)
        collapsible, answer_part = format_response(state, elapsed)
        history = raw_history
        history[-1][1] = "\n\n".join(collapsible + [answer_part])
        
        if not full_response:
            logger.warning("No response generated by model")
            history[-1][1] = "No response generated. Please try again or select a different model."
            
        yield history
        
    except Exception as e:
        logger.error(f"Error in generate: {str(e)}")
        history = raw_history
        if not history or not isinstance(history, list):
            history = [[None, f"Error: {str(e)}. Please try again or select a different model."]]
        else:
            history[-1][1] = f"Error: {str(e)}. Please try again or select a different model."
            
        yield history
    finally:
        active_gen[0] = False