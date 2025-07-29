from config import logger

def gemma_3_1b_instruct_vi_medical_lora(tokenizer, messages):
    """Prompt style for Gemma-3-1B-Instruct-Vi-Medical-LoRA: Simple user prompt with chat template"""
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

def gemma_3_1b_grpo_vi_medical_lora(tokenizer, messages):
    """Prompt style for Gemma-3-1B-GRPO-Vi-Medical-LoRA: System prompt with reasoning and answer format"""
    SYSTEM_PROMPT = """
Trả lời theo định dạng sau đây:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
    if not messages or not isinstance(messages, list) or not messages[0].get("role") == "user":
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Vui lòng cung cấp câu hỏi để tôi trả lời."}],
            add_generation_prompt=True,
            tokenize=False
        )
    
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i, msg in enumerate(messages):
        conversation.append(msg)
        if msg["role"] == "user" and (i == len(messages) - 1 or messages[i + 1]["role"] != "assistant"):
            conversation.append({"role": "assistant", "content": ""})
    
    return tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )

def llama_3_2_3b_instruct_vi_medical_lora(tokenizer, messages):
    """Prompt style for Llama-3.2-3B-Instruct-Vi-Medical-LoRA: Extract answer from context"""
    instruction = '''Bạn là một trợ lý hữu ích được giao nhiệm vụ trích xuất các đoạn văn trả lời câu hỏi của người dùng từ một ngữ cảnh cho trước. Xuất ra các đoạn văn chính xác từng từ một trả lời câu hỏi của người dùng. Không xuất ra bất kỳ văn bản nào khác ngoài các đoạn văn trong ngữ cảnh. Xuất ra lượng tối thiểu để trả lời câu hỏi, ví dụ chỉ 2-3 từ từ đoạn văn. Nếu không thể tìm thấy câu trả lời trong ngữ cảnh, xuất ra 'Ngữ cảnh không cung cấp câu trả lời...' '''
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": instruction}] + messages,
        add_generation_prompt=True,
        tokenize=False
    )

def llama_3_2_1b_instruct_vi_medical_lora(tokenizer, messages):
    """Prompt style for Llama-3.2-1B-Instruct-Vi-Medical-LoRA: Extract answer from context"""
    return llama_3_2_3b_instruct_vi_medical_lora(tokenizer, messages)

def llama_3_2_3b_reasoning_vi_medical_lora(tokenizer, question):
    """Prompt style for Llama-3.2-3B-Reasoning-Vi-Medical-LoRA: Reasoning prompt with think tag"""
    inference_prompt_style = """Bên dưới là một hướng dẫn mô tả một tác vụ, đi kèm với một thông tin đầu vào để cung cấp thêm ngữ cảnh.
Hãy viết một phản hồi để hoàn thành yêu cầu một cách phù hợp.
Trước khi trả lời, hãy suy nghĩ cẩn thận về câu hỏi và tạo một chuỗi suy nghĩ từng bước để đảm bảo phản hồi logic và chính xác.

### Instruction:
Bạn là một chuyên gia y tế có kiến thức chuyên sâu về lập luận lâm sàng, chẩn đoán và lập kế hoạch điều trị.
Vui lòng trả lời câu hỏi y tế sau đây.

### Question:
{}

### Response:
<think>
"""
    return inference_prompt_style.format(question) + tokenizer.eos_token

def qwen_3_0_6b_instruct_vi_medical_lora(tokenizer, messages):
    """Prompt style for Qwen-3-0.6B-Instruct-Vi-Medical-LoRA: Qwen-specific with enable_thinking=False"""
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False
    )

def qwen_3_0_6b_reasoning_vi_medical_lora(tokenizer, question):
    """Prompt style for Qwen-3-0.6B-Reasoning-Vi-Medical-LoRA: Same as Llama-3.2-3B-Reasoning-Vi-Medical-LoRA"""
    return llama_3_2_3b_reasoning_vi_medical_lora(tokenizer, question)

PROMPT_FUNCTIONS = {
    "Gemma-3-1B-Instruct-Vi-Medical-LoRA": gemma_3_1b_instruct_vi_medical_lora,
    "Gemma-3-1B-GRPO-Vi-Medical-LoRA": gemma_3_1b_grpo_vi_medical_lora,
    "Llama-3.2-3B-Instruct-Vi-Medical-LoRA": llama_3_2_3b_instruct_vi_medical_lora,
    "Llama-3.2-1B-Instruct-Vi-Medical-LoRA": llama_3_2_1b_instruct_vi_medical_lora,
    "Llama-3.2-3B-Reasoning-Vi-Medical-LoRA": llama_3_2_3b_reasoning_vi_medical_lora,
    "Qwen-3-0.6B-Instruct-Vi-Medical-LoRA": qwen_3_0_6b_instruct_vi_medical_lora,
    "Qwen-3-0.6B-Reasoning-Vi-Medical-LoRA": qwen_3_0_6b_reasoning_vi_medical_lora
}