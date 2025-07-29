import os
import sys
import gradio as gr

sys.path.append(os.path.join(os.path.dirname(__file__), 'gradio_app'))

from config import logger, MODEL_IDS
from model_handler import ModelHandler
from generator import generate_response

DESCRIPTION = '''
<div class="intro-container">
    <h1><span class="intro-icon">⚕️</span> Vietnamese Health Chat LoRA</h1>
    <h2>Discover advanced models fine-tuned with LoRA for precise medical reasoning in Vietnamese</h2>
    <div class="intro-disclaimer">
        <span class="intro-icon">ℹ️</span> Important Notice:
            <span class="intro-purpose">
                For research purposes only. AI responses may have limitations due to development, datasets, or architecture.
            </span>
        <br>    
        <span class="intro-alert emphasis">
            🚨Always consult a certified medical professional for personalized health advice🩺
        </span>
    </div>
</div>
'''

# Load local CSS file
CSS = open("gradio_app/static/styles.css").read()

def user(message, history):
    if not isinstance(history, list):
        history = []
    return "", history + [[message, None]]

def create_ui(model_handler):
    with gr.Blocks(css=CSS, theme=gr.themes.Default(), elem_classes="app-container") as demo:
        gr.HTML(DESCRIPTION)
        gr.HTML('<script src="file=gradio_app/static/script.js"></script>')
        active_gen = gr.State([False])
        model_handler_state = gr.State(model_handler)
        
        chatbot = gr.Chatbot(
            elem_id="output-container",
            height=250,
            show_label=False,
            render_markdown=True
        )

        with gr.Row():
            msg = gr.Textbox(
                label="Your Medical Query",
                placeholder="Ask about symptoms, treatments, or medical advice in Vietnamese...",
                container=False,
                scale=4
            )
            submit_btn = gr.Button(
                value="Send",
                variant='primary',
                elem_classes="chat-send-button",
                scale=1
            )
        
        with gr.Row():
            clear_btn = gr.Button(
                value="Clear",
                variant='secondary',
                elem_classes="clear-button"
            )
            stop_btn = gr.Button(
                value="Stop",
                variant='stop',
                elem_classes="stop-button"
            )
        
        with gr.Row():
            with gr.Column(scale=1):
                auto_clear = gr.Checkbox(
                    label="Auto-Clear Chat History",
                    value=True,
                    info="Automatically resets internal conversation history after each response, keeping displayed messages intact for a smooth experience.",
                    elem_classes="enhanced-checkbox"
                )
            with gr.Column(scale=1):
                with gr.Blocks():
                    model_dropdown = gr.Dropdown(
                        choices=MODEL_IDS,
                        value=MODEL_IDS[0],
                        label="Select Model",
                        interactive=True
                    )
                    model_load_output = gr.Textbox(label="Model Load Status")
        
        with gr.Column(scale=2):
            with gr.Accordion("Advanced Parameters", open=False):
                temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.7, label="Temperature")
                top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, label="Top-p")
                top_k = gr.Slider(minimum=1, maximum=100, value=64, step=1, label="Top-k")
                max_tokens = gr.Slider(minimum=128, maximum=4084, value=512, step=32, label="Max Tokens")
                seed = gr.Slider(minimum=0, maximum=2**32, value=123456, step=1, label="Random Seed")

        gr.Examples(
            examples=[
                ["Khi nghi ngờ bị loét dạ dày tá tràng nên đến khoa nào tại bệnh viện để thăm khám?"],
                ["Triệu chứng của loét dạ dày tá tràng là gì?"],
                ["Tôi bị mất ngủ, tôi phải làm gì?"],
                ["Tôi bị trĩ, tôi có nên mổ không?"]
            ],
            inputs=msg,
            label="Sample Medical Queries"
        )
        
        model_dropdown.change(
            fn=model_handler.load_model,
            inputs=[model_dropdown, chatbot],
            outputs=[model_load_output, chatbot]
        )
        
        submit_event = submit_btn.click(
            fn=user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False
        ).then(
            fn=lambda: [True],
            outputs=active_gen
        ).then(
            fn=generate_response,
            inputs=[model_handler_state, chatbot, temperature, top_p, top_k, max_tokens, seed, active_gen, model_dropdown, auto_clear],
            outputs=chatbot
        )
        
        msg.submit(
            fn=user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False
        ).then(
            fn=lambda: [True],
            outputs=active_gen
        ).then(
            fn=generate_response,
            inputs=[model_handler_state, chatbot, temperature, top_p, top_k, max_tokens, seed, active_gen, model_dropdown, auto_clear],
            outputs=chatbot
        )
        
        stop_btn.click(
            fn=lambda: [False],
            inputs=None,
            outputs=active_gen,
            cancels=[submit_event]
        )
        
        clear_btn.click(
            fn=lambda: None,
            inputs=None,
            outputs=chatbot,
            queue=False
        )
    
    return demo

def main():
    model_handler = ModelHandler()
    model_handler.load_model(MODEL_IDS[0], [])
    demo = create_ui(model_handler)
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860)
    except Exception as e:
        logger.error(f"Failed to launch Gradio app: {str(e)}")
        raise

if __name__ == "__main__":
    main()
