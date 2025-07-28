import os
import sys
import gradio as gr

sys.path.append(os.path.join(os.path.dirname(__file__), 'gradio_app'))

from config import logger, MODEL_IDS
from model_handler import ModelHandler
from generator import generate_response

DESCRIPTION = '''
<h1><span class="intro-icon">⚕️</span> Medical Chatbot with LoRA Models</h1>
    <h2>AI-Powered Medical Insights</h2>
    <div class="intro-highlight">
        <strong>Explore our advanced models, fine-tuned with LoRA for medical reasoning in Vietnamese.</strong>
    </div>
    <div class="intro-disclaimer">
        <strong><span class="intro-icon">ℹ️</span> Notice:</strong> For research purposes only. AI responses may have limitations due to development, datasets, and architecture. <strong>Always consult a medical professional for health advice 🩺</strong>.
</div>
'''
CSS_PATH = "gardio_app/static/style.css"
JS_PATH = "gardio_app/static/script.js"
def user(message, history):
    if not isinstance(history, list):
        history = []
    return "", history + [[message, None]]

def create_ui(model_handler):
    with gr.Blocks(css=CSS_PATH, theme=gr.themes.Default()) as demo:
        gr.Markdown(DESCRIPTION)
        gr.HTML('<script src=JS_PATH></script>')
        active_gen = gr.State([False])
        model_handler_state = gr.State(model_handler)
        
        chatbot = gr.Chatbot(
            elem_id="chatbot",
            height=500,
            show_label=False,
            render_markdown=True
        )

        with gr.Row():
            msg = gr.Textbox(
                label="Message",
                placeholder="Type your medical query in Vietnamese...",
                container=False,
                scale=4
            )
            submit_btn = gr.Button("Send", variant='primary', scale=1)
        
        with gr.Column(scale=2):
            with gr.Row():
                clear_btn = gr.Button("Clear", variant='secondary')
                stop_btn = gr.Button("Stop", variant='stop')
            
            with gr.Accordion("Parameters", open=False):
                model_dropdown = gr.Dropdown(
                    choices=MODEL_IDS,
                    value=MODEL_IDS[0],
                    label="Select Model",
                    interactive=True
                )
                temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.7, label="Temperature")
                top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, label="Top-p")
                top_k = gr.Slider(minimum=1, maximum=100, value=64, step=1, label="Top-k")
                max_tokens = gr.Slider(minimum=128, maximum=4084, value=512, step=32, label="Max Tokens")
                seed = gr.Slider(minimum=0, maximum=2**32, value=42, step=1, label="Random Seed")
                auto_clear = gr.Checkbox(label="Auto Clear History", value=True, 
                                         info="Clears internal conversation history after each response but keeps displayed previous messages.")

        gr.Examples(
            examples=[
                ["Khi nghi ngờ bị loét dạ dày tá tràng nên đến khoa nào tại bệnh viện để thăm khám?"],
                ["Triệu chứng của loét dạ dày tá tràng là gì?"],
                ["Tôi bị mất ngủ, tôi phải làm gì?"],
                ["Tôi bị trĩ, tôi có nên mổ không?"]
            ],
            inputs=msg,
            label="Example Medical Queries"
        )
        
        model_load_output = gr.Textbox(label="Model Load Status")
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
