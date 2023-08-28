import gradio as gr
from utils import ChatEngine

model = "gpt-4-0613"
def quotify(s: str) -> str:
    """Adds quotes to a string.
    Args:
        s (str): The string to add quotes to.
    Returns:
        (str) The string with quotes added.
    """
    return "```" + s + "```"

ChatEngine.setup(model, quotify_fn=quotify)
chat_engine = ChatEngine()

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    with gr.Row():
        verbose = gr.Checkbox(label="Verbose Mode", value=chat_engine.verbose)
        style = gr.Button(value="Set Style")
        clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        reply_combined = "\n".join(reply for reply in chat_engine.reply_message(message))
        chat_history.append((message, reply_combined))
        return "", chat_history

    def set_verbose_mode(value):
        chat_engine.verbose = value
    
    def set_style(message):
        global chat_engine
        message = message.strip()
        if message:
            print(f"Style: {message}")
            chat_engine = ChatEngine(style=message)

        return "", chat_engine.verbose
    
    def clear_chat_engine():
        global chat_engine
        chat_engine = ChatEngine()
        return chat_engine.verbose
        
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    verbose.change(set_verbose_mode, verbose)
    style.click(set_style, [msg], [msg, verbose])
    clear.click(clear_chat_engine, [], [verbose])

demo.launch()
