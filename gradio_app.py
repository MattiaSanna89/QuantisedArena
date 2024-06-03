"""
Quantised Arena: A Gradio Interface for Chatting with Multiple Quantized Language Models

This code sets up a Gradio interface for chatting with multiple locally deployed quantized language models
simultaneously and comparing their responses. It supports two pre-trained models: Llama3 Instruct 8B and
Mistral Instruct v0.2 7B.

The interface provides the following features:

- A text input box for entering queries.
- Two chat windows, one for each model, where the user's queries and the models' responses are displayed.
- A "Regenerate" button to regenerate the responses for the current query.
- A "New Round" button to clear the chat history and start a new conversation.
- An accordion panel to adjust the generation parameters, including temperature, top-p, top-k, and max output tokens.

The backend of the application is powered by a FastAPI server running locally on port 8000. The server handles
the requests for generating text from the pre-trained models and streams the responses back to the Gradio interface.

To use the application, follow these steps:

1. Start the FastAPI server by running the provided server script.
2. Run this script to launch the Gradio interface.
3. Enter your query in the text input box and press Enter.
4. The queries will be sent to the FastAPI server, and the responses will be displayed in the respective chat windows.
5. Use the "Regenerate" button to generate new responses for the current query.
6. Use the "New Round" button to clear the chat history and start a new conversation.
7. Adjust the generation parameters in the accordion panel as needed.

Note: This application assumes that the FastAPI server is running locally on port 8000. If the server is running
on a different port or address, update the URL in the `gen_responses` function accordingly.

Run by executing the following terminal command
 
 python gradio_app.py
"""

import time
import requests
import gradio as gr

######## BACKEND ########
def activate_chat_buttons():
    """
    Activates the regenerate and clear buttons in the Gradio UI.

    Returns
    -------
    regenerate_btn : gradio.Button
        The regenerate button component with interactive set to True.
    clear_btn : gradio.ClearButton
        The clear button component with interactive set to True.
    """
    regenerate_btn = gr.Button(
        value="🔄  Regenerate", interactive=True, elem_id="regenerate_btn"
    )
    clear_btn = gr.ClearButton(
        elem_id="clear_btn",
        interactive=True,
    )
    return regenerate_btn, clear_btn


def deactivate_chat_buttons():
    """
    Deactivates the regenerate and clear buttons in the Gradio UI.

    Returns
    -------
    regenerate_btn : gradio.Button
        The regenerate button component with interactive set to False.
    clear_btn : gradio.ClearButton
        The clear button component with interactive set to False.
    """
    regenerate_btn = gr.Button(
        value="🔄  Regenerate", interactive=False, elem_id="regenerate_btn"
    )
    clear_btn = gr.ClearButton(
        elem_id="clear_btn",
        interactive=False,
    )
    return regenerate_btn, clear_btn

def user(message, states1, states2):
    """
    Appends the user's message to the chat history for both chatbots.

    Parameters
    ----------
    message : str
        The user's message.
    states1 : gradio.State or None
        The state object containing the chat history for the first chatbot.
    states2 : gradio.State or None
        The state object containing the chat history for the second chatbot.

    Returns
    -------
    history1 : list
        The updated chat history for the first chatbot.
    history2 : list
        The updated chat history for the second chatbot.
    """
    history1 = states1.value if states1 else []
    history2 = states2.value if states2 else []
    states = [states1, states2]
    history = [history1, history2]
    for hist in history:
        hist.append([message, None])
    
    return history1, history2

def gen_responses(bot0, bot1, states1, states2, 
                  do_sample, temperature, top_p, top_k, 
                  max_output_tokens):
    """
    Generates responses from the two chatbots for the last user message.

    Parameters
    ----------
    bot0 : gradio.Chatbot
        The Gradio Chatbot component for the first chatbot.
    bot1 : gradio.Chatbot
        The Gradio Chatbot component for the second chatbot.
    states1 : gradio.State
        The state object containing the chat history for the first chatbot.
    states2 : gradio.State
        The state object containing the chat history for the second chatbot.
    do_sample : bool
        Whether to use sampling for text generation.
    temperature : float
        The temperature value for sampling.
    top_p : float
        The top-p value for sampling.
    top_k : int
        The top-k value for sampling.
    max_output_tokens : int
        The maximum number of tokens to generate.

    Yields
    ------
    history0 : list
        The updated chat history for the first chatbot.
    history1 : list
        The updated chat history for the second chatbot.
    """
    history = [bot0, bot1]
    states = [states1, states2]
    for i, url in enumerate(["llama3/", "mistral/"]):
        message = history[i][-1][0]
        with requests.post("http://localhost:8000/generate/" + url, 
                           json={
                               'inputs':message, 
                               'parameters': {
                                   "temperature":temperature,
                                   "do_sample":do_sample, 
                                   "top_p":top_p, 
                                   "top_k":top_k,
                                   "max_new_tokens":max_output_tokens}
                           }, stream=True) as r:
            history[i][-1][1] = ""
            for new_text in r.iter_content(1024):
                history[i][-1][1]  = new_text.decode("utf-8")
                states[i] = gr.State(history[i])
                yield history[0], history[1]        



######## GUI ########
with gr.Blocks(
    title="Quantised Arena",
    theme='snehilsanyal/scikit-learn') as demo:
    num_sides=2
    states = [gr.State() for _ in range(num_sides)] # Create state objects to store chat histories
    chatbots = [None] * num_sides # Initialise empty list to store chatbot components
    gr.Markdown(
        "# Quantised Arena\n\nChat with multiple locally deployed quantised LLM at the same time and compare their responses."
    )

    # Create textbox for user input
    msg = gr.Textbox(
            show_label=False,
            placeholder="Enter your query and press ENTER",
            elem_id="input_box",
            scale=4,
        )

    # Create chatbot components
    with gr.Row():
        for i,model in enumerate(["astronomer/Llama-3-8B-Instruct-GPTQ-4-Bit", "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ-4-Bit"]):
            label = model
            with gr.Column():
                chatbots[i] = gr.Chatbot(
                    label=label,
                    elem_id=f"chatbot",
                    height=550,
                    show_copy_button=True,
                )

    # Create utility buttons
    with gr.Row() as button_row:
        clear_btn = gr.ClearButton(
            value="🎲 New Round",
            elem_id="clear_btn",
            interactive=False,
            components=chatbots + states,
        )
        regenerate_btn = gr.Button(
            value="🔄 Regenerate", interactive=False, elem_id="regenerate_btn"
        )

    # Model Generation Parameters (shared between the models)
    with gr.Accordion("Parameters", open=False) as parameter_row:
        
        do_sample = gr.Checkbox(
            value=False,
            label="Do Sample",
            interactive=True,
        )        
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.0,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.95,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        top_k = gr.Slider(
            minimum=0.0,
            maximum=80,
            value=40,
            step=1,
            interactive=True,
            label="Top K",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=400,
            value=100,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    # Action on user submission
    # Updates bots states
    # Generate responses
    # activate utility buttons
    msg.submit(fn=user, inputs=[msg, states[0], states[1]], outputs=[chatbots[0], chatbots[1]], queue=False)\
        .then(gen_responses, 
              inputs=[chatbots[0], chatbots[1], states[0], states[1], 
                      do_sample, temperature, top_p, top_k, max_output_tokens], 
              outputs=[chatbots[0], chatbots[1]])\
        .then(activate_chat_buttons, inputs=[], outputs=[regenerate_btn, clear_btn])

    # Action on Regenerate click
    regenerate_btn.click(
        gen_responses,
        inputs=[
            chatbots[0], chatbots[1], states[0], states[1],
            do_sample, temperature, top_p, top_k, max_output_tokens
        ],
        outputs=[chatbots[0], chatbots[1]],
    )
    
    # Action on Clear click
    clear_btn.click(
        deactivate_chat_buttons,
        inputs=[],
        outputs=[regenerate_btn, clear_btn],
    )

if __name__ == "__main__":
    demo.queue().launch(share=False)