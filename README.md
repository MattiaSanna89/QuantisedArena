# Quantised Arena: Chatting with Multiple Quantized Language Models

<div align="center">

![Python Badge](https://img.shields.io/badge/Python-3.10-black?style=plastic&logo=python&logoColor=%233776AB&labelColor=%23FF005A&color=%233776AB)
![fastapi](https://img.shields.io/badge/FastApi-0.111.0-black?style=plastic&labelColor=%23FF005A)
![gradio](https://img.shields.io/badge/Gradio-4.32.1-black?style=plastic&labelColor=%23FF005A)
![transformers](https://img.shields.io/badge/Transformers-4.41.2-black?style=plastic&labelColor=%23FF005A)
![sentencetransformers](https://img.shields.io/badge/Sentence--Transformers-3.0.0-black?style=plastic&labelColor=%23FF005A)
![gptq](https://img.shields.io/badge/AutoGptq-0.8.0.dev0+cu121-black?style=plastic&labelColor=%23FF005A)

</div>

Quantised Arena is a web application that allows you to chat with multiple locally deployed quantized language models simultaneously and compare their responses. It provides a user-friendly interface built with Gradio and leverages a FastAPI backend for serving the pre-trained models.

## Features

- Chat with two pre-trained quantized language models: Llama3 Instruct 8B and Mistral Instruct v0.2 7B.
- Simultaneous display of user queries and model responses in separate chat windows.
- Regenerate responses for the current query with a single click.
- Clear the chat history and start a new conversation.
- Adjust generation parameters such as temperature, top-p, top-k, and max output tokens.
- Streaming response display for real-time updates.

## Prerequisites

Before running the application, ensure you have the following dependencies installed:

``` python
pip install -r requierements.txt
```