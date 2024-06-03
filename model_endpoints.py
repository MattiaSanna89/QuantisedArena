"""
FastAPI Application for Serving Language Models

This application provides a FastAPI server for serving pre-trained language models
for text generation. It supports two models, Llama3 Instruct 8B and Mistral Instruct v0.2 7B,
and allows users to send prompts and parameters to the API endpoints and receive
the generated text as a streaming response.

The application includes the following endpoints:

- /gpu_info/: Checks if a GPU is available and returns information about the GPU.
- /generate/llama3: Generates text using the Llama3 Instruct 8B model.
- /generate/mistral: Generates text using the Mistral Instruct v0.2 7B model.

The input data is expected to be in the following format:

{
    "inputs": "The prompt text for text generation",
    "parameters": {
        "max_new_tokens": 50,
        "temperature": 0.1,
        "...": "..."
    }
}

The generated text is returned as a streaming response, allowing real-time display
or processing of the generated text as it becomes available.

To run the app adapt the following terminal command

uvicorn model_endpoints:app --host 127.0.0.1 --port 8000
"""

import sys
import torch
import asyncio
from typing import Any, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from llm import GenerativeModel
from fastapi.responses import StreamingResponse

app = FastAPI()
  
DEFAULT_LOADING_PARAM = {
    "device_map":"cuda:0",
    "low_cpu_mem_usage":True,
    "attn_implementation":"flash_attention_2",
    "trust_remote_code":False,
    "revision": None}

# Llama3 Instruct 8B model
MODEL_NAME_OR_PATH = "astronomer-io/Llama-3-8B-Instruct-GPTQ-4-Bit"  
llama3 = GenerativeModel(MODEL_NAME_OR_PATH, DEFAULT_LOADING_PARAM)

# Mistral Instruct v0.2 7B model
MODEL_NAME_OR_PATH = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"  
DEFAULT_LOADING_PARAM["revision"] = "gptq-4bit-32g-actorder_True"
mistral = GenerativeModel(MODEL_NAME_OR_PATH, DEFAULT_LOADING_PARAM)


# Get GPU availability and info
@app.get("/gpu_info/")
def status_gpu_check() -> Dict[str, str]:
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_properties(device=torch.cuda.current_device())
        gpu_msg = str({"model": gpu_info.name, "memory": gpu_info.total_memory, "multiprocessor": gpu_info.multi_processor_count})
    
    else:
        gpu_msg = "Unavailable"
    
    return {
        "status": "Ready!",
        "gpu": gpu_msg
    }

# Expected data input
class TextInput(BaseModel):
    inputs: str
    parameters: Dict[str, Any] | None

# Llama3 endpoint for text streaming generation
@app.post("/generate/llama3")
async def generate_text(data: TextInput) -> Dict[str, str]:
    try:
        params = data.parameters or {}
        return StreamingResponse(llama3.gradio_generate(data.inputs, "llama", data.parameters), media_type='text/event-stream')
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=len(str(e)))

# Mistral endpoint for text streaming generation
@app.post("/generate/mistral")
async def generate_text(data: TextInput) -> Dict[str, str]:
    try:
        params = data.parameters or {}
        return StreamingResponse(mistral.gradio_generate(data.inputs, "mistral", data.parameters), media_type='text/event-stream')
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=len(str(e)))