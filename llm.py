import time
import torch
import random
import asyncio
import numpy as np
from typing import Dict
from threading import Thread
from transformers import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Prompt Structures
LLAMA = '''
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
'''

MISTRAL='''<s>[INST] {message} [/INST]'''

# Set seeds for reproducibility
random_seed = 42
np_seed = 42
torch_seed = 42
transformers_seed = 42

random.seed(random_seed)
np.random.seed(np_seed)
torch.manual_seed(torch_seed)
set_seed(transformers_seed)


class GenerativeModel():
    """
    A class for loading and generating text from a pre-trained language model.

    Parameters
    ----------
    model_name_or_path : str
        The name or path of the pre-trained language model to load.
    model_loading_kwargs : Dict, optional
        A dictionary of keyword arguments to pass to the AutoModelForCausalLM.from_pretrained method.
        Default is {"device_map":"cuda:0", "low_cpu_mem_usage":True, "attn_implementation":"flash_attention_2", "trust_remote_code":False}.
    tokenizer_kwargs : Dict, optional
        A dictionary of keyword arguments to pass to the AutoTokenizer.from_pretrained method.
        Default is {"use_fast":True}.
    streamer_kwargs : Dict, optional
        A dictionary of keyword arguments to pass to the TextIteratorStreamer.
        Default is {"skip_prompt":True, "skip_special_tokens":True}.

    Attributes
    ----------
    model : AutoModelForCausalLM
        The pre-trained language model.
    tokenizer : AutoTokenizer
        The tokenizer for the pre-trained language model.
    streamer : TextIteratorStreamer
        A streamer for generating text iteratively.
    streamer_kwargs : Dict
        A dictionary of keyword arguments for the TextIteratorStreamer.
    prompt_struct : Dict
        A dictionary containing prompt structures for different model types.
    """

    def __init__(self,
        model_name_or_path:str, 
        model_loading_kwargs:Dict={
            "device_map":"cuda:0",
            "low_cpu_mem_usage":True,
            "attn_implementation":"flash_attention_2",
            "trust_remote_code":False
            },
        tokenizer_kwargs:Dict={"use_fast":True},
        streamer_kwargs: Dict={"skip_prompt":True, "skip_special_tokens":True}
        ):
        self.model = None
        self.tokenizer = None
        self.streamer = None
        self.streamer_kwargs = streamer_kwargs
        self.load_model(
            model_name_or_path, 
            model_loading_kwargs,
            tokenizer_kwargs
        )
        self.prompt_struct = {"llama": LLAMA, "mistral": MISTRAL}
    
    def load_model(self,
        model_name_or_path:str,
        hf_automodel_kwrgs:Dict=dict(), 
        hf_tokenizer_kwrgs:Dict=dict()
        ):
        """
        Load a pre-trained language model and tokenizer.

        Parameters
        ----------
        model_name_or_path : str
            The name or path of the pre-trained language model to load.
        hf_automodel_kwrgs : Dict, optional
            A dictionary of keyword arguments to pass to the AutoModelForCausalLM.from_pretrained method.
            Default is an empty dictionary.
        hf_tokenizer_kwrgs : Dict, optional
            A dictionary of keyword arguments to pass to the AutoTokenizer.from_pretrained method.
            Default is an empty dictionary.

        Returns
        -------
        None
        """
        
        # Free up memory by deleting the previous model
        if self.model:
            del self.model
            torch.cuda.empty_cache()
        
        # Load the tokenizer and model from the specified path    
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **hf_tokenizer_kwrgs) 
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **hf_automodel_kwrgs)
        
        return None

    async def gradio_generate(self, 
                      message: str,
                      model_type: str="llama",
                      params: Dict={
                          "max_new_tokens": 50,
                          "temperature": 0.1,
                          "do_sample":True, 
                          "top_p":0.95, 
                          "top_k":40},
                      streamer_kwargs: Dict | None=None    
                     ):
        """
        Generate text from a pre-trained language model using a prompt.

        Parameters
        ----------
        message : str
            The prompt message to use for generating text.
        model_type : str, optional
            The type of prompt structure to use. Can be "llama" or "mistral".
            Default is "llama".
        params : Dict, optional
            A dictionary of keyword arguments to pass to the model.generate method.
            Default is {"max_new_tokens": 50, "temperature": 0.1, "do_sample":True, "top_p":0.95, "top_k":40}.
        streamer_kwargs : Dict or None, optional
            A dictionary of keyword arguments to pass to the TextIteratorStreamer.
            If None, the default streamer_kwargs from the class instance will be used.

        Yields
        ------
        str
            The generated text, streamed as it is produced.
        """
        if not isinstance(message, str):
            print(message)
            raise TypeError
        # Format the prompt based on the specified model type
        try:
            prompt= self.prompt_struct[model_type].format(message=message)
        except:
            print(self.prompt_struct[model_type])

        # Tokenize the prompt and convert it to a tensor on the GPU
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.cuda()

        # Initiate the text streamer
        if not streamer_kwargs:
            streamer_kwargs = self.streamer_kwargs
        self.streamer = TextIteratorStreamer(self.tokenizer, **streamer_kwargs)

        # Start the generation in a separate thread
        params.update({"input_ids": input_ids, "streamer": self.streamer})        
        thread = Thread(target=self.model.generate, 
                        kwargs=params)
        thread.start()

        # Iterate over the streamed text and yield it in chunk
        partial_message = ""
        for new_text in self.streamer:
            partial_message += new_text
            yield partial_message
            # Add a small delay for better streaming.
            await asyncio.sleep(0.1)
    