from djl_python import Input, Output
from djl_python.streaming_utils import StreamingUtils
import os
import deepspeed
import torch
import logging
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModel
from torch import autocast

model = None
tokenizer = None


def get_model(properties):
    model_name = properties["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).half().cuda()

    return model, tokenizer


def inference(inputs):
    try:
        input_map = inputs.get_as_json()
        data = input_map.pop("inputs", input_map)
        parameters = input_map.pop("parameters", {})
        outputs = Output()
        
        with torch.no_grad():
            
            inputs = tokenizer(data, return_tensors='pt').to('cuda:0')
            pred = model.generate(**inputs,
                                  max_new_tokens = parameters['max_new_tokens'],
                                  repetition_penalty = parameters['repetition_penalty']
                                 )
            
            response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        
        outputs.add_as_json({"response": response})
        return outputs
    
    except Exception as e:
        logging.exception("Inference failed")
        # error handling
        outputs = Output().error(str(e))



def handle(inputs: Input) -> None:
    global model, tokenizer
    if not model:
        model, tokenizer = get_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None

    return inference(inputs)