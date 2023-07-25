from djl_python import Input, Output
from djl_python.streaming_utils import StreamingUtils
import os
import deepspeed
import torch
import logging
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
##
# from transformers import AutoModelForCausalLM

model = None
tokenizer = None


def get_model(properties):
    model_name = properties["model_id"]

    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                         # low_cpu_mem_usage=True, 
                                         torch_dtype=torch.float16,
                                         trust_remote_code=True,
                                         device_map="auto")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def inference(inputs):
    try:
        input_map = inputs.get_as_json()
        data = input_map.pop("inputs", input_map)
        parameters = input_map.pop("parameters", {})
        outputs = Output()

        input_tokens = tokenizer(data, padding=True,
                                return_tensors="pt").to(
                                torch.cuda.current_device())

        with torch.no_grad():
            output_tokens = model.generate(input_tokens.input_ids, **parameters)
        generated_text = tokenizer.batch_decode(output_tokens,
                                        skip_special_tokens=True)

        outputs.add_as_json([{"generated_text": s} for s in generated_text])
        return outputs
    except Exception as e:
        logging.exception("Huggingface inference failed")
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