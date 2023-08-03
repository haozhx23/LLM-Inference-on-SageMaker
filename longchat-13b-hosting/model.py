from djl_python import Input, Output
from djl_python.streaming_utils import StreamingUtils
import os
import deepspeed
import torch
import logging
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoConfig
##

os.system('python -m pip install .')

model = None
tokenizer = None

# def get_gpu_memory(max_gpus=None):
#     """Get available memory for each GPU."""
#     import torch

#     gpu_memory = []
#     num_gpus = (
#         torch.cuda.device_count()
#         if max_gpus is None
#         else min(max_gpus, torch.cuda.device_count())
#     )

#     for gpu_id in range(num_gpus):
#         with torch.cuda.device(gpu_id):
#             device = torch.cuda.current_device()
#             gpu_properties = torch.cuda.get_device_properties(device)
#             total_memory = gpu_properties.total_memory / (1024**3)
#             allocated_memory = torch.cuda.memory_allocated() / (1024**3)
#             available_memory = total_memory - allocated_memory
#             gpu_memory.append(available_memory)
#     return gpu_memory

# kwargs = {"torch_dtype": torch.float16}
# if num_gpus != 1:
#     kwargs["device_map"] = "auto"
#     if max_gpu_memory is None:
#         kwargs[
#             "device_map"
#         ] = "sequential"  # This is important for not the same VRAM sizes
#         available_gpu_memory = get_gpu_memory(num_gpus)
#         kwargs["max_memory"] = {
#             i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
#             for i in range(num_gpus)
#         }
#     else:
#         kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}


def get_model(properties):
    model_name = properties["model_id"]

    revision = 'main'
    config = AutoConfig.from_pretrained(model_name, revision=revision)
    from fastchat.model.llama_condense_monkey_patch import (
            replace_llama_with_condense,
        )

    replace_llama_with_condense(config.rope_condense_ratio)

    tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, revision=revision, trust_remote_code=True
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        max_memory={i: "20GiB" for i in range(4)}
    )
    ## g5.12xl - 4*A10g

    return model, tokenizer


def inference(inputs):
    try:
        input_map = inputs.get_as_json()
        data = input_map.pop("inputs", input_map)
        parameters = input_map.pop("parameters", {})
        outputs = Output()

        input_tokens = tokenizer(data, padding=True,
                                 return_tensors="pt",
                                 return_token_type_ids=False, # https://github.com/huggingface/peft/issues/506
                                ).to(torch.cuda.current_device())

        # output_ids = model.generate(
        #     **input_tokens, # **parameters
        #     do_sample=True,
        #     temperature=0.7,
        #     repetition_penalty=1.1,
        #     max_new_tokens=100,
        # )
        
        output_ids = model.generate(
            **input_tokens,
            **parameters
        )

        output_ids = output_ids[0][len(input_tokens["input_ids"][0]) :]
        generated_text = tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        outputs.add_as_json({"generated_text": generated_text})
        
        # generated_text = tokenizer.batch_decode(output_tokens, ...)
        # outputs.add_as_json([{"generated_text": s} for s in generated_text])
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