from djl_python import Input, Output
from djl_python.streaming_utils import StreamingUtils
import os
import torch
import logging


from InstructorEmbedding import INSTRUCTOR

model = None
tokenizer = None


def get_model(properties):
    model_name = properties["model_id"]
    # tensor_parallel_degree = properties["tensor_parallel_degree"]
    # max_tokens = int(properties.get("max_tokens", "1024"))
    # dtype = torch.float16
    model = INSTRUCTOR(model_name)

    return model


def inference(inputs):
    try:
        input_map = inputs.get_as_json()
        data = input_map.pop("inputs", input_map)
        parameters = input_map.pop("parameters", {})
        outputs = Output()

        instructs = data['instructs']
        docs = data['docs']
        embeddings = model.encode([[instructs[i], docs[i]] for i in range(len(docs))])

        outputs.add_as_json([{"emb_vec": s} for s in embeddings])
        return outputs
    except Exception as e:
        logging.exception("Huggingface inference failed")
        # error handling
        outputs = Output().error(str(e))



def handle(inputs: Input) -> None:
    global model, tokenizer
    if not model:
        model = get_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None

    return inference(inputs)

