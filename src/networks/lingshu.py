from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import pandas as pd
import os
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')


## NOTE:  this model hallucinates A LOT needs proper base prompt tuning i think.

def run(messages, device):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "lingshu-medical-mllm/Lingshu-32B",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("lingshu-medical-mllm/Lingshu-32B")

    text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
    )
    #print("Rendered prompt:", repr(text[:500]))
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=5000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return [{'text':output_text}]

