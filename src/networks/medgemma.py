from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
import warnings; warnings.filterwarnings('ignore')
import torch

def run(messages, device):
    pipe = pipeline(
        "image-text-to-text",
        model="models/inf",
        device=device,
        torch_dtype=torch.bfloat16,
    )
    # streamer = TextStreamer(
    #     AutoTokenizer.from_pretrained(pth),
    #     skip_prompt=True,        # Don't re-print your input prompt
    #     skip_special_tokens=True # Cleaner output
    # )

    output = pipe(text=messages, max_new_tokens=3000)

    return(output[0]["generated_text"][-1]["content"])


def run_medgemma(messages, pth, device):
    pipe = pipeline(
        "image-text-to-text",
        model=pth,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    # streamer = TextStreamer(
    #     AutoTokenizer.from_pretrained(pth),
    #     skip_prompt=True,        # Don't re-print your input prompt
    #     skip_special_tokens=True # Cleaner output
    # )

    output = pipe(text=messages, max_new_tokens=3000)

    return(output[0]["generated_text"][-1]["content"])