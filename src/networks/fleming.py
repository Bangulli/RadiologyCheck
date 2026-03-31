from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
import warnings; warnings.filterwarnings('ignore')
import torch

def run(messages, device):
    tokenizer = AutoTokenizer.from_pretrained("UbiquantAI/Fleming-R1-32B")
    model = AutoModelForCausalLM.from_pretrained(
        "UbiquantAI/Fleming-R1-32B",
        torch_dtype="auto",
        device_map=device
    )
    text = tokenizer.apply_chat_template(
        flatten_messages(messages),
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=3000
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    thinking_content = output.split("<think>")[-1].split("</think>")[0]
    output_text = output.split("</think>")[-1]

    return [{'text':output_text}]

def flatten_messages(messages):
    """Convert OpenAI-style list content to plain strings."""
    flat = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, list):
            # Extract text from each block
            content = "\n".join(
                block["text"] for block in content if block.get("type") == "text"
            )
        flat.append({"role": msg["role"], "content": content})
    return flat