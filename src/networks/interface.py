from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
import warnings; warnings.filterwarnings('ignore')
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def run_any_model(messages, pth, device, key):
    if key.lower() == 'medgemma':
        pipe = pipeline(
            "image-text-to-text",
            model=pth,
            device=device,
            torch_dtype=torch.bfloat16,
        )

        output = pipe(text=messages, max_new_tokens=3000)

        return(output[0]["generated_text"][-1]["content"])
    
    if key.lower() == 'lingshu':
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pth,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=device,
        )
        processor = AutoProcessor.from_pretrained(pth)

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
        generated_ids = model.generate(**inputs, max_new_tokens=3000)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return [{'text':output_text}]
    
    if key.lower() == 'fleming':
        tokenizer = AutoTokenizer.from_pretrained(pth)
        model = AutoModelForCausalLM.from_pretrained(
            pth,
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