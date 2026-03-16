from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
import warnings; warnings.filterwarnings('ignore')
import torch, os, json

## NOTE:  this model hallucinates A LOT needs proper base prompt tuning i think.

def get_avg_report_length(pth, dir):
    tokenizer = AutoTokenizer.from_pretrained("google/medgemma-1.5-4b-it")
    avg_length = 0
    cnt = 0
    for pat in [p for p in os.listdir(pth) if p.startswith('P')]:
        if not any(os.listdir(f"{pth}/{pat}/{dir}")): continue
        rep = [f for f in os.listdir(f"{pth}/{pat}/{dir}") if f.endswith('.txt')][-1]
        with open(f"{pth}/{pat}/{dir}/{rep}", "r") as f:
            report = f.read()
        cnt += 1
        tok = tokenizer(report)
        avg_length += len(tok['input_ids'])
    print(f"Average report length in tokens is: {avg_length/cnt}") # 
        

if __name__ == '__main__':
    # get_avg_report_length("/mnt/nas6/data/PARSOR/02_Cases", "05_Final_Second_Opinion")
    # RESULT: Average report length in tokens is: 1632.8188976377953
    with open('/home/lorenz/RadiologyCheck/v2_fewshot_baseprompt.json', 'r') as j:
        dct = json.load(j)
        with open('/home/lorenz/RadiologyCheck/examples/p014/rep.txt', 'r') as f:
            txt = f.read()
    
    dct[4]['content'][0]['text'] = txt
    with open('/home/lorenz/RadiologyCheck/v3_fewshot_baseprompt.json', 'w') as j:
        json.dump(dct, j, indent=2)
    