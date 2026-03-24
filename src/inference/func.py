from pypdf import PdfReader
import sys, os, pprint, pathlib as pl, json, tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.networks.medgemma import run_medgemma
from src.utils.io import extract_text
from src.utils.language_handling import LangDct
from src.networks.translation import Translator
from src.utils.eval_bleu import bleu
from src.utils.semantic_sim import semantic_similarity

def infer(ds, out, enable_3rd_party_translation=False):
    out = pl.Path(out)
    os.makedirs(out, exist_ok=True)
    with open("config.json", "r") as f:
        cfg = json.load(f)
    needs_to_english = LangDct(enable_3rd_party_translation)
    to_english = Translator(model_name=cfg['tran'], device=cfg['device'])

    for id, prompt in tqdm.tqdm(ds, desc='Inferring'):
        try:
            if os.path.exists(out/id):continue
            ## fuse with few shot prompt here ----------------------------------------------------------------------------
            with open(cfg['baseprompt'], "r") as msg:
                messages = json.load(msg)
            messages.append(prompt)
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": "## Radiology Second Opinion Report\n\n### Part 1: Clinical History\n\n"}]
            })
            os.makedirs(out/id, exist_ok=True)
            with open(out/id/'prmpt.json', 'w') as f:
                json.dump(messages, f, indent=4)
            ## run inference
            result = run_medgemma(messages, pth=cfg['inf'], device=cfg['device'])
            
            ## save output
        
            with open(out/id/'generated_report.txt', 'w') as f:
                f.write(result[-1]['text'])   
        except: continue
def eval(ds, out):
    out = pl.Path(out)
    ds.return_type='dict'
    os.makedirs(out, exist_ok=True)
    cnt = 0
    bleu_accum = 0
    semsim_accum = 0
    scores = {}
    for sample in tqdm.tqdm(ds):
        try:
            with open(out/sample['id']/'generated_report.txt', 'r') as f:
                result = f.read()
                bleu_score = bleu(result, sample['final_report'])
                semsim = semantic_similarity(result, sample['final_report'])
            print(f"Sample {sample['id']} achieved BLEU {bleu_score}; SemSim {semsim}")
            cnt += 1; bleu_accum+=bleu_score; semsim_accum+=semsim
            scores[sample['id']] = {"bleu": bleu_score, "SemSim": semsim}
        except: continue
    
    with open(out/'eval.json', 'w') as f:
        json.dump(scores, f, indent=4)
        
    print(f"Avg over {cnt} samples is: BLEU {bleu_accum/cnt}; SemSim {semsim_accum/cnt}")