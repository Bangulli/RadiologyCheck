from pypdf import PdfReader
import sys, os, pprint, pathlib as pl, json
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.networks.medgemma import run_medgemma
from src.utils.io import extract_text
from src.utils.language_handling import LangDct
from src.networks.translation import Translator
from src.networks.interface import run_any_model

def IODialogue():
    ## Inputs ----------------------------------------------------------------------------
    subspecialty = input(
        "Enter the required subspecialty: "
    )
    referral = input(
        "Enter the patients information (referral, reports, history, ...) : "
    )
    files = []
    # if input("Are there any additional text (.pdf or .txt) files available? (y/n)").lower()=="y":
    #     while True:
    #         fp = input("Enter the filepath or type 'ESCAPE' to end:\n")
    #         if fp.lower() == "escape" or fp.lower() == "'escape'":
    #             break
    #         else: files.append(fp)

    expert_findings = input(
        "Enter your expert findings: "
    )
    return subspecialty, referral, files, expert_findings

def PromptAssembly(specialty, history, findings):
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Specialty: {specialty}\n\n"
                        f"History: {history}\n\n"
                        f"Findings: {findings}"
                    )
                }
            ]
        }
    

def infer(args):
    ## Vars ----------------------------------------------------------------------------
    wdir = pl.Path(args.dir)
    with open("config.json", "r") as f:
        cfg = json.load(f)
    needs_to_english = LangDct()
    to_english = Translator(model_name=cfg['tran'])
    
    ## from scratch case ----------------------------------------------------------------------------
    if not args.resume:
        i = 1
        suffix = None
        while wdir.exists():
            if suffix is not None: nme = wdir.name.removesuffix(suffix)
            else: nme = wdir.name
            suffix = f"_{i}"
            wdir = wdir.parent/f"{nme}{suffix}"
            i += 1
        os.mkdir(wdir)        
        
        subspecialty, referral, files, expert_findings = IODialogue()
        
        if needs_to_english(referral): referral = to_english(referral)
        if needs_to_english(expert_findings): expert_findings = to_english(expert_findings)
        
        ## Parsing ----------------------------------------------------------------------------
        supplementary_files = []
        for fp in files:
            txt = extract_text(fp)
            if needs_to_english(txt): txt = to_english(txt)
            dct = {
                'type': 'text',
                'text': txt
            }
            supplementary_files.append(dct)
        
        ## User prompt ----------------------------------------------------------------------------
        prompt = PromptAssembly(subspecialty, referral, expert_findings)
        
        ## Fuse supplementary files ----------------------------------------------------------------------------
        #if any(supplementary_files): prompt["content"] += supplementary_files
        
        ## fuse with few shot prompt here ----------------------------------------------------------------------------
        with open(cfg['baseprompt'], "r") as msg:
            messages = json.load(msg)
        messages.append(prompt)
        messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": "## Radiology Second Opinion Report\n\n### Part 1: Clinical History\n\n"}]
            })
    
    ## resume case ----------------------------------------------------------------------------
    else:
        reprompt = input("Type your new prompt: ")
        if needs_to_english(reprompt): reprompt = to_english(reprompt)
        with open(wdir/"prompt.json", "r") as f:
            messages = json.load(f)
        messages += [
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": reprompt}
                    ]
                }
            ]
    
    ## run model ----------------------------------------------------------------------------
    while True:
        ## infer ----------------------------------------------------------------------------
        result = run_any_model(messages, pth=cfg['inf'], device=cfg['device'], key=cfg['model'])
        
        ## save full prompt ----------------------------------------------------------------------------
        messages += [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text",
                         "text": result}
                    ]
                }
            ]
        with open(wdir/"prompt.json", "w") as f:
            json.dump(messages, f, indent=4)
        
        ## result handling ----------------------------------------------------------------------------
        with open(wdir/"report.md", "w") as f:
            f.write(result[-1]['text'])
        print(f"Output saved to {wdir/"report.md"}!")
        
        ## refinement handling ----------------------------------------------------------------------------
        req_more = input("Would you like to refine the report? (y/n)")
        if req_more.lower() == "y":
            reprompt = input("Type your new prompt: ")
            if needs_to_english(reprompt): reprompt = to_english(reprompt)
            messages += [
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": reprompt}
                    ]
                }
            ]
            
        ## exit ----------------------------------------------------------------------------
        else: break
    if not args.ret: return
    else: return result

