from pypdf import PdfReader
import sys, os, pprint, pathlib as pl, json
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.networks.medgemma import run_medgemma
from src.utils.io import extract_text
from src.utils.language_handling import LangDct
from src.networks.translation import Translator

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
        
        ## Inputs ----------------------------------------------------------------------------
        subspecialty = input(
            "Enter the required subspecialty: "
        )
        referral = input(
            "Enter the patients information (referral, reports, history, ...) : "
        )
        if needs_to_english(referral): referral = to_english(referral)
        files = []
        if input("Are there any additional text (.pdf or .txt) files available? (y/n)").lower()=="y":
            while True:
                fp = input("Enter the filepath or type 'ESCAPE' to end:\n")
                if fp.lower() == "escape" or fp.lower() == "'escape'":
                    break
                else: files.append(fp)
        
        expert_findings = input(
            "Enter your expert findings: "
        )
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
        prompt = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"You are a top-tier subspecialty radiologist ({subspecialty}) with 20+ years of experience, writing a second-opinion report for RadiologyCheck. Your task is to analyze the clinical records, then generate a comprehensive, timeline and clinical history that includes: Patient-Centered Context - Clinical History: - Extract & summarize indications, symptoms, lab results, clinical functional test results past medical/surgical history, medications, and risk factors (e.g., smoking, oncology history). - Highlight key clinical questions, highlight the main diagnostic or therapeutic question(e.g., 'Rule out metastasis in a patient with breast cancer'). - Medical Timeline: - Create a chronological table of prior imaging, biopsies, and treatments (include dates, modalities, and key findings). Date Modality KeyFindings Impact - Problem List: Prioritize active issues (e.g., '1. Growing pulmonary nodule; 2. Enlarged mediastinal lymph nodes')."
                    },
                    {
                        "type": "text",
                        "text": f"Information: {referral}"
                    },
                    {
                        "type": "text",
                        "text": f"You are a top-tier subspecialty abdominal radiologist ({subspecialty}) with 20+ years of experience. Your task is to generate a comprehensive, publication-quality report that includes: My second opinion findings report (Structured & Detailed) A. Technical Details - Scan type (e.g., 'Non-contrast CT chest, 120 kVp, iterative reconstruction'). - Limitations (e.g., 'Motion artifact limits evaluation of liver dome'). B. Systematic Findings - Positive Findings: Describe with precise measurements, location, and characteristics (e.g., '8mm spiculated LUL nodule with SUV 4.2 on prior PET'). - Negative Findings: Explicitly state normalcy (e.g., 'No pleural effusion, pneumothorax, or acute fractures'). - Comparison: Use quantitative changes (e.g., 'Nodule increased from 6mm to 8mm over 4 months [33% growth rate]'). - Use radiology scoring systems (e.g., LI-RADS for liver, PI-RADS for prostate). --- Differential Diagnosis & Clinical Integration - Prioritized DDx: Rank likely diagnoses (e.g., '1. Primary lung adenocarcinoma; 2. Granuloma; 3. Metastasis'). - Evidence-Based Rationale: Cite guidelines (e.g., 'Fleischner Criteria recommend 6-month follow-up for this 8mm solid nodule'). - Correlation: Match imaging with lab results/biopsies (e.g., 'Nodule growth correlates with rising CEA levels'). - Proofreading & Quality Assurance - Consistency Check: Ensure measurements, dates, and descriptions align across reports. - Error Detection: Flag discrepancies (e.g., 'Prior report described a 5mm nodule; now measures 7mm'). - Style Refinement: - Use active voice ('The liver shows no lesions' → 'No hepatic lesions are seen'). - Avoid vague terms ('suspicious for' → 'features suggest malignancy'). Premium Recommendations - Next Steps: - Short-term (e.g., 'Biopsy recommended for the LUL nodule'). - Long-term (e.g., 'Annual low-dose CT screening for lung cancer'). - Patient history: - Lifestyle (e.g., 'Smoking cessation reduces malignancy risk by 50%'). - Prognosis (e.g., '5-year survival for Stage I NSCLC is 80% with resection'). - Multidisciplinary Coordination: - Suggest referrals (e.g., 'Oncology consult for possible immunotherapy'). Patient-Friendly Report: A simplified version for the patient (e.g., 'Your scan shows a small lung spot; we recommend a follow-up in 6 months'). - Research Integration: Include recent literature (e.g., 'Per 2024 ACR guidelines, this nodule warrants PET-CT'). 1. 'Peace of Mind' Boosters Add to Report: - Likelihood Statements: - This finding has a >90% chance of being benign based on size/stability. - No signs of urgent or life-threatening conditions were detected. - Natural History: - Explain what typically happens with similar findings (e.g., Most nodules like this remain stable or resolve without treatment). - False-Reassurance Avoidance: - While this appears benign, follow-up ensures we catch rare exceptions early. Prompt Addition: Include statistical probabilities (when evidence-based) and natural history explanations for all findings to reduce anxiety. --- 2. Happiness & Trust Drivers A. Transparency Tools: - 'Why We Think This' Section: - This is classified as 'likely benign' because: (1) smooth margins, (2) stable for 2 years, (3) no risk factors. - Peer Comparison: - Our conclusion aligns with 95% of academic radiologists who reviewed this case anonymously. B. Patient Empowerment: - 'Your Next Steps' Table: | Action | Purpose | Timeline | |--------|---------|----------| | Follow-up CT | Confirm nodule stability | 6 months | | Smoking cessation | Reduce future risks | Immediate | - Example Questions for Their Doctor: - 'Ask your physician': (1) Should we consider a PET scan? (2) Are there symptoms I should watch for? Prompt Addition: Add a 'Why We Think This' rationale for key conclusions and a clear action table with timelines. Include 2-3 questions the patient should ask their primary doctor."
                    },
                    {
                        "type": "text",
                        "text": f"My findings: {expert_findings}"
                    }
                ]
            }
        
        ## Fuse supplementary files ----------------------------------------------------------------------------
        if any(supplementary_files): prompt["content"] += supplementary_files
        
        ## fuse with few shot prompt here ----------------------------------------------------------------------------
        with open(cfg['baseprompt'], "r") as msg:
            messages = json.load(msg)
        messages.append(prompt)
    
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
        result = run_medgemma(messages, pth=cfg['inf'])
        
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
            f.write(result)
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
    return