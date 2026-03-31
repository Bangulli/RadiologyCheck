import os, sys, pathlib
from torch.utils.data import Dataset
from src.utils.language_handling import LangDct
from src.networks.translation import Translator

class PARSORData(Dataset):
    def __init__(self, enable_3rd_party_translation=True, return_type='prompt'):
        ## the datadir
        self.path = pathlib.Path(os.path.realpath(__file__)).parent
        
        ## the information fields
        self.fields = ['specialty', 'history', 'findings', 'final_report']
        self.lngdct = LangDct(enable_3rd_party_translation)
        self.enable_trans = enable_3rd_party_translation
        self.to_english = Translator() if enable_3rd_party_translation else None
        
        ## read the ignored cases
        with open(self.path/'.dsignore', 'r') as file:
            self.dsignore = file.readlines()
            
        ## read the translate cases
        ### translate cases are such cases where the final_report.txt is in german.
        with open(self.path/'.dstranslate', 'r') as file:
            self.dstranslate = file.readlines()
            
        ## remove linebreaks
        self.dsignore = [p.removesuffix('\n') for p in self.dsignore]
        self.dstranslate = [p.removesuffix('\n') for p in self.dstranslate]
        
        ## filter patients according to dsignore
        self.patients = [p for p in os.listdir(self.path/'Patients') if p not in self.dsignore and p != 'template']
        
        ## set return type
        self.return_type=return_type
    
    def __len__(self): 
        return len(self.patients)
    
    def __getitem__(self, idx): 
        if self.return_type != 'prompt': return self._get_sample(self.patients[idx]) 
        else: return self.patients[idx], self._get_prompt(**self._get_sample(self.patients[idx]))
    
    def _get_sample(self, id):
        item = {}
        item['id'] = id
        for field in self.fields:
            with open(self.path/'Patients'/id/f"{field}.txt", "r") as f:
                if field != 'final_report': item[field] = f.read()
                else:
                    item[field] = self.to_english(f.read()) if id in self.dstranslate and self.enable_trans else f.read()
        return item
    
    def make_baseprompt(self):
        messages = []
        messages.append(self._get_sysprompt())
        for samp in self.dsignore:
            data = self._get_sample(samp)
            messages.append(self._get_prompt(**data))
            messages.append(self._get_response_prompt(**data))
        return messages
    
    def _get_prompt(self, id, specialty, history, findings, final_report):
        specialty_text = self.to_english(specialty) if self.lngdct(specialty) else specialty
        history_text   = self.to_english(history)   if self.lngdct(history)   else history
        findings_text  = self.to_english(findings)  if self.lngdct(findings)  else findings

        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Specialty: {specialty_text}\n\n"
                        f"History: {history_text}\n\n"
                        f"Findings: {findings_text}"
                    )
                }
            ]
        }
    
    def _get_sysprompt(self):
        return {
            "role": "system",
            "content": [
            {
                "type": "text",
                "text": r"""
                You are a subspecialty radiologist's assistant helping to write a second-opinion report. 
                The expert radiologist gives you all the information you need: specialty, history, and findings. 
                Write the report from the given inputs in markdown. Do not repeat the inputs. Stop after the Patient Summary section.
                Correct all spelling, grammar, and terminology errors in the input before writing the report.
                Be concise and direct, do not explain your reasoning, only provide the requested output.
                ---

                Section 1: Clinical Summary
                - Summarize the patient history, symptoms, risk factors, and relevant lab or test results.
                - List key clinical questions this report addresses.
                - Create a brief chronological timeline of relevant imaging and procedures.
                - List active problems in order of priority.

                Section 2: Technical Details
                - State the imaging modality and technique.
                - Note any limitations affecting interpretation.

                Section 3: Findings
                - Describe positive findings with measurements and location.
                - Explicitly state what was not found.
                - Note any comparison to prior imaging if available.

                Section 4: Differential Diagnosis
                - List the most likely diagnoses in order of probability.
                - For each, give a one-sentence evidence-based rationale.

                Section 5: Recommendations
                - Short-term next steps.
                - Long-term follow-up plan.
                - Suggested referrals if indicated.

                Section 6: Patient Summary
                - Explain the findings in plain language.
                - State the likelihood of benign vs. concerning findings.
                - Provide a simple action table: what to do, why, and when.
                - List 2 questions the patient should ask their doctor.
                """
            }
            ]
        }
        
    def _get_response_prompt(self, id, specialty, history, findings, final_report):
        return {
            "role": "assistant",
            "content": [
            {
                "type": "text",
                "text": final_report
            }
            ]
        }
        
    