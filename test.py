from data.PARSOR_FIXED.dataloader import PARSORData
from src.inference.func import infer, eval
import transformers, json
from transformers.utils.logging import disable_progress_bar
transformers.logging.set_verbosity_error()
disable_progress_bar()

ds = PARSORData(enable_3rd_party_translation=False)
print(len(ds))
with open('/home/lorenz/RadiologyCheck/v3_fewshot_baseprompt.json', 'w') as f:
    json.dump(ds.make_baseprompt(), f, indent=4)

output = 'data/outputs/experiment_3'

infer(ds, output)
eval(ds, output)
