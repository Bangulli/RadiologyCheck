from data.PARSOR_FIXED.dataloader import PARSORData
from src.inference.func import infer_eval, infer, eval
import transformers
from transformers.utils.logging import disable_progress_bar
transformers.logging.set_verbosity_error()
disable_progress_bar()

ds = PARSORData()
print(len(ds))

output = 'data/outputs/experiment_1'


eval(ds, output)