
import argparse, sys
from src.inference.inference import infer
  
#------------------------------------------------ MAIN PARSER ------------------------------------------------#
p = argparse.ArgumentParser()
p.add_argument("-model", default="medgemma", help="which model to use")
p.add_argument("-dir", default="outputs/run", help="A directory where the prompt as well as the model output are stored.")

if __name__ == "__main__":
    args = p.parse_args()
    sys.exit(infer(args))