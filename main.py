
import argparse, sys
from src.inference.inference import infer
  
#------------------------------------------------ MAIN PARSER ------------------------------------------------#
p = argparse.ArgumentParser()
p.add_argument("-dir", default="outputs/run", help="A directory where the prompt as well as the model output are stored.")
p.add_argument("--resume", action="store_true", help="When used will load the prompt from the specified directory and resume the refining loop.")

if __name__ == "__main__":
    args = p.parse_args()
    sys.exit(infer(args))