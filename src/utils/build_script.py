from huggingface_hub import snapshot_download
import argparse, sys, json

p = argparse.ArgumentParser()
p.add_argument("-tok", required=True)

def dl(args):
    with open("config.json", "r") as f:
        cfg = json.load(f)
    snapshot_download(cfg['tran_id'],  token=args.tok, local_dir=cfg['tran'])
    snapshot_download(cfg['inf_id'], token=args.tok, local_dir=cfg['inf'])

if __name__ == "__main__":
    args = p.parse_args()
    sys.exit(dl(args))
    
