#!/usr/bin/env python3
# SRL to JSON (frames + frame elements)

import argparse, json, sys
from typing import Dict
import nltk
from nltk.tokenize import sent_tokenize
from frame_semantic_transformer import FrameSemanticTransformer

def ensure_nltk():
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

def extract_frames(text: str, model_size: str = "base") -> Dict:
    fst = FrameSemanticTransformer(model_size)
    sents = sent_tokenize(text)
    out = {"sentences": []}
    for i, sent in enumerate(sents):
        res = fst.detect_frames(sent)
        frames = []
        for fr in res.frames:
            fes = [{"name": fe.name, "text": fe.text.strip()} for fe in fr.frame_elements]
            frames.append({"name": fr.name, "elements": fes})
        out["sentences"].append({"index": i, "text": sent, "frames": frames})
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--infile", type=str, help="Path to raw text file")
    p.add_argument("--text", type=str, help="Raw text (alternative to --infile)")
    p.add_argument("--outfile", type=str, required=True, help="Where to write JSON")
    p.add_argument("--model", type=str, default="base", choices=["small","base"])
    args = p.parse_args()

    ensure_nltk()

    if args.text is None and args.infile is None:
        print("Provide --text or --infile", file=sys.stderr); sys.exit(2)
    if args.text is not None and args.infile is not None:
        print("Provide only one of --text or --infile", file=sys.stderr); sys.exit(2)

    raw = args.text if args.text is not None else open(args.infile, "r", encoding="utf-8", errors="ignore").read()
    data = extract_frames(raw, model_size=args.model)
    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
