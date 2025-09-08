import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import csv
import sys

csv.field_size_limit(sys.maxsize)

MODEL_NAME = "microsoft/codebert-base"

def embed_text(text, tokenizer, model, device="cpu"):
    """Return CLS embedding for given text."""
    if not text or text.strip() == "nan":
        return np.zeros((1,768))
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:,0,:].cpu().numpy()  # CLS token

def process_rows(df, test_hashes, tokenizer, model, device="cpu", checkpoint_every=25):
    results = []  # instead of modifying df
    out_path = "bug_fixes_with_scores.csv"

    for counter, (idx, row) in enumerate(df.iterrows()):
        diff = str(row["Diff"]) if pd.notna(row["Diff"]) else ""
        dev_msg = str(row["Message"])
        llm_msg = str(row["LLM Inference (fix type)"])
        rect_msg = str(row["Rectified Message"])
        commit_hash = str(row["Hash"])

        emb_diff = embed_text(diff, tokenizer, model, device)
        emb_dev  = embed_text(dev_msg, tokenizer, model, device)
        emb_llm  = embed_text(llm_msg, tokenizer, model, device)
        emb_rect = embed_text(rect_msg, tokenizer, model, device)

        score_dev  = cosine_similarity(emb_dev, emb_diff)[0][0]
        score_llm  = cosine_similarity(emb_llm, emb_diff)[0][0]
        score_rect = cosine_similarity(emb_rect, emb_diff)[0][0]

        results.append((commit_hash, score_dev, score_llm, score_rect))

        if (counter+1) % checkpoint_every == 0:
            pd.DataFrame(results, columns=["Hash", "DevScore", "LLMScore", "RectScore"])\
              .to_csv(out_path, index=False)
            print(f"Checkpoint saved {counter+1} rows to {out_path}")

    df_out = pd.DataFrame(results, columns=["Hash", "DevScore", "LLMScore", "RectScore"])
    df_out.to_csv(out_path, index=False)

    df_test = df_out[df_out["Hash"].isin(test_hashes)]
    df_test.to_csv("bug_fixes_test_hashes.csv", index=False)

    print("Final saved:", out_path)
    print("Test subset saved: bug_fixes_test_hashes.csv")
    print("Developer mean score:", np.mean(df_out["DevScore"]))
    print("LLM mean score:", np.mean(df_out["LLMScore"]))
    print("Rectifier mean score:", np.mean(df_out["RectScore"]))

skipped_rows = []

def bad_line_handler(line):
    skipped_rows.append(line)
    return None  # skip this row

if __name__ == "__main__":
    df = pd.read_csv(
        "bug_fixes_complete.csv",
        engine="python",
        on_bad_lines=bad_line_handler
    )

    print(f"Loaded {len(df)} rows")
    print(f"Skipped {len(skipped_rows)} bad rows")
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 100)

    print("\nSample rows:")
    print(df.head(5))

    test_hashes = { # need to sample a few to see
        "48d71b95f22ab9141e28c711fc644cde2a69a809",
        "47c21316bd41f39a0a0d65b0164adf1473570565",
        "d1d23f62ec0f84a55efed15c43e33666d92bca95",
        "c41bed2b5bc41ecd9ef62c4e9ddfaae51a819660",
        "b6c61c42c1ebec11d6b3eab9838761b471744fc0",
        "ebb9be584e22052f87704963d84e0141b0ac231e",
        "d77f8f8c876884e6784e745378d7f39f8542d7af",
        "399384c818ec4f75e4a5e218466b1c087ec7e6c5",
        "bef24aa15c72fd2955f9fc540b97fb0f78a93293",
        "10a168d19c71445c77b86a8f272e52690c3210c1"
    } # same 10 as human and LLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    process_rows(df, test_hashes, tokenizer, model, device=device)
