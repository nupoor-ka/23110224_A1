import random
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
    """Return mean embedding for given text."""
    if not text or text.strip().lower() == "nan":
        return np.zeros((1,768))
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # return outputs.last_hidden_state.mean(dim=1).cpu().numpy() # mean pooling
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
    return None

def sanity_check(df, tokenizer, model, device="cpu", n=10):
    scores_real, scores_fake = [], []
    samples = []

    for _ in range(n):
        row = df.sample(1).iloc[0]
        diff = str(row["Diff"])
        real_msg = str(row["Message"])
        wrong_msg = "asiub buhefijx, iauhfhv."

        emb_diff = embed_text(diff, tokenizer, model, device)
        emb_real = embed_text(real_msg, tokenizer, model, device)
        emb_fake = embed_text(wrong_msg, tokenizer, model, device)

        real_score = cosine_similarity(emb_diff, emb_real)[0][0]
        fake_score = cosine_similarity(emb_diff, emb_fake)[0][0]

        scores_real.append(real_score)
        scores_fake.append(fake_score)
        samples.append((real_score, fake_score, real_msg[:60], wrong_msg[:60]))

    print("Mean REAL similarity:", np.mean(scores_real))
    print("Mean FAKE similarity:", np.mean(scores_fake))
    print("\nSample comparisons (real vs fake):")
    for r, f, rm, fm in samples[:5]:  # print first 5
        print(f"REAL={r:.4f}, FAKE={f:.4f} | RealMsg='{rm}...' | FakeMsg='{fm}...'")


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    sanity_check(df, tokenizer, model, "cpu")
