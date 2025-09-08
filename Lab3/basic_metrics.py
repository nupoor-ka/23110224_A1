import pandas as pd
import csv
import sys

from radon.raw import analyze # lines of code
from radon.complexity import cc_visit # cyclomatic complexity
from radon.metrics import mi_visit # maintainability index
import ast # for parsing, get truncated error free code

import torch
from transformers import AutoTokenizer, AutoModel

import sacrebleu
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

# Hash,Message,Filename,Source Code (before),Source Code (current),Diff,LLM Inference (fix type),Rectified Message
hexsha = "Hash"
dev_commit = "Message"
file = "Filename"
code_before = "Source Code (before)"
code_after = "Source Code (current)"
diff = "Diff"
llm = "LLM Inference (fix type)"
rectified = "Rectified Message"
cols = [hexsha, dev_commit, file, code_before, code_after, diff, llm, rectified]

# new cols, to-do
mi_del = "MI_Change"
cc_del = "CC_Change"
loc_del = "LOC_Change"
sem_sim = "Semantic_Similarity"
token_sim = "Token_Similarity"
sem_class = "Semantic_Class"
token_class = "Token_Class"
agree = "Classes_Agree"

csv.field_size_limit(sys.maxsize) # to ensure that this file opens even thought the entries go beyond the normal limit for loading csv

skipped_rows = [] # recording rows that weren't processed while loading csv
failed_rows = [] # recording rows that caused errors with radon, sacrebleu or codebert

def bad_line(line): # row which had weird formatting
    skipped_rows.append(line)
    return None

def to_text(value): # if NaN or Noe, return "", else return stripped str val
    if pd.isna(value):
        return ""
    return str(value).replace("\\n", "\n").replace("\\r", "\r")

def mi_cc_loc(code): # mi,cc,loc
    try:
        if not isinstance(code, str) or code.strip() == "":
            return None, None, None
        mi = mi_visit(code, True)
        cc_scores = cc_visit(code) # list of blocks
        avg_cc = sum(c.complexity for c in cc_scores) / len(cc_scores) if cc_scores else 0
        # loc, lloc, sloc, comments, multi, blank = analyze(code)
        tup = analyze(code)
        return mi, avg_cc, tup.loc
    except Exception as e:
        print(e)
        return None, None, None

def get_embedding(text):
    if text.strip() == "":
        hidden_size = model.config.hidden_size
        return torch.zeros(1, hidden_size)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1) # uses cls by default, wasn't great, hence using mean pooling here
    return embeddings

def cosine_similarity(vec1, vec2):
    return torch.nn.functional.cosine_similarity(vec1, vec2).item()

def codebert(code_bef, code_aft):
    emb_before = get_embedding(code_bef)
    emb_after = get_embedding(code_aft)
    sim = cosine_similarity(emb_before, emb_after)
    return sim

def comp_bleu(code_bef, code_aft):
    bleu = (sacrebleu.corpus_bleu([code_aft], [[code_bef]]).score)/100.0
    return bleu

df = pd.read_csv("bug_fixes_complete.csv", engine="python", on_bad_lines=bad_line) # bad line if too many entries in row

print(f"{len(skipped_rows)} rows skipped due to initial parsing errors")

num_commit_file_pairs = df.shape[0]
num_commits = df[hexsha].nunique()
num_files = df[file].nunique()
avg_files_commit = df.groupby(hexsha).size().mean()
file_freq = df['Filename'].value_counts() # returns a pd series
ext_freq = df['Filename'].str.split('.').str[-1].value_counts()
llm_fix_types =  df[llm].str.split().str[0].value_counts() # the fix type seems to be the first word

print("Baseline descriptive statistics:")
print("Number of commits =", num_commits)
print("Number of commit-file pairs =", num_commit_file_pairs)
print("Number of files =", num_files)
print("Average number of modified files per commit =", avg_files_commit)
print("\nDistribution of fix types from LLM Inference (fix type):")
print(llm_fix_types)
print("\nMost frequently modified files:")
print(file_freq)
print("\nDistribution of extensions of modified files:")
print(ext_freq)

op_file = "bug_fix_commits_metrics.csv"

# the process was taking a minute a row, so I sampled 120 rows from the total 509, that also took more than 3 hrs
# the reason for not working on colab or kaggle notebooks was that they weren't handling these large csvs well
# they have limited memory which is used up when loading models for inference

idx_processed = []
idx_not_processed = []
for idx, row in df.iterrows():
    if idx%40==0:
        print(f"{idx} rows processed")
        temp_file = f"checkpoint_{idx}.csv"
        df.to_csv(temp_file, index=False)
        print(f"Checkpoint saved at row {idx} -> {temp_file}")
    if len(idx_processed)==120:
        break
    code_bef = to_text(row[code_before])
    code_aft = to_text(row[code_after])
    mi_bef, cc_bef, loc_bef = mi_cc_loc(code_bef)
    mi_aft, cc_aft, loc_aft = mi_cc_loc(code_aft)
    if mi_bef is None or mi_aft is None:
        failed_rows.append(f"{idx}, radon")
        print(f"Error in row {idx}")
        df.loc[idx,cc_del] = None
        df.loc[idx,mi_del] = None
        df.loc[idx,loc_del] = None
        idx_not_processed.append(idx)
        continue
    else:
        df.loc[idx,cc_del] = cc_aft-cc_bef
        df.loc[idx,mi_del] = mi_aft-mi_bef
        df.loc[idx,loc_del] = loc_aft-loc_bef
        idx_processed.append(idx)
    semanticsimilarity = codebert(code_bef, code_aft)
    tokensimilarity = comp_bleu(code_bef, code_aft)
    df.loc[idx,sem_sim] = semanticsimilarity
    df.loc[idx,token_sim] = tokensimilarity
    if semanticsimilarity is not None and semanticsimilarity>=0.8:
        df.loc[idx,sem_class] = "Minor"
    else:
        df.loc[idx,sem_class] = "Major"
    if tokensimilarity is not None and tokensimilarity>=0.75:
        df.loc[idx,token_class] = "Minor"
    else:
        df.loc[idx,token_class] = "Major"
    if df.loc[idx,token_class]is not None and df.loc[idx,sem_class]is not None and df.loc[idx,token_class]==df.loc[idx,sem_class]:
        df.loc[idx,agree]="YES"
    else:
        df.loc[idx,agree]="NO"
    print(f"row {idx} processed")

df.to_csv(op_file, index=False)
pd.DataFrame(failed_rows).to_csv("failed_rows.csv", index=False)
print(f"{len(failed_rows)} rows had errors with radon")
print(f"Saved to {op_file}")
print(f"processed {len(idx_processed)} rows")
print(idx_processed)
print(f"unable to resolve code from csv entry for {len(idx_not_processed)} rows")
print(idx_not_processed)

# some code to get the first 10 rows, everything truncated to 100 chars
pd.set_option("display.max_columns", None) # all cols
pd.set_option("display.width", 200) # wider console output
pd.set_option("display.max_colwidth", 100) # truncate cell contents to 100 chars for display
first5_idx = idx_processed[:10]
print(df.loc[first5_idx])
