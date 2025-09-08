# add one line to quantify the enormity of the diff, support argument for sampling 120
# can measure correlation btwn semantic and token sims
# counts of yes and no in classes_agree, counts of major and minor in semantic and token similarities

import pandas as pd
import csv
import sys

skipped_rows = []

def bad_line(line):
    skipped_rows.append(line)
    return None

metrics = ["MI_Change", "CC_Change", "LOC_Change", "Semantic_Similarity", "Token_Similarity"]
idx_processed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                 18, 19, 20, 21, 22, 26, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 57, 58, 60, 63, 64,
                 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                 82, 83, 84, 99, 113, 114, 115, 148, 150, 204, 215, 244, 259, 292,
                 293, 295, 306, 402, 413, 414, 415, 416, 417, 418, 419, 420, 425,
                 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438,
                 439, 440, 442, 443, 444, 445, 446, 447, 448, 449, 450, 452, 453,
                 454] # these were the indices processed, using only these

csv.field_size_limit(sys.maxsize)

df = pd.read_csv("bug_fix_commits_metrics.csv", engine="python", on_bad_lines=bad_line)
df_proc = df.loc[idx_processed]

print("Mean values:")
print(df_proc[metrics].mean(), "\n")

print("Median values:")
print(df_proc[metrics].median(), "\n")

print("Standard deviation:")
print(df_proc[metrics].std(), "\n")

print("Enormity of Diff (Range)")
for m in ["MI_Change", "CC_Change", "LOC_Change"]:
    print(f"{m}: min={df_proc[m].min()}, max={df_proc[m].max()}, range={df_proc[m].max()-df_proc[m].min()}")
print()

print("Correlation between similarities")
print(df_proc[["Semantic_Similarity", "Token_Similarity"]].corr(), "\n")

print("Counts in categorical columns")
print("Classes_Agree counts:")
print(df_proc["Classes_Agree"].value_counts(), "\n")

print("Semantic_Class counts:")
print(df_proc["Semantic_Class"].value_counts(), "\n")

print("Token_Class counts:")
print(df_proc["Token_Class"].value_counts(), "\n")

print("Max char length of source code columns")
before_max = df_proc["Source Code (before)"].str.len().max()
current_max = df_proc["Source Code (current)"].str.len().max()

print(f"Max length in 'Source Code (before)': {before_max} chars")
print(f"Max length in 'Source Code (current)': {current_max} chars")
