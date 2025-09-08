from pydriller import Repository
import pandas as pd
import matplotlib.pyplot as plt

def processing_myers(diff_str): # myers output is a raw string
    added, deleted = [], []
    for line in diff_str.splitlines():
        if line.startswith("+") and not line.startswith("++"):
            added.append(line[1:].strip())
        elif line.startswith("-") and not line.startswith("--"):
            deleted.append(line[1:].strip())
    return added, deleted


def discrepancy(row):
    add_myers, del_myers = processing_myers(row["diff_myers"])
    try:
        hist_dict = eval(row["diff_hist"]) # get a dictionary from hist
    except Exception:
        return "Error"
    add_hist = [line.strip() for _, line in hist_dict.get("added", [])]
    del_hist = [line.strip() for _, line in hist_dict.get("deleted", [])]
    if set(add_myers) == set(add_hist) and set(del_myers) == set(del_hist):
        return "Yes"
    else:
        return "No"

def file_type(path):
    if path is None:
        return "Unknown"
    if "test" in path.lower():
        return "Test"
    elif "readme" in path.lower():
        return "README"
    elif "license" in path.lower():
        return "LICENSE"
    else:
        return "Source"

repos = ["javascript-algorithms", "openai-cookbook", "ventoy"]

data = []

for repo in repos:
    for commit in Repository(repo).traverse_commits():
        for mod in commit.modified_files:
            diff_myers = mod.diff.replace(" ", "").replace("\n\n", "\n")
            diff_hist = mod.diff_parsed
            
            data.append({
                "old_file_path": mod.old_path,
                "new_file_path": mod.new_path,
                "commit_SHA": commit.hash,
                "parent_commit_SHA": commit.parents[0] if commit.parents else None,
                "commit_message": commit.msg,
                "diff_myers": diff_myers,
                "diff_hist": str(diff_hist)
            })

df = pd.DataFrame(data)
df.to_csv("consolidated_dataset.csv", index=False)
df["discrepancy"] = df.apply(discrepancy, axis=1)
df.to_csv("final_dataset.csv", index=False)

pd.set_option("display.max_columns", None) # all cols
pd.set_option("display.width", 200) # wider console output
pd.set_option("display.max_colwidth", 100) # truncate cell contents to 100 chars for display
print(df.head(5))

df["file_type"] = df["new_file_path"].apply(file_type)

mismatch_counts = df[df["discrepancy"] == "No"]["file_type"].value_counts()
print("Mismatches by file type: ")
print(mismatch_counts)
print()
match_counts = df[df["discrepancy"] == "Yes"]["file_type"].value_counts()
print("Matches by file type: ")
print(match_counts)

plt.figure(figsize=(8, 5))
mismatch_counts.plot(kind="bar", title="Mismatches by File Type")
plt.xlabel("File Type")
plt.ylabel("Number of Mismatches")
plt.tight_layout()

plt.savefig("mismatches_plot.png")
print("Plot saved as mismatches_plot.png")
