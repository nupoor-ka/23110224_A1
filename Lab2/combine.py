import pandas as pd

bug_commits = pd.read_csv("bug_commits_rectified.csv")
rectified_commits = pd.read_csv("rectified_commits.csv")

merged = pd.merge(
    bug_commits,
    rectified_commits[["Hash", "Filename", "Rectified Message"]],
    on=["Hash", "Filename"],
    how="inner"
)

final = merged[
    [
        "Hash",
        "Message",
        "Filename",
        "Source Code (before)",
        "Source Code (current)",
        "Diff",
        "LLM Inference (fix type)",
        "Rectified Message"
    ]
]

final.to_csv("bug_fixes_complete.csv", index=False)
print("bug_fixes_complete.csv written with", len(final), "rows")
