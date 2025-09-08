import pandas as pd
import csv
import sys

csv.field_size_limit(sys.maxsize)

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

    # configure pandas display
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 100)  # truncate to 100 chars

    print("\nSample rows (truncated to 100 chars per field):")
    print(df.head(5))

    # test hashes to filter
    test_hashes = {
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
    }

    df_filtered = df[df["Hash"].isin(test_hashes)]

    print(f"\nFound {len(df_filtered)} rows matching test hashes.\n")

    for _, row in df_filtered.iterrows():
        print("="*80)
        print(f"Hash: {row['Hash']}")
        print(f"Filename: {row['Filename']}")
        diff = str(row['Diff']) if pd.notna(row['Diff']) else ""
        print(f"Diff (truncated): {diff[:300]}{'...' if len(diff) > 300 else ''}\n")
        print(f"Commit message 1 : {row['Message']}")
        print(f"Commit message 2 : {row['LLM Inference (fix type)']}")
        print(f"Commit message 3 : {row['Rectified Message']}")
        print()
