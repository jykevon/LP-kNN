import pandas as pd

INFILE = "final.csv"
OUTFILE = "final_cut.csv"
PERCENT = 20  # e.g. 50 for 50%, 100 for full dataset
RANDOM_STATE = 42


df = pd.read_csv(INFILE)
classname = df.columns[-1]

if PERCENT <= 0 or PERCENT > 100:
    raise ValueError("PERCENT must be in (0, 100].")

frac = PERCENT / 100.0

if frac < 1.0:
    df = (
        df.groupby(classname, group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=RANDOM_STATE))
        .sample(frac=1.0, random_state=RANDOM_STATE)  # shuffle
        .reset_index(drop=True)
    )

print(f"Using {len(df)} samples ({PERCENT}%)")

df.to_csv(OUTFILE, index=False)
print(f"Saved: {OUTFILE}")
