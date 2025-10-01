import pandas as pd
from pathlib import Path
import sys

ODS_FILE = Path('logP_calculation.ods')
OUTPUT_CSV = Path('logP_data_from_ods.csv')
ENRICHED_OUTPUT = Path('logP_data_from_ods_enriched.csv')

BASELINE_A = 0.8645
BASELINE_B = -0.1688


def detect_and_extract(df):
    # Known pattern: Row 0 group labels, Row 1 detailed labels (MNSol_id, Name, Water, Octanol, logP, Water, Octanol, logP, NaN, NaN)
    # Data starts at row 2. We'll take first 8 columns explicitly.
    if df.shape[0] < 4:
        raise ValueError("Sheet too small")
    data = df.iloc[2:, :8].copy()
    data.columns = ['MNSol_id','Name','BAR_Water','BAR_Octanol','BAR_logP','Exp_Water','Exp_Octanol','Exp_logP']
    return data


def normalize_columns(body: pd.DataFrame) -> pd.DataFrame:
    keep = ['MNSol_id','Name','BAR_Water','BAR_Octanol','BAR_logP','Exp_Water','Exp_Octanol','Exp_logP']
    missing = [c for c in keep if c not in body.columns]
    if missing:
        raise ValueError(f"Missing expected columns after detect: {missing}")
    body = body[keep].copy()
    # Clean types
    for c in ['MNSol_id','Name']:
        body[c] = body[c].astype(str).str.strip().str.replace('"','', regex=False)
    num_cols = [c for c in keep if c not in ('MNSol_id','Name')]
    for c in num_cols:
        body[c] = pd.to_numeric(body[c], errors='coerce')
    body.dropna(how='all', subset=num_cols, inplace=True)
    return body


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['PolarityDiff'] = (df['BAR_Water'] - df['BAR_Octanol']).abs()
    maxd = df['PolarityDiff'].max()
    df['PolarityProxy'] = df['PolarityDiff'] / maxd if maxd else 0.0
    df['D1_BAR_logP'] = df['BAR_logP']
    df['D2_PolarityProxy'] = df['PolarityProxy']
    df['Baseline_Pred'] = BASELINE_A * df['BAR_logP'] + BASELINE_B
    df['Baseline_Residual'] = df['Exp_logP'] - df['Baseline_Pred']
    order = [
        'MNSol_id','Name','BAR_Water','BAR_Octanol','BAR_logP',
        'Exp_Water','Exp_Octanol','Exp_logP','PolarityDiff','PolarityProxy',
        'D1_BAR_logP','D2_PolarityProxy','Baseline_Pred','Baseline_Residual'
    ]
    return df[order]


def main():
    if not ODS_FILE.exists():
        print(f"ODS file not found: {ODS_FILE}", file=sys.stderr)
        sys.exit(1)
    try:
        raw = pd.read_excel(ODS_FILE, engine='odf', sheet_name=0, header=None)
    except Exception as e:
        print(f"Failed reading ODS: {e}", file=sys.stderr)
        sys.exit(2)

    body = detect_and_extract(raw)
    core = normalize_columns(body)
    core.to_csv(OUTPUT_CSV, index=False)
    enriched = enrich(core)
    enriched.to_csv(ENRICHED_OUTPUT, index=False)
    print(f"Wrote {len(core)} core rows -> {OUTPUT_CSV}")
    print(f"Wrote enriched file with {enriched.shape[1]} columns -> {ENRICHED_OUTPUT}")
    print(enriched.head(8).to_string())


if __name__ == '__main__':
    main()
