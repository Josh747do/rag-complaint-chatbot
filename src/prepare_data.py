import argparse
from pathlib import Path
import pandas as pd


def main(raw_path: Path, out_path: Path, top_k: int):
    df = pd.read_csv(raw_path, low_memory=False)

    print(f"Loaded {len(df):,} rows from {raw_path}")

    # Basic EDA summaries
    print("\nColumn names:")
    print(list(df.columns))

    if 'product' not in df.columns:
        raise RuntimeError("Expected column 'product' in the dataset")

    product_counts = df['product'].value_counts()
    print('\nTop product counts:')
    print(product_counts.head(20))

    # Narrative length distribution
    if 'consumer_complaint_narrative' in df.columns:
        lengths = df['consumer_complaint_narrative'].dropna().astype(str).str.len()
        print('\nConsumer narrative length describe:')
        print(lengths.describe())
    else:
        print('\nNo `consumer_complaint_narrative` column found.')

    # Choose top_k products by frequency
    top_products = product_counts.index[:top_k].tolist()
    print(f"\nFiltering to top {top_k} products: {top_products}")

    # Filter and clean
    df_filtered = df[df['product'].isin(top_products)].copy()

    # Drop rows without narrative
    if 'consumer_complaint_narrative' in df_filtered.columns:
        before = len(df_filtered)
        df_filtered = df_filtered[df_filtered['consumer_complaint_narrative'].notna()]
        after = len(df_filtered)
        print(f"Dropped {before-after:,} rows without narrative; {after:,} rows remain.")

    # Keep a useful subset of columns when available
    preferred_cols = [
        'complaint_id', 'date_received', 'product', 'sub_product', 'issue', 'sub_issue',
        'company', 'state', 'zip_code', 'consumer_consent_provided', 'consumer_complaint_narrative'
    ]

    cols = [c for c in preferred_cols if c in df_filtered.columns]
    df_out = df_filtered[cols].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"Saved filtered dataset to {out_path} with {len(df_out):,} rows")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare and filter CFPB complaint data')
    parser.add_argument('--raw', type=Path, default=Path('data/raw/complaints.csv'))
    parser.add_argument('--out', type=Path, default=Path('data/filtered_complaints.csv'))
    parser.add_argument('--top-k', type=int, default=5, help='Number of top products to keep')
    args = parser.parse_args()

    main(args.raw, args.out, args.top_k)
