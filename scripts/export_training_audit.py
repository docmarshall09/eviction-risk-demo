"""Export yearly training-row audit files for the Eviction Lab logistic model.

Run:
    python scripts/export_training_audit.py --outdir local_exports
"""

import argparse
from pathlib import Path
import sys


def _add_project_root_to_python_path() -> None:
    """Ensure the repository root is on sys.path for `python scripts/...` runs."""
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    sys.path.insert(0, str(project_root))


_add_project_root_to_python_path()

from src.main import _load_or_build_yearly_feature_table  # noqa: E402
from src.pipelines.yearly_training_dataset import (  # noqa: E402
    build_yearly_training_dataset_with_audit,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training-audit export command."""
    parser = argparse.ArgumentParser(
        description="Export yearly training row/filter audit CSVs."
    )
    parser.add_argument(
        "--outdir",
        default="local_exports",
        help="Directory where audit CSV files will be written.",
    )
    return parser.parse_args()


def export_training_audit(outdir: Path) -> None:
    """Build and export training row/filter audit artifacts.

    Args:
        outdir: Output directory for audit CSV files.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    feature_df = _load_or_build_yearly_feature_table()
    training_df, row_audit_df, filter_counts_df = build_yearly_training_dataset_with_audit(
        feature_df
    )

    row_audit_path = outdir / "training_row_audit.csv"
    filter_counts_path = outdir / "training_filter_counts.csv"

    row_audit_export_df = row_audit_df[
        ["county_fips", "target_year", "as_of_year", "kept", "drop_reason"]
    ].copy()
    row_audit_export_df.to_csv(row_audit_path, index=False)

    filter_counts_export_df = filter_counts_df[
        ["step_name", "rows_remaining", "rows_dropped", "drop_reason"]
    ].copy()
    filter_counts_export_df.to_csv(filter_counts_path, index=False)

    print(f"Wrote row-level audit: {row_audit_path}")
    print(f"Wrote filter-count audit: {filter_counts_path}")
    print(f"Final training rows kept: {len(training_df)}")


def main() -> None:
    """Run the training-audit export command."""
    args = parse_args()
    export_training_audit(Path(args.outdir))


if __name__ == "__main__":
    main()
