import os
from glob import glob
from pathlib import Path
from urllib.parse import quote
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from .entropy import calculate_base_entropy, calculate_tree_entropy
from .html.html_deprel import create_html


def calculate_metrics(
    language_data: Dict[str, Tuple[Pipeline, pd.DataFrame]],
    target_col: str = "deprel_order",
    binary_entropy: bool = False,
    smoothing: float = 0.5,
) -> pd.DataFrame:
    """Calculate metrics for all languages.

    Args:
        language_data: Dict mapping language_name -> (fitted_pipeline, dataframe)
        target_col: Column name containing word order labels
        binary_entropy: If True, use binary entropy. If False, use six-class.
        smoothing: Smoothing factor for entropy calculation

    Returns:
        DataFrame with columns: language, base_entropy, reduced_entropy,
                                delta_entropy, accuracy, n_items, n_flexible, n_fully_flexible, total_pairs
    """
    metrics = []

    for lang_name, (dt, df) in tqdm(language_data.items()):
        # Calculate entropies
        base_ent = calculate_base_entropy(
            df, target_col, binary=binary_entropy, smoothing=smoothing
        )
        reduced_ent = calculate_tree_entropy(
            dt, df, target_col, binary=binary_entropy, smoothing=smoothing
        )
        delta_ent = base_ent - reduced_ent

        # Calculate accuracy on full training data
        X = df.drop(columns=[target_col])
        y = df[target_col]
        accuracy = dt.score(X, y)

        # Number of items
        n_items = len(df)

        # Swap statistics (if num_swaps column exists)
        if "num_swaps" in df.columns:
            n_flexible = (df["num_swaps"] >= 1).sum()
            n_fully_flexible = (df["num_swaps"] == 4).sum()
        else:
            n_flexible = 0
            n_fully_flexible = 0

        metrics.append(
            {
                "language": lang_name,
                "base_entropy": base_ent,
                "reduced_entropy": reduced_ent,
                "delta_entropy": delta_ent,
                "accuracy": accuracy,
                "n_items": n_items,
                "n_flexible": n_flexible,
                "n_fully_flexible": n_fully_flexible,
                "total_pairs": sum(df["num_swaps"]),
            }
        )

    return pd.DataFrame(metrics).sort_values("language")


def generate_html_deprel_index(
    data_dir: str,
    html_directory: str,
    language_data: Dict[str, Tuple[Pipeline, pd.DataFrame]] | None = None,
    target_col: str = "deprel_order",
    smoothing: float = 0.5,
) -> None:
    """Generate interactive overview page with metrics and language links.

    Args:
        language_data: Dict mapping language_name -> (fitted_pipeline, dataframe)
        html_directory: Directory to save the index.html file
        target_col: Column name containing word order labels
        smoothing: Smoothing factor for entropy calculation
    """
    html_directory = Path(html_directory)

    if language_data is None:
        language_data = {}

        for model_fn in tqdm(glob(os.path.join(data_dir, "*.joblib"))):
            lang_name = Path(model_fn).stem
            csv_fn = os.path.join(data_dir, lang_name + ".csv")
            if not os.path.exists(csv_fn):
                continue
            language_data[lang_name] = (joblib.load(model_fn), pd.read_csv(csv_fn))

    # Calculate metrics for BOTH entropy types
    metrics_six = calculate_metrics(
        language_data, target_col, binary_entropy=False, smoothing=smoothing
    )
    metrics_binary = calculate_metrics(
        language_data, target_col, binary_entropy=True, smoothing=smoothing
    )

    # Find corresponding HTML files
    html_files = {
        f.stem: f
        for f in html_directory.glob("*.html")
        if f.name.lower() != "index.html"
    }
    deprel = html_directory.name

    # Generate table rows for both entropy types
    def generate_rows(metrics_df):
        rows = []
        for _, row in metrics_df.iterrows():
            lang_name = row["language"].replace("_", " ")
            lang_file = html_files.get(row["language"].replace(" ", "_"))

            if lang_file:
                lang_link = (
                    f'<a href="/multiblimp/{deprel}/{quote(lang_file.stem)}">{lang_name}</a>'
                )
            else:
                lang_link = lang_name

            rows.append(
                f"""
            <tr>
                <td data-sort="{lang_name.lower()}">{lang_link}</td>
                <td data-sort="{row['base_entropy']:.4f}">{row['base_entropy']:.3f}</td>
                <td data-sort="{row['reduced_entropy']:.4f}">{row['reduced_entropy']:.3f}</td>
                <td data-sort="{row['delta_entropy']:.4f}">{row['delta_entropy']:.3f}</td>
                <td data-sort="{row['accuracy']:.4f}">{row['accuracy']:.1%}</td>
                <td data-sort="{row['n_items']}">{row['n_items']:,}</td>
                <td data-sort="{row['n_flexible']}">{row['n_flexible']:,}</td>
                <td data-sort="{row['n_fully_flexible']}">{row['n_fully_flexible']:,}</td>
                <td data-sort="{row['total_pairs']}">{row['total_pairs']:,}</td>
            </tr>
            """
            )
        return "".join(rows)

    rows_six = generate_rows(metrics_six)
    rows_binary = generate_rows(metrics_binary)

    # Generate scatter plot data for both entropy types
    def generate_plot_data(metrics_df):
        data = []
        for _, row in metrics_df.iterrows():
            lang_name = row["language"].replace("_", " ")
            lang_file = html_files.get(row["language"].replace(" ", "_"))
            url = f"/multiblimp/{deprel}/{quote(lang_file.stem)}" if lang_file else None

            data.append(
                {
                    "name": lang_name,
                    "base": row["base_entropy"],
                    "reduced": row["reduced_entropy"],
                    "n_items": int(row["n_items"]),
                    "url": url,
                }
            )
        return data

    plot_data_six = generate_plot_data(metrics_six)
    plot_data_binary = generate_plot_data(metrics_binary)

    # Convert to JSON for JavaScript
    import json

    plot_data_six_json = json.dumps(plot_data_six)
    plot_data_binary_json = json.dumps(plot_data_binary)

    html_content = create_html(rows_six, rows_binary, plot_data_six_json, plot_data_binary_json)
    output_path = html_directory / "index.html"
    output_path.write_text(html_content, encoding="utf-8")
