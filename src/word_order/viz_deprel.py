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
        # ensure dtype matching of loaded data and classifier
        cat_cols = [col for name, _, cols in dt.named_steps["preprocessor"].transformers_ if name == "cat"
                    for col in cols if col in df.columns]
        num_cols = [col for name, _, cols in dt.named_steps["preprocessor"].transformers_ if name == "num"
                    for col in cols if col in df.columns]
        df[cat_cols] = df[cat_cols].astype(str)
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
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

def scatter_color(metrics_row, language_data, target_col, exclude_labels=None):
    """Yellow if all labels in , blue otherwise."""
    if exclude_labels:
        lang_name = metrics_row["language"]
        if lang_name not in language_data:
            return "#2563eb"
        _, df = language_data[lang_name]
        labels = set(df[target_col].unique())
        if labels <= {"--", "+-"}:
            return "#e5c64d"
    return "#2563eb"

def extract_trivial_label(html_path: Path) -> str | None:
    """Extract the sole label from a placeholder HTML file, or None if it's a real tree page."""
    content = html_path.read_text(encoding="utf-8")
    match = re.search(r'<div class="label">(.+?)</div>', content)
    if match and "No decision tree to display" in content:
        return match.group(1).strip()
    return None

def generate_html_deprel_index(
    data_dir: str,
    html_directory: str,
    language_data: Dict[str, Tuple[Pipeline, pd.DataFrame]] | None = None,
    target_col: str = "deprel_order",
    smoothing: float = 0.5,
    exclude_labels=None,
) -> None:
    """Generate interactive overview page with metrics and language links.

    Args:
        language_data: Dict mapping language_name -> (fitted_pipeline, dataframe)
        html_directory: Directory to save the index.html file
        target_col: Column name containing word order labels
        smoothing: Smoothing factor for entropy calculation
    """
    html_directory = Path(html_directory)

    trivial_langs = {}
    for html_file in html_directory.glob("*.html"):
        if not html_file.name.lower() == "index.html":
            label = extract_trivial_label(html_file)
            if label is not None:
                trivial_langs[html_file.stem] = label

    if language_data is None:
        language_data = {}

        for model_fn in tqdm(glob(os.path.join(data_dir, "*.joblib"))):
            lang_name = Path(model_fn).stem
            if lang_name in trivial_langs:
                continue
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

    for l in metrics_six[metrics_six["base_entropy"] == 0.0]["language"]:
        trivial_langs[l] =set(language_data[l][target_col].values)[0] # if placeholders dont exist yet

    metrics_six    = metrics_six[~metrics_six["language"].isin(trivial_langs.keys())]
    metrics_binary = metrics_binary[~metrics_binary["language"].isin(trivial_langs.keys())]

    # Find corresponding HTML files
    html_files = {
        f.stem: f
        for f in html_directory.glob("*.html")
        if f.name.lower() != "index.html"
    }
    deprel = html_directory.name

    # Generate table rows for both entropy types
    def generate_rows(metrics_df, lang_colors):
        rows = []
        for _, row in metrics_df.iterrows():
            lang_name = row["language"].replace("_", " ")
            lang_file = html_files.get(row["language"].replace(" ", "_"))
            color = lang_colors.get(lang_name, "#2563eb")          # ← was "#1c1917"
            name_style = f' style="color:{color};font-weight:600;"' if color != "#2563eb" else ""

            if lang_file:
                lang_link = (
                    f'<a href="/multiblimp/{deprel}/{quote(lang_file.stem)}">{lang_name}</a>'
                    f'<a href="/multiblimp/{deprel}/{quote(lang_file.stem)}"{name_style}>{lang_name}</a>'
                )
            else:
                lang_link = f'<span{name_style}>{lang_name}</span>'

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

    lang_colors = {
        row["language"].replace("_", " "): scatter_color(row, language_data, target_col, exclude_labels=exclude_labels)
        for _, row in metrics_six.iterrows()
    }
    rows_six = generate_rows(metrics_six, lang_colors)
    rows_binary = generate_rows(metrics_binary, lang_colors)

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
                    "color": lang_colors.get(lang_name, "#2563eb"),
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
    if trivial_langs:
        skipped_links = []
        for lang, pred_tag in sorted(trivial_langs.items()):
            lang_display = lang.replace("_", " ")
            lang_file = html_files.get(lang.replace(" ", "_"))
            if lang_file:
                url = f"/multiblimp/{deprel}/{quote(lang_file.stem)}"
                skipped_links.append(f'<a href="{url}">{lang_display}</a> ({pred_tag})')
            else:
                skipped_links.append(f'{lang_display} ({pred_tag})')

        trivial_note = (
            f'<p class="trivial-note">The following languages were omitted because '
            f'all samples share a single agreement label: {", ".join(skipped_links)}.</p>'
        )
    else:
        trivial_note = ""

    html_content = create_html(rows_six, rows_binary, plot_data_six_json, plot_data_binary_json,
                               trivial_note=trivial_note)
    output_path = html_directory / "index.html"
    output_path.write_text(html_content, encoding="utf-8")
