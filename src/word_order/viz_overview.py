from pathlib import Path
from urllib.parse import quote
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import math
from sklearn.pipeline import Pipeline
from tqdm import tqdm


def order_entropy(
        n_ab: int,
        n_ba: int,
        smoothing_a: float = 0.5,  # Jeffreys prior
) -> float:
    """
    Compute Shannon entropy (in bits) for word order frequencies.

    Args:
        n_ab (int): Count of A>B order
        n_ba (int): Count of B>A order
        smoothing_a (float): Smoothing factor

    Returns:
        float: Entropy in bits
    """
    n_ab += smoothing_a
    n_ba += smoothing_a

    total = n_ab + n_ba
    if total == 0:
        return 0.0

    p_ab = n_ab / total
    p_ba = n_ba / total

    # avoid log(0) issues
    def safe_term(p):
        return -p * math.log2(p) if p > 0 else 0.0

    return safe_term(p_ab) + safe_term(p_ba)


def calculate_base_entropy(df: pd.DataFrame, target_col: str, binary: bool = False, smoothing: float = 0.5) -> float:
    """Calculate entropy of word order distribution.

    Args:
        df: DataFrame containing the data
        target_col: Column name containing word order labels
        binary: If True, calculate binary entropy (majority-class vs rest).
                If False, calculate six-class entropy.
        smoothing: Smoothing factor for entropy calculation (Jeffreys prior)

    Returns:
        Entropy value in bits
    """
    value_counts = df[target_col].value_counts()

    if binary:
        # Binary entropy: majority class vs. rest
        n_majority = value_counts.iloc[0]  # Most frequent class
        n_rest = len(df) - n_majority
        return order_entropy(n_majority, n_rest, smoothing_a=smoothing)
    else:
        # Six-class entropy with smoothing
        counts = value_counts.values
        smoothed_counts = counts + smoothing
        total = smoothed_counts.sum()
        probabilities = smoothed_counts / total

        return sum(-p * math.log2(p) if p > 0 else 0.0 for p in probabilities)


def calculate_tree_entropy(dt: Pipeline, df: pd.DataFrame, target_col: str, binary: bool = False,
                           smoothing: float = 0.5) -> float:
    """Calculate weighted entropy after decision tree split.

    Args:
        dt: Fitted sklearn Pipeline containing the decision tree
        df: DataFrame containing the features
        target_col: Column name containing word order labels
        binary: If True, calculate binary entropy. If False, six-class entropy.
        smoothing: Smoothing factor for entropy calculation

    Returns:
        Weighted average entropy of leaf nodes
    """
    # Get the decision tree classifier from the pipeline
    tree_model = dt.named_steps['clf']

    # Prepare features (drop target column)
    X = df.drop(columns=[target_col])

    # Transform features through preprocessor and get leaf assignments
    leaf_ids = tree_model.apply(dt.named_steps['preprocessor'].transform(X))

    # Calculate entropy for each leaf
    weighted_entropy = 0.0
    total_samples = len(df)

    for leaf_id in np.unique(leaf_ids):
        leaf_mask = leaf_ids == leaf_id
        leaf_df = df[leaf_mask]
        leaf_weight = len(leaf_df) / total_samples
        leaf_entropy = calculate_base_entropy(leaf_df, target_col, binary=binary, smoothing=smoothing)
        weighted_entropy += leaf_weight * leaf_entropy

    return weighted_entropy


def calculate_metrics(
    language_data: Dict[str, Tuple[Pipeline, pd.DataFrame]],
    target_col: str = "core_args",
    binary_entropy: bool = False,
    smoothing: float = 0.5
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
        base_ent = calculate_base_entropy(df, target_col, binary=binary_entropy, smoothing=smoothing)
        reduced_ent = calculate_tree_entropy(dt, df, target_col, binary=binary_entropy, smoothing=smoothing)
        delta_ent = base_ent - reduced_ent

        # Calculate accuracy on full training data
        X = df.drop(columns=[target_col])
        y = df[target_col]
        accuracy = dt.score(X, y)

        # Number of items
        n_items = len(df)

        # Swap statistics (if num_swaps column exists)
        if 'num_swaps' in df.columns:
            n_flexible = (df['num_swaps'] >= 1).sum()
            n_fully_flexible = (df['num_swaps'] == 4).sum()
        else:
            n_flexible = 0
            n_fully_flexible = 0

        metrics.append({
            'language': lang_name,
            'base_entropy': base_ent,
            'reduced_entropy': reduced_ent,
            'delta_entropy': delta_ent,
            'accuracy': accuracy,
            'n_items': n_items,
            'n_flexible': n_flexible,
            'n_fully_flexible': n_fully_flexible,
            'total_pairs': sum(df['num_swaps']),
        })

    return pd.DataFrame(metrics).sort_values('language')


def generate_html_index(
        language_data: Dict[str, Tuple[Pipeline, pd.DataFrame]],
        html_directory: str,
        target_col: str = "core_args",
        smoothing: float = 0.5
) -> None:
    """Generate interactive overview page with metrics and language links.

    Args:
        language_data: Dict mapping language_name -> (fitted_pipeline, dataframe)
        html_directory: Directory to save the index.html file
        target_col: Column name containing word order labels
        smoothing: Smoothing factor for entropy calculation
    """
    html_directory = Path(html_directory)

    # Calculate metrics for BOTH entropy types
    metrics_six = calculate_metrics(language_data, target_col, binary_entropy=False, smoothing=smoothing)
    metrics_binary = calculate_metrics(language_data, target_col, binary_entropy=True, smoothing=smoothing)

    # Find corresponding HTML files
    html_files = {f.stem: f for f in html_directory.glob("*.html") if f.name.lower() != "index.html"}

    # Generate table rows for both entropy types
    def generate_rows(metrics_df):
        rows = []
        for _, row in metrics_df.iterrows():
            lang_name = row['language'].replace("_", " ")
            lang_file = html_files.get(row['language'].replace(" ", "_"))

            if lang_file:
                lang_link = f'<a href="/multiblimp/{quote(lang_file.stem)}">{lang_name}</a>'
            else:
                lang_link = lang_name

            rows.append(f"""
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
            """)
        return ''.join(rows)

    rows_six = generate_rows(metrics_six)
    rows_binary = generate_rows(metrics_binary)

    # Generate scatter plot data for both entropy types
    def generate_plot_data(metrics_df):
        data = []
        for _, row in metrics_df.iterrows():
            lang_name = row['language'].replace("_", " ")
            lang_file = html_files.get(row['language'].replace(" ", "_"))
            url = f"/multiblimp/{quote(lang_file.stem)}" if lang_file else None

            data.append({
                'name': lang_name,
                'base': row['base_entropy'],
                'reduced': row['reduced_entropy'],
                'url': url
            })
        return data

    plot_data_six = generate_plot_data(metrics_six)
    plot_data_binary = generate_plot_data(metrics_binary)

    # Convert to JSON for JavaScript
    import json
    plot_data_six_json = json.dumps(plot_data_six)
    plot_data_binary_json = json.dumps(plot_data_binary)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MultiBLiMP v2 - Language Overview</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --bg: #f5f5f4;
            --card: #ffffff;
            --text: #1c1917;
            --accent: #2563eb;
            --border: #e7e5e4;
            --hover: #f5f5f5;
        }}
        body {{
            margin: 0;
            min-height: 100vh;
            font-family: 'DM Sans', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 2rem;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: var(--card);
            padding: 2.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.08);
            border: 1px solid var(--border);
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 2rem;
        }}
        .title-section {{
            flex: 1;
        }}
        h1 {{
            margin: 0 0 0.5rem 0;
            font-size: 1.75rem;
            font-weight: 600;
            color: var(--text);
        }}
        .subtitle {{
            color: #78716c;
            font-size: 0.95rem;
        }}
        .controls {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        .controls label {{
            font-size: 0.875rem;
            font-weight: 500;
            color: #57534e;
        }}
        select {{
            padding: 0.5rem 2rem 0.5rem 0.875rem;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: white;
            color: var(--text);
            font-family: 'DM Sans', system-ui, sans-serif;
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%2357534e' d='M6 8L2 4h8z'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 0.625rem center;
        }}
        select:hover {{
            border-color: var(--accent);
        }}
        select:focus {{
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1.5rem;
        }}
        thead {{
            position: sticky;
            top: 0;
            z-index: 10;
            background: var(--hover);
            border-bottom: 2px solid var(--border);
        }}
        th {{
            padding: 0.875rem 1rem;
            text-align: left;
            font-weight: 600;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.025em;
            color: #57534e;
            cursor: pointer;
            user-select: none;
            position: relative;
            background: var(--hover);
        }}
        th:hover {{
            background: #ececeb;
        }}
        th.sortable::after {{
            content: '⇅';
            position: absolute;
            right: 0.5rem;
            opacity: 0.3;
            font-size: 0.75rem;
        }}
        th.sort-asc::after {{
            content: '↑';
            opacity: 1;
        }}
        th.sort-desc::after {{
            content: '↓';
            opacity: 1;
        }}
        td {{
            padding: 0.875rem 1rem;
            border-bottom: 1px solid var(--border);
        }}
        tbody tr:hover {{
            background: var(--hover);
        }}
        tbody tr:last-child td {{
            border-bottom: none;
        }}
        td:nth-child(2), td:nth-child(3), td:nth-child(4), td:nth-child(5), td:nth-child(6), td:nth-child(7), td:nth-child(8) {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        a {{
            color: var(--accent);
            text-decoration: none;
            font-weight: 500;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        #scatterPlot {{
            width: 100%;
            height: 500px;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title-section">
                <h1>MultiBLiMP v2 - Language Overview</h1>
                <div class="subtitle">Word order prediction through decision trees</div>
            </div>
            <div class="controls">
                <label for="entropyType">Entropy Type:</label>
                <select id="entropyType">
                    <option value="six" selected>Six-class</option>
                    <option value="binary">Binary (majority vs. rest)</option>
                </select>
            </div>
        </div>

        <div id="scatterPlot"></div>

        <table id="dataTable">
            <thead>
                <tr>
                    <th class="sortable" data-column="0">Language</th>
                    <th class="sortable" data-column="1">Base Entropy</th>
                    <th class="sortable" data-column="2">Reduced Entropy</th>
                    <th class="sortable" data-column="3">Δ Entropy</th>
                    <th class="sortable" data-column="4">DT Acc%</th>
                    <th class="sortable" data-column="5">N Items</th>
                    <th class="sortable" data-column="6">N 1 swap</th>
                    <th class="sortable" data-column="7">N 4 swap</th>
                    <th class="sortable" data-column="8">N Pairs</th>
                </tr>
            </thead>
            <tbody id="tableBody">
                {rows_six}
            </tbody>
        </table>
    </div>

    <script>
        // Store both data sets for table and plot
        const tableData = {{
            six: `{rows_six}`,
            binary: `{rows_binary}`
        }};

        const plotData = {{
            six: {plot_data_six_json},
            binary: {plot_data_binary_json}
        }};

        const table = document.getElementById('dataTable');
        const tbody = document.getElementById('tableBody');
        const headers = table.querySelectorAll('th.sortable');
        const entropySelect = document.getElementById('entropyType');

        let currentSort = {{ column: 0, ascending: true }};
        let currentEntropyType = 'six';

        // Initialize plot
        renderPlot(currentEntropyType);

        // Sort by language (alphabetically) on load
        sortTable(0, true, true);

        // Handle entropy type change
        entropySelect.addEventListener('change', (e) => {{
            currentEntropyType = e.target.value;
            tbody.innerHTML = tableData[currentEntropyType];
            renderPlot(currentEntropyType);
            // Re-apply current sort
            sortTable(currentSort.column, currentSort.ascending, true);
        }});

        // Render scatter plot
        function renderPlot(entropyType) {{
            const data = plotData[entropyType];

            const trace = {{
                x: data.map(d => d.base),
                y: data.map(d => d.reduced),
                mode: 'markers',
                type: 'scatter',
                text: data.map(d => d.name),
                customdata: data.map(d => d.url),
                hovertemplate: '<b>%{{text}}</b><br>' +
                              'Base Entropy: %{{x:.3f}}<br>' +
                              'Reduced Entropy: %{{y:.3f}}<br>' +
                              '<extra></extra>',
                marker: {{
                    size: 8,
                    color: '#2563eb',
                    opacity: 0.7,
                    line: {{
                        color: '#1e40af',
                        width: 1
                    }}
                }}
            }};

            const layout = {{
                title: {{
                    text: 'Base Entropy vs. Reduced Entropy',
                    font: {{
                        family: 'DM Sans, system-ui, sans-serif',
                        size: 16,
                        color: '#1c1917'
                    }}
                }},
                xaxis: {{
                    title: 'Base Entropy',
                    gridcolor: '#e7e5e4',
                    zeroline: false
                }},
                yaxis: {{
                    title: 'Reduced Entropy',
                    gridcolor: '#e7e5e4',
                    zeroline: false
                }},
                plot_bgcolor: '#ffffff',
                paper_bgcolor: '#ffffff',
                font: {{
                    family: 'DM Sans, system-ui, sans-serif',
                    color: '#1c1917'
                }},
                hovermode: 'closest',
                margin: {{ t: 50, r: 30, b: 50, l: 60 }}
            }};

            const config = {{
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                displaylogo: false
            }};

            Plotly.newPlot('scatterPlot', [trace], layout, config);

            // Add click event to navigate to language page
            document.getElementById('scatterPlot').on('plotly_click', function(data) {{
                const url = data.points[0].customdata;
                if (url) {{
                    window.location.href = url;
                }}
            }});
        }}

        // Handle column header clicks
        headers.forEach(header => {{
            header.addEventListener('click', () => {{
                const column = parseInt(header.dataset.column);
                const ascending = currentSort.column === column ? !currentSort.ascending : true;
                sortTable(column, ascending);
            }});
        }});

        function sortTable(column, ascending, skipToggle = false) {{
            const rows = Array.from(tbody.querySelectorAll('tr'));

            rows.sort((a, b) => {{
                const aVal = parseFloat(a.children[column].dataset.sort) || a.children[column].dataset.sort;
                const bVal = parseFloat(b.children[column].dataset.sort) || b.children[column].dataset.sort;

                if (typeof aVal === 'string' && typeof bVal === 'string') {{
                    return ascending ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                }}
                return ascending ? aVal - bVal : bVal - aVal;
            }});

            rows.forEach(row => tbody.appendChild(row));

            if (!skipToggle) {{
                // Update header indicators
                headers.forEach(h => {{
                    h.classList.remove('sort-asc', 'sort-desc');
                }});
                headers[column].classList.add(ascending ? 'sort-asc' : 'sort-desc');

                currentSort = {{ column, ascending }};
            }} else {{
                headers[0].classList.add('sort-asc');
            }}
        }}
    </script>
</body>
</html>"""

    output_path = html_directory / "index.html"
    output_path.write_text(html_content, encoding="utf-8")