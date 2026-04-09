import numpy as np
import plotly.graph_objects as go
import scipy
import json
import seaborn as sns
from collections import defaultdict

from matplotlib.colors import to_hex

from .entropy import order_entropy


def clean_rule(rule):
    rule = (
        rule.replace("num__", "")
        .replace("cat__", "")
        .replace("sibling-deprel", "sibling")
    )

    if rule == "in_question":
        return "in a question?"

    rule_elements = rule.split("_")
    rule_elements[0] = (
        rule_elements[0].replace("nsubj", "subject").replace("obj", "object")
    )

    if rule_elements[-1] == "nan":
        rule = ".".join(rule_elements[:-1]) + " is not set"
    elif rule_elements[-2] in ["sibling", "sibling-pos"]:
        rule = f"{rule_elements[0]} has {rule_elements[-1]} sibling?"
    elif rule_elements[-2] in ["sibling-L"]:
        rule = f"{rule_elements[0]} has left {rule_elements[-1]} sibling?"
    elif rule_elements[-2] in ["sibling-R"]:
        rule = f"{rule_elements[0]} has right {rule_elements[-1]} sibling?"
    elif len(rule_elements) == 3:
        rule = f"{rule_elements[0]}.{rule_elements[1]} = {rule_elements[2]}"
    else:
        rule = "_".join(rule_elements[:-1]) + f" = {rule_elements[-1]}"

    return rule


def get_correlated_features(prep, dt_df):
    X = prep.transform(dt_df)

    if isinstance(X, scipy.sparse._csr.csr_matrix):
        return defaultdict(list)

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(X, rowvar=False)

    feature_names = prep.get_feature_names_out()
    rule_names = [clean_rule(feature) for feature in feature_names]

    correlated_features = defaultdict(list)

    for idx, phi1 in enumerate(rule_names):
        for jdx, phi2 in enumerate(rule_names):
            if (jdx > idx) and corr[idx, jdx] == 1.0:
                correlated_features[phi1].append(phi2)
                correlated_features[phi2].append(phi1)

    return correlated_features


def get_sample_ids(prep, clf, dt_df, predictor_var, max_rows=100, seed=42):
    predictor_samples = {}

    for predictor_value, df in dt_df.groupby(predictor_var):
        X_model = prep.transform(df)

        rng = np.random.default_rng(seed)
        node_indicator = clf.decision_path(X_model)
        n_nodes = clf.tree_.node_count

        out = {}
        for node_id in range(n_nodes):
            rows = node_indicator[:, node_id].nonzero()[0]
            if rows.size > max_rows:
                rows = rng.choice(rows, size=max_rows, replace=False)
            out[node_id] = df.index[rows]

        predictor_samples[predictor_value] = out

    return predictor_samples


def get_samples(
    prep,
    clf,
    full_df,
    label_distribution,
    class2idx,
    dt_df,
    predictor_var,
    max_rows=100,
    seed=42,
):
    sample_ids = get_sample_ids(prep, clf, dt_df, predictor_var, max_rows, seed)

    full_df["sen_str"] = [" ".join(sen) for sen in full_df["sen"]]
    full_df["treebank_link"] = [
        f"<a href='https://universal.grew.fr/?corpus={treebank}@2.17' target='_blank'>{treebank}</a>"
        for treebank in full_df["treebank"]
    ]
    keep_columns = [
        "sen_str",
        "nsubj_form",
        "obj_form",
        "head_form",
        "treebank_link",
        "sent_id",
    ]
    predictor_samples = {}
    for predictor, node_sample_ids in sample_ids.items():
        predictor_samples[predictor] = {
            int(node_idx): {
                "rows": full_df.loc[sample_ids][keep_columns].to_dict("records"),
                "count": len(sample_ids),
                "total_count": label_distribution[node_idx][class2idx[predictor]],
            }
            for node_idx, sample_ids in node_sample_ids.items()
        }

    return predictor_samples


def compute_tree_layout(tree):
    """
    Returns dicts: node_id -> x, node_id -> y, plus edge data with sample counts
    Root at top, leaves evenly spaced.
    """
    children_left = tree.children_left
    children_right = tree.children_right
    n_samples = tree.n_node_samples

    node_x = {}
    node_y = {}
    current_x = 0

    def dfs(node_id, depth):
        nonlocal current_x

        left = children_left[node_id]
        right = children_right[node_id]

        if left == -1:  # leaf
            node_x[node_id] = current_x
            node_y[node_id] = -depth
            current_x += 1
        else:
            dfs(left, depth + 1)
            dfs(right, depth + 1)
            node_x[node_id] = (node_x[left] + node_x[right]) / 2
            node_y[node_id] = -depth

    dfs(0, 0)

    node_ids = list(node_x.keys())

    edge_x_true, edge_y_true = [], []
    edge_x_false, edge_y_false = [], []
    edge_samples_true = []
    edge_samples_false = []

    for node_id in node_ids:
        left = children_left[node_id]
        right = children_right[node_id]

        if left != -1:
            # FALSE branch (left) - absolute sample count
            edge_x_false += [node_x[node_id], node_x[left], None]
            edge_y_false += [node_y[node_id], node_y[left], None]
            edge_samples_false += [n_samples[left], n_samples[left], None]

            # TRUE branch (right) - absolute sample count
            edge_x_true += [node_x[node_id], node_x[right], None]
            edge_y_true += [node_y[node_id], node_y[right], None]
            edge_samples_true += [n_samples[right], n_samples[right], None]

    return (
        node_x,
        node_y,
        edge_x_true,
        edge_y_true,
        edge_x_false,
        edge_y_false,
        edge_samples_true,
        edge_samples_false,
        node_ids,
    )


def interpolate_color(hex1, hex2, t):
    """Linearly interpolate between two hex colors."""
    hex1 = hex1.lstrip("#")
    hex2 = hex2.lstrip("#")

    r1, g1, b1 = int(hex1[0:2], 16), int(hex1[2:4], 16), int(hex1[4:6], 16)
    r2, g2, b2 = int(hex2[0:2], 16), int(hex2[2:4], 16), int(hex2[4:6], 16)

    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)

    return f"#{r:02x}{g:02x}{b:02x}"


# The 6 canonical SVO permutations — always show these in a fixed order
SVO_PERMUTATIONS = ["vos", "vso", "ovs", "svo", "osv", "sov"]


# ============================================================
# HTML export
# ============================================================


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _json(obj):
    return json.dumps(obj, cls=_NumpyEncoder)


def write_html(
    fig,
    node_samples,
    node_data,
    out_file,
    classes,
    hex_colors,
    root_dist_counts,
    accuracy,
    meta=None,
):
    div_id = "tree-figure"

    html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        div_id=div_id,
    )

    # Build legend items: swatch + label + count + horizontal bar
    total_root = sum(root_dist_counts) or 1
    max_root = max(root_dist_counts) or 1
    legend_items = ""
    for cls, color, cnt in zip(classes, hex_colors, root_dist_counts):
        bar_w = round((cnt / max_root) * 80)  # max 80px wide bar
        pct = cnt / total_root * 100
        dimmed = "opacity:0.35;" if cnt == 0 else ""
        legend_items += f"""
        <div class="legend-item" style="{dimmed}">
          <span class="legend-swatch" style="background:{color};"></span>
          <span class="legend-label">{cls}</span>
          <span class="legend-count">{cnt}</span>
          <div class="legend-bar-track">
            <div class="legend-bar-fill" style="width:{bar_w}px;background:{color};"></div>
          </div>
        </div>"""

    # Build meta rows
    meta_rows = ""
    if meta:
        for key, val in meta.items():
            meta_rows += f'<div class="meta-row"><span class="meta-key">{key}</span><span class="meta-val">{val}</span></div>'

    accuracy_pct = f"{accuracy * 100:.1f}%"

    html += f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg: #f5f5f4;
    --surface: #ffffff;
    --surface-raised: #fafaf9;
    --border: #e7e5e4;
    --border-subtle: #f0efee;
    --text-primary: #1c1917;
    --text-secondary: #78716c;
    --text-tertiary: #a8a29e;
    --accent: #2563eb;
    --accent-light: #eff6ff;
    --font-ui: 'DM Sans', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
    --radius: 10px;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
    --shadow-lg: 0 8px 24px rgba(0,0,0,0.10), 0 4px 8px rgba(0,0,0,0.04);
  }}

  body {{
    font-family: var(--font-ui);
    background: var(--bg);
    color: var(--text-primary);
    overflow: hidden;
    height: 100vh;
  }}

  .page {{
    display: flex;
    height: 100vh;
    overflow: hidden;
  }}

  .tree-panel {{
    flex: 0 0 auto;
    width: 62%;
    position: relative;
    overflow: hidden;
    background: var(--bg);
  }}

  .table-panel {{
    flex: 1 1 auto;
    overflow-y: auto;
    background: var(--surface);
    border-left: 1px solid var(--border);
    font-family: var(--font-ui);
    font-size: 14px;
    display: flex;
    flex-direction: column;
  }}

  .table-panel-header {{
    padding: 16px 20px 12px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    position: sticky;
    top: 0;
    z-index: 10;
    text-align: center;
  }}

  .table-panel-header h2 {{
    font-size: 14px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }}

  .table-panel-body {{
    padding: 0 20px 20px;
    flex: 1;
  }}

  .splitter {{
    width: 5px;
    cursor: col-resize;
    background: var(--border);
    transition: background 0.15s;
    z-index: 20;
    flex-shrink: 0;
  }}

  .splitter:hover {{ background: #a8a29e; }}

  /* ── Info panel ── */
  .info-panel {{
    position: absolute;
    top: 16px;
    left: 16px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-md);
    z-index: 100;
    width: 240px;
    overflow: hidden;
    font-family: var(--font-ui);
  }}

  .info-panel-header {{
    padding: 10px 14px;
    background: var(--text-primary);
    color: white;
  }}

  .info-panel-header .language-name {{
    font-size: 15px;
    font-weight: 600;
    line-height: 1.2;
  }}

  .info-panel-header .accuracy-badge {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-top: 5px;
    background: rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 500;
    color: rgba(255,255,255,0.9);
  }}

  .accuracy-dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #4ade80;
  }}

  .info-panel-meta {{
    padding: 8px 14px;
    border-bottom: 1px solid var(--border-subtle);
  }}

  .meta-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3px 0;
  }}

  .meta-key {{
    font-size: 11px;
    color: var(--text-tertiary);
    font-weight: 500;
  }}

  .meta-val {{
    font-size: 11px;
    color: var(--text-secondary);
    font-family: var(--font-mono);
    font-weight: 500;
  }}

  .info-panel-legend {{
    padding: 8px 14px 10px;
  }}

  .legend-title {{
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-tertiary);
    margin-bottom: 6px;
  }}

  .legend-item {{
    display: grid;
    grid-template-columns: 9px 28px 28px 1fr;
    align-items: center;
    gap: 5px;
    padding: 3px 0;
  }}

  .legend-swatch {{
    width: 9px; height: 9px;
    border-radius: 2px;
    flex-shrink: 0;
  }}

  .legend-label {{
    font-size: 11px;
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-weight: 500;
  }}

  .legend-count {{
    font-size: 10px;
    color: var(--text-tertiary);
    font-family: var(--font-mono);
    text-align: right;
  }}

  .legend-bar-track {{
    height: 6px;
    background: var(--border-subtle);
    border-radius: 3px;
    overflow: hidden;
  }}

  .legend-bar-fill {{
    height: 100%;
    border-radius: 3px;
  }}

  /* ── Branch legend ── */
  .branch-legend {{
    display: flex;
    gap: 10px;
    padding: 10px 14px;
    border-top: 1px solid var(--border-subtle);
  }}

  .branch-pill {{
    display: flex;
    align-items: center;
    gap: 5px;
    background: var(--surface-raised);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-secondary);
  }}

  .branch-line {{
    width: 16px; height: 2.5px;
    border-radius: 2px;
  }}

  /* ── Per-node bar card (permanently visible, JS-positioned) ── */
  #node-card-layer {{
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
  }}

  .node-dist-card {{
    position: absolute;
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.13);
    border-radius: 4px;
    padding: 4px 5px 3px;
    display: flex;
    flex-direction: row;
    align-items: flex-end;
    gap: 3px;
    pointer-events: all;
    transform: translateX(-50%);
    box-shadow: 0 1px 4px rgba(0,0,0,0.10);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    cursor: default;
    z-index: 50;
  }}

  .node-dist-card.expanded {{
    transform: translateX(-50%) scale(2);
    transform-origin: top center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.18);
    z-index: 200;
  }}

  .node-dist-col {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1px;
    width: 14px;
  }}

  .node-dist-bar-wrap {{
    width: 14px;
    height: 30px;
    display: flex;
    align-items: flex-end;
  }}

  .node-dist-bar {{
    width: 100%;
    border-radius: 2px 2px 0 0;
    min-height: 1px;
  }}

  .node-dist-count {{
    font-size: 7px;
    color: #44403c;
    font-family: var(--font-mono);
    text-align: center;
    line-height: 1.2;
    white-space: nowrap;
  }}

  .node-dist-label {{
    font-size: 8.5px;
    color: #78716c;
    font-family: var(--font-mono);
    text-align: center;
    line-height: 1.2;
  }}

  /* ── Node distribution panel in table header ── */
  .node-dist-panel {{
    display: flex;
    flex-direction: row;
    align-items: flex-end;
    justify-content: center;
    gap: 6px;
    padding: 12px 20px 14px;
    border-bottom: 1px solid var(--border);
    background: var(--surface-raised);
  }}

  .node-dist-panel-col {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    width: 28px;
  }}

  .node-dist-panel-bar-wrap {{
    width: 28px;
    height: 60px;
    display: flex;
    align-items: flex-end;
  }}

  .node-dist-panel-bar {{
    width: 100%;
    border-radius: 3px 3px 0 0;
    min-height: 2px;
  }}

  .node-dist-panel-count {{
    font-size: 12px;
    color: #44403c;
    font-family: var(--font-mono);
    text-align: center;
    line-height: 1.3;
    white-space: nowrap;
  }}

  .node-dist-panel-label {{
    font-size: 9px;
    color: #78716c;
    font-family: var(--font-mono);
    text-align: center;
    line-height: 1.3;
  }}

  /* ── Decision path visualization ── */
  .decision-path-panel {{
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }}

  .decision-path-title {{
    font-size: 10px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
  }}

  .decision-path {{
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}

  .path-step {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    line-height: 1.4;
  }}

  .path-arrow {{
    color: var(--text-tertiary);
    font-size: 10px;
    flex-shrink: 0;
  }}

  .path-condition {{
    font-family: var(--font-mono);
    color: var(--text-primary);
    font-weight: 500;
  }}

  .path-branch {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 600;
    margin-left: 6px;
  }}

  .path-branch.true {{
    background: rgba(22, 163, 74, 0.1);
    color: #16a34a;
  }}

  .path-branch.false {{
    background: rgba(220, 38, 38, 0.1);
    color: #dc2626;
  }}

  /* ── Right panel table ── */
  .group-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 14px 0 6px;
  }}

  .group-swatch {{
    width: 10px; height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
  }}

  .group-title {{
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary);
  }}

  .group-count {{
    margin-left: auto;
    font-size: 13px;
    color: var(--text-tertiary);
    font-family: var(--font-mono);
  }}

  .node-table {{
    border-collapse: collapse;
    width: 100%;
    table-layout: fixed;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
  }}

  .node-table thead th {{
    background: var(--surface-raised);
    color: var(--text-secondary);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 6px 10px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}

  .node-table td {{
    padding: 5px 10px;
    border-bottom: 1px solid var(--border-subtle);
    vertical-align: top;
    color: var(--text-primary);
    font-size: 13px;
    line-height: 1.35;
    overflow: hidden;
    text-overflow: ellipsis;
  }}

  .node-table th.col-sen_str, .node-table td.col-sen_str {{
    width: 45%;
    white-space: normal;
    word-break: break-word;
  }}

  .node-table th.col-other, .node-table td.col-other {{
    width: auto;
    white-space: nowrap;
  }}

  .node-table tbody tr:last-child td {{ border-bottom: none; }}
  .node-table tbody tr:hover td {{ background: var(--accent-light); }}

  .node-table a {{
    color: var(--accent);
    text-decoration: none;
    font-weight: 500;
  }}

  .node-table a:hover {{ text-decoration: underline; }}

  .empty-state {{
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 100%;
    min-height: 300px;
    color: var(--text-tertiary);
    gap: 8px;
  }}

  .empty-state-icon {{ font-size: 32px; opacity: 0.4; }}
  .empty-state-text {{ font-size: 13px; font-weight: 500; }}
  .empty-state-sub {{ font-size: 12px; color: var(--text-tertiary); }}

  /* ── Toggle ── */
  .toggle-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 14px;
    border-top: 1px solid var(--border-subtle);
    cursor: pointer;
    user-select: none;
  }}

  .toggle-row:hover {{ background: var(--surface-raised); }}

  .toggle-label {{
    font-size: 11px;
    font-weight: 500;
    color: var(--text-secondary);
  }}

  .toggle-switch {{
    width: 28px;
    height: 16px;
    border-radius: 8px;
    background: var(--accent);
    position: relative;
    transition: background 0.2s;
    flex-shrink: 0;
  }}

  .toggle-switch.off {{ background: var(--border); }}

  .toggle-switch::after {{
    content: "";
    position: absolute;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: white;
    top: 2px;
    left: 14px;
    transition: left 0.2s;
    box-shadow: 0 1px 2px rgba(0,0,0,0.2);
  }}

  .toggle-switch.off::after {{ left: 2px; }}
</style>

<div class="page">

  <div class="info-panel">
    <div class="info-panel-header">
      <div class="language-name">{meta.get('Language', 'Decision Tree').replace('_', ' ') if meta else 'Decision Tree'}</div>
      <div class="accuracy-badge">
        <span class="accuracy-dot"></span>
        {accuracy_pct} accuracy
      </div>
    </div>
    {"<div class='info-panel-meta'>" + meta_rows + "</div>" if meta_rows else ""}
    <div class="info-panel-legend">
      <div class="legend-title">Classes · dataset distribution</div>
      {legend_items}
    </div>
    <div class="toggle-row" id="dist-toggle-row" onclick="toggleDistributions()">
      <span class="toggle-label">Show node distributions</span>
      <span class="toggle-switch" id="dist-toggle-switch"></span>
    </div>
    <div class="branch-legend">
      <div class="branch-pill">
        <span class="branch-line" style="background:#dc2626;"></span>False
      </div>
      <div class="branch-pill">
        <span class="branch-line" style="background:#16a34a;"></span>True
      </div>
    </div>
  </div>

  <div class="tree-panel">
    <div id="node-card-layer"></div>
  </div>

  <div class="splitter" id="dragbar"></div>

  <div class="table-panel">
    <div class="table-panel-header">
      <h2 id="table-panel-title">Examples</h2>
    </div>
    <div id="node-path-panel" style="display:none;"></div>
    <div id="node-dist-panel" style="display:none;"></div>
    <div class="table-panel-body">
      <div class="empty-state" id="node-table">
        <div class="empty-state-icon">⬡</div>
        <div class="empty-state-text">No node selected</div>
        <div class="empty-state-sub">Click any node in the tree to explore examples</div>
      </div>
    </div>
  </div>
</div>

<script>
const nodeSamples = {_json(node_samples)};
const nodeData = {_json({str(k): v for k, v in node_data.items()})};
const classColors = {_json(dict(zip([str(c) for c in classes], hex_colors)))};

// ── Per-node distribution bar cards ──
const cardLayer = document.getElementById("node-card-layer");

function buildCard(nd) {{
  const dist = nd.dist;
  const maxCnt = Math.max(...dist.map(d => d.cnt), 1);
  const BAR_MAX_H = 30;

  const card = document.createElement("div");
  card.className = "node-dist-card";

  dist.forEach(d => {{
    const h = Math.max(1, Math.round((d.cnt / maxCnt) * BAR_MAX_H));
    const dim = d.cnt === 0 ? "opacity:0.18;" : "";
    const col = document.createElement("div");
    col.className = "node-dist-col";
    col.style.cssText = dim;
    // No count in the compact view — only bar + label
    col.innerHTML = `
      <div class="node-dist-bar-wrap">
        <div class="node-dist-bar" style="height:${{h}}px;background:${{d.color}};"></div>
      </div>
      <div class="node-dist-label">${{d.cls}}</div>`;
    card.appendChild(col);
  }});

  // Hover: expand to 2× and show counts
  card.addEventListener("mouseenter", function() {{
    card.classList.add("expanded");
    // Inject count elements
    card.querySelectorAll(".node-dist-col").forEach((col, idx) => {{
      const existing = col.querySelector(".node-dist-count");
      if (!existing) {{
        const countEl = document.createElement("div");
        countEl.className = "node-dist-count";
        countEl.textContent = dist[idx].cnt;
        // Insert between bar and label
        const label = col.querySelector(".node-dist-label");
        col.insertBefore(countEl, label);
      }}
    }});
  }});

  card.addEventListener("mouseleave", function() {{
    card.classList.remove("expanded");
    card.querySelectorAll(".node-dist-count").forEach(el => el.remove());
  }});

  return card;
}}

function toggleDistributions() {{
  showDistributions = !showDistributions;
  document.getElementById("dist-toggle-switch").classList.toggle("off", !showDistributions);
  renderNodeCards();
}}

// ── Example table ──
function renderNodeOverview(nodeId) {{
  const container = document.getElementById("node-table");
  const titleEl = document.getElementById("table-panel-title");
  const distPanel = document.getElementById("node-dist-panel");
  const pathPanel = document.getElementById("node-path-panel");

  const groups = [];
  for (const [predictorValue, nodeMap] of Object.entries(nodeSamples)) {{
    const entry = nodeMap[nodeId];
    if (entry && entry.count > 0) {{
      groups.push({{
        predictor: predictorValue,
        rows: entry.rows,
        count: entry.count,
        total_count: entry.total_count
      }});
    }}
  }}

  const totalSamples = groups.reduce((s,g) => s + g.total_count, 0);
  titleEl.textContent = totalSamples > 0
    ? `Node ${{nodeId}} · ${{totalSamples}} samples`
    : `Node ${{nodeId}}`;

  // ── Build decision path from root to this node ──
  const path = [];
  let currentNode = nodeId;
  const nd = nodeData[String(nodeId)];

  console.log("Building path for node:", nodeId);
  console.log("Node data:", nd);

  while (nd && nodeData[String(currentNode)]) {{
    const current = nodeData[String(currentNode)];
    console.log("Current node:", currentNode, "Rule:", current.rule, "Parent:", current.parent);
    if (current.rule) {{
      path.unshift({{
        rule: current.rule[0],
        isTrueBranch: current.rule[1]
      }});
    }}
    if (current.parent === null || current.parent === undefined) break;
    currentNode = current.parent;
  }}

  console.log("Built path:", path);

  // Render decision path
  if (path.length > 0) {{
    let pathHtml = '<div class="decision-path-panel">';
    pathHtml += '<div class="decision-path-title">Decision Path</div>';
    pathHtml += '<div class="decision-path">';

    path.forEach((step, idx) => {{
      const branchClass = step.isTrueBranch ? 'true' : 'false';
      const branchLabel = step.isTrueBranch ? 'TRUE' : 'FALSE';
      const arrow = idx === 0 ? '▶' : '↳';

      pathHtml += `
        <div class="path-step">
          <span class="path-arrow">${{arrow}}</span>
          <span class="path-condition">${{step.rule}}</span>
          <span class="path-branch ${{branchClass}}">${{branchLabel}}</span>
        </div>`;
    }});

    pathHtml += '</div></div>';
    pathPanel.innerHTML = pathHtml;
    pathPanel.style.display = "block";
    console.log("Path panel HTML set");
  }} else {{
    pathPanel.style.display = "none";
    console.log("No path to display");
  }}

  // ── Distribution bar panel ──
  if (nd) {{
    const dist = nd.dist;
    const maxCnt = Math.max(...dist.map(d => d.cnt), 1);
    const BAR_MAX_H = 60;
    let panelHtml = '<div class="node-dist-panel">';
    dist.forEach(d => {{
      const h = Math.max(2, Math.round((d.cnt / maxCnt) * BAR_MAX_H));
      const dim = d.cnt === 0 ? "opacity:0.18;" : "";
      panelHtml += `
        <div class="node-dist-panel-col" style="${{dim}}">
          <div class="node-dist-panel-bar-wrap">
            <div class="node-dist-panel-bar" style="height:${{h}}px;background:${{d.color}};"></div>
          </div>
          <div class="node-dist-panel-count">${{d.cnt}}</div>
          <div class="node-dist-panel-label">${{d.cls}}</div>
        </div>`;
    }});
    panelHtml += "</div>";
    distPanel.innerHTML = panelHtml;
    distPanel.style.display = "block";
  }} else {{
    distPanel.style.display = "none";
  }}

  if (groups.length === 0) {{
    container.innerHTML = `<div class="empty-state">
      <div class="empty-state-icon">∅</div>
      <div class="empty-state-text">No examples for this node</div></div>`;
    return;
  }}

  groups.sort((a, b) => b.total_count - a.total_count);
  let html = "";

  for (const group of groups) {{
    const color = classColors[group.predictor] || "#888";
    const pct = totalSamples > 0 ? (group.total_count / totalSamples * 100).toFixed(1) : "0.0";
    const shownLabel = group.count < group.total_count
      ? `showing ${{group.count}} of ${{group.total_count}}`
      : `${{group.total_count}}`;
    html += `
      <div class="group-header">
        <span class="group-swatch" style="background:${{color}};"></span>
        <span class="group-title">${{group.predictor}}</span>
        <span class="group-count">${{shownLabel}} (${{pct}}%)</span>
      </div>`;

    if (!group.rows || group.rows.length === 0) {{
      html += `<p style="color:var(--text-tertiary);font-size:12px;padding:4px 0 12px;">No sample rows available.</p>`;
      continue;
    }}

    const cols = Object.keys(group.rows[0]);
    html += "<table class='node-table'><thead><tr>";
    cols.forEach(c => {{
      const cls = c === "sen_str" ? "col-sen_str" : "col-other";
      html += `<th class="${{cls}}">${{c}}</th>`;
    }});
    html += "</tr></thead><tbody>";
    group.rows.forEach(r => {{
      html += "<tr>";
      cols.forEach(c => {{
        const cls = c === "sen_str" ? "col-sen_str" : "col-other";
        html += `<td class="${{cls}}">${{r[c]}}</td>`;
      }});
      html += "</tr>";
    }});
    html += "</tbody></table>";
  }}

  container.innerHTML = html;
}}

// ── Plotly setup ──
const plot = document.getElementById("{div_id}");
document.querySelector(".tree-panel").prepend(plot);
plot.style.width = "100%";
plot.style.height = "100%";
Plotly.Plots.resize(plot);

function scaleTreeFont() {{
  const w = window.innerWidth;
  const newSize = w < 1400 ? 10 : w < 1920 ? 11 : 13;
  const annotations = plot.layout.annotations;
  for (let i = 0; i < annotations.length; i++) {{
    annotations[i].font.size = newSize;
  }}
  Plotly.relayout(plot, {{ annotations }});
}}

let showDistributions = true;

function renderNodeCards() {{
  cardLayer.innerHTML = "";
  if (!showDistributions) return;

  // Plotly renders each annotation as <g class="annotation" data-index="N">
  // We read the annotation's stored name (= sklearn node_id) via plot.layout.annotations[N].name
  const panelRect = document.querySelector(".tree-panel").getBoundingClientRect();
  const annotGroups = plot.querySelectorAll("g.annotation");

  annotGroups.forEach((g, annotIdx) => {{
    // Get the sklearn node_id from the annotation's name field
    const ann = plot.layout.annotations[annotIdx];
    if (!ann) return;
    const nodeId = ann.name;
    if (nodeId === undefined) return;
    const nd = nodeData[nodeId];
    if (!nd) return;

    const bgRect = g.querySelector("rect.bg");
    if (!bgRect) return;

    const bbox = bgRect.getBoundingClientRect();
    const cx = bbox.left + bbox.width / 2 - panelRect.left;
    const top = bbox.bottom - panelRect.top;

    const card = buildCard(nd);
    card.style.left = cx + "px";
    card.style.top = top + "px";
    cardLayer.appendChild(card);
  }});
}}

scaleTreeFont();

plot.on("plotly_afterplot", renderNodeCards);
plot.on("plotly_relayout", renderNodeCards);
window.addEventListener("resize", () => {{
  Plotly.Plots.resize(plot);
  scaleTreeFont();
  // cards re-render via plotly_afterplot triggered by resize
}});

plot.on("plotly_hover", function(e) {{
  const drag = plot.querySelector(".nsewdrag");
  if (drag) drag.style.cursor = "pointer";
}});

plot.on("plotly_unhover", function() {{
  const drag = plot.querySelector(".nsewdrag");
  if (drag) drag.style.cursor = "default";
}});

plot.on("plotly_click", function(e) {{
  const nodeId = e.points[0].customdata.node_id;
  renderNodeOverview(nodeId);
}});

// ── Draggable splitter ──
const dragbar = document.getElementById("dragbar");
const treePanel = document.querySelector(".tree-panel");
let isDragging = false;

dragbar.addEventListener("mousedown", (e) => {{
  isDragging = true;
  document.body.style.cursor = "col-resize";
  document.body.style.userSelect = "none";
  e.preventDefault();
}});
document.addEventListener("mouseup", () => {{
  isDragging = false;
  document.body.style.cursor = "default";
  document.body.style.userSelect = "";
}});
document.addEventListener("mousemove", (e) => {{
  if (!isDragging) return;
  const page = document.querySelector(".page");
  const rect = page.getBoundingClientRect();
  const pct = ((e.clientX - rect.left) / rect.width) * 100;
  if (pct > 20 && pct < 80) {{
    treePanel.style.width = pct + "%";
    treePanel.style.flex = "0 0 auto";
    Plotly.Plots.resize(plot);
  }}
}});
</script>
"""

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)


def tree2html(
    pipeline_model, dt_df, full_df, predictor_var, out_file, max_rows=100, meta=None
):
    """
    pipeline_model:
        sklearn Pipeline with steps:
            - "preprocessor"
            - "clf" (DecisionTree*)

    dt_df / full_df:
        pandas DataFrames

    meta (optional dict):
        Extra info shown in the info panel, e.g.:
            {
                "Language": "Kurmanji",
                "Nodes": 23,
                "Depth": 5,
                "Training samples": 4486,
            }
        If not provided, these values are computed automatically where possible.
    """

    prep = pipeline_model.named_steps["preprocessor"]
    clf = pipeline_model.named_steps["clf"]

    # Layout
    tree = clf.tree_
    (
        node_x,
        node_y,
        edge_x_true,
        edge_y_true,
        edge_x_false,
        edge_y_false,
        edge_samples_true,
        edge_samples_false,
        node_ids,
    ) = compute_tree_layout(tree)

    # Node metadata
    feature = tree.feature
    n_samples = tree.n_node_samples
    model_classes = list(pipeline_model.classes_)
    class2idx_model = {c: idx for idx, c in enumerate(model_classes)}
    n_model_classes = len(model_classes)
    n_nodes = len(node_ids)
    impurity = tree.impurity
    max_impurity = np.log2(n_model_classes) or 1
    tree_values = tree.value
    predicted_class_ids = [value[0].argmax() for value in tree_values]
    predicted_class = [clf.classes_[idx] for idx in predicted_class_ids]
    label_distribution_model = [
        (tree_values[i][0] * n_samples[i]).astype(int).tolist() for i in range(n_nodes)
    ]
    accuracy = pipeline_model.score(dt_df, dt_df[predictor_var])

    # Pad to full SVO permutation set so legend & nodes always show all 6 classes.
    # Build display_classes in canonical SVO order; append any extras.
    display_classes = []
    for p in SVO_PERMUTATIONS:
        match = next((c for c in model_classes if c.lower() == p), None)
        display_classes.append(match if match else p)
    for c in model_classes:
        if c not in display_classes:
            display_classes.append(c)
    classes = display_classes
    n_classes = len(classes)

    # Expand label_distribution to full display_classes (missing classes get 0)
    label_distribution = []
    for dist in label_distribution_model:
        full_dist = []
        for cls in classes:
            if cls in class2idx_model:
                full_dist.append(dist[class2idx_model[cls]])
            else:
                full_dist.append(0)
        label_distribution.append(full_dist)

    class2idx = {c: idx for idx, c in enumerate(classes)}

    # Auto-build meta if not provided
    if meta is None:
        meta = {}

    tree_depth = clf.get_depth()
    n_leaves = clf.get_n_leaves()
    root_samples = n_samples[0]

    if "Nodes" not in meta:
        meta["Nodes"] = f"{n_nodes} ({n_leaves} leaves)"
    if "Depth" not in meta:
        meta["Depth"] = str(tree_depth)
    if "Training samples" not in meta:
        meta["Training samples"] = f"{root_samples:,}"
    if "Predictor" not in meta:
        meta["Predictor"] = predictor_var

    predictor_samples = get_samples(
        prep,
        clf,
        full_df,
        label_distribution,
        class2idx,
        dt_df,
        predictor_var,
        max_rows=max_rows,
    )

    feature_names = prep.get_feature_names_out()
    correlated_features = get_correlated_features(prep, dt_df)

    # Always generate palette across full SVO space so colours are stable
    palette = sns.color_palette("husl", n_colors=max(n_classes, len(SVO_PERMUTATIONS)))
    hex_colors = [to_hex(c) for c in palette[:n_classes]]

    # Grey out classes with zero samples at the root (absent from this language)
    root_dist = label_distribution[0]
    hex_colors = [
        color if root_dist[idx] > 0 else "#d4d4d4"
        for idx, color in enumerate(hex_colors)
    ]

    annotations = []
    node_data = {}

    # Build parent relationships and rules for path tracing
    children_left = tree.children_left
    children_right = tree.children_right
    parent_map = {}
    rule_map = {}  # node_id -> (rule_text, is_true_branch)

    # First pass: build parent map and rule map
    for node_id in node_ids:
        left_child = children_left[node_id]
        right_child = children_right[node_id]

        if left_child != -1:
            parent_map[left_child] = node_id
            parent_map[right_child] = node_id

            # Store the rule text for this internal node
            if feature[node_id] != -2:
                rule_text = clean_rule(feature_names[feature[node_id]])
                rule_map[left_child] = (rule_text, False)  # False branch (left)
                rule_map[right_child] = (rule_text, True)  # True branch (right)

    for i in node_ids:
        # Get the predicted class name from the model
        predicted_class_name = predicted_class[i]
        # Map it to the display class index
        display_class_idx = class2idx[predicted_class_name]
        node_color = hex_colors[display_class_idx]
        relative_entropy = impurity[i] / max_impurity
        dist_i = label_distribution[i]
        n_right = max(dist_i)
        n_wrong = sum(dist_i) - n_right
        binary_entropy = order_entropy(n_right, n_wrong)
        node_color = interpolate_color(node_color, "#ffffff", relative_entropy * 0.72)

        total_node = sum(dist_i) or 1

        # Small dim style shared by node-id and stats
        meta_style = "color:#44403c;font-size:9px"

        if feature[i] != -2:
            # Internal node: rule on top, [n] n= H= on second line
            rule = clean_rule(feature_names[feature[i]])
            corr = correlated_features[rule]
            corr_note = " [+]" if corr else ""
            label = (
                f"<b>{rule}{corr_note}</b><br>"
                f"<span style='{meta_style}'>"
                f"<b>[{i}]</b>  n={n_samples[i]}  H={binary_entropy:.2f}</span>"
            )
            hover_corr = list(corr)
        else:
            # Leaf: include all classes with count >= 50% of the majority
            sorted_dist = sorted(zip(dist_i, classes), key=lambda x: x[0], reverse=True)
            top_cnt, top_cls = sorted_dist[0]
            qualifying = [top_cls] + [
                cls for cnt, cls in sorted_dist[1:] if cnt > 0 and cnt >= 0.5 * top_cnt
            ]
            leaf_label = f"<b>{' / '.join(qualifying)}</b>"
            label = (
                f"{leaf_label}<br>"
                f"<span style='{meta_style}'>"
                f"<b>[{i}]</b>  n={n_samples[i]}  H={binary_entropy:.2f}</span>"
            )
            hover_corr = []

        annotations.append(
            dict(
                x=node_x[i],
                y=node_y[i],
                text=label,
                name=str(i),  # node_id stored here for JS lookup
                showarrow=False,
                align="center",
                font=dict(size=11, family="JetBrains Mono, monospace"),
                bgcolor=node_color,
                bordercolor="rgba(0,0,0,0.14)",
                borderwidth=1,
                borderpad=7,
            )
        )

        node_data[i] = {
            "x": float(node_x[i]),
            "y": float(node_y[i]),
            "color": node_color,
            "dist": [
                {"cls": cls, "cnt": int(cnt), "frac": cnt / total_node, "color": color}
                for cls, cnt, color in zip(classes, label_distribution[i], hex_colors)
            ],
            "corr": hover_corr,
            "is_leaf": bool(feature[i] == -2),
            "parent": int(parent_map[i]) if i in parent_map else None,
            "rule": rule_map.get(i, None),
        }

    # ── Plot ──
    fig = go.Figure()
    fig.update_layout(clickmode="event")

    # Calculate edge widths based on absolute sample counts
    # Find max samples for normalization
    all_edges = edge_samples_true + edge_samples_false
    max_samples = (
        0 if len(all_edges) == 0 else max([s for s in all_edges if s is not None])
    )
    min_width = 0.5
    max_width = 8.0

    # Create individual edge traces with variable width
    def add_edges_with_variable_width(edge_x, edge_y, edge_samples, color, name):
        i = 0
        while i < len(edge_x):
            if edge_x[i] is not None:
                # Get the edge segment (2 points)
                x_segment = [edge_x[i], edge_x[i + 1]]
                y_segment = [edge_y[i], edge_y[i + 1]]
                samples = edge_samples[i]

                # Calculate width based on sample proportion
                if max_samples > 0:
                    width = min_width + (max_width - min_width) * (
                        samples / max_samples
                    )
                else:
                    width = min_width

                fig.add_trace(
                    go.Scatter(
                        x=x_segment,
                        y=y_segment,
                        mode="lines",
                        line=dict(width=width, color=color),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                i += 3  # Skip to next edge (current, next, None)
            else:
                i += 1

    # Add True branch edges (green)
    add_edges_with_variable_width(
        edge_x_true, edge_y_true, edge_samples_true, "#16a34a", "True"
    )

    # Add False branch edges (red)
    add_edges_with_variable_width(
        edge_x_false, edge_y_false, edge_samples_false, "#dc2626", "False"
    )

    fig.update_layout(annotations=annotations)

    # Large invisible hit-target markers — big enough to cover the full annotation box.
    # The hover shows correlated features (if any); click opens the example table.
    scatter_customdata = []
    scatter_hovertemplates = []
    for i in node_ids:
        nd = node_data[i]
        corr = nd["corr"]
        if corr:
            ht = (
                "<br>".join(f"≈ {c}" for c in corr)
                + "<br><i style='color:#9c9490'>Click to explore</i>"
            )
        else:
            ht = "<i style='color:#9c9490'>Click to explore examples</i>"
        scatter_customdata.append({"node_id": i, "hovertext": ht})

    fig.add_trace(
        go.Scatter(
            x=[node_x[i] for i in node_ids],
            y=[node_y[i] for i in node_ids],
            mode="markers",
            marker=dict(
                size=56,
                color="rgba(0,0,0,0)",
                line=dict(width=0),
            ),
            customdata=scatter_customdata,
            hovertemplate="%{customdata.hovertext}<extra></extra>",
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="#e7e5e4",
                font=dict(family="DM Sans, sans-serif", size=12, color="#1c1917"),
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        autosize=True,
        clickmode="event",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="DM Sans, sans-serif"),
    )

    write_html(
        fig,
        predictor_samples,
        node_data,
        out_file,
        classes,
        hex_colors,
        label_distribution[0],
        accuracy,
        meta,
    )


if __name__ == "__main__":
    import joblib
    import pandas as pd

    lang = "Kurmanji"

    model = joblib.load(
        f"/home/jaap/Documents/multiblimp/word_order/decision_trees/core_arg_nsubj/{lang}.joblib"
    )

    full_df = pd.read_csv("full_df.csv", index_col=0)
    dt_df = pd.read_csv("dt_df.csv", index_col=0)

    html_file = f"word_order/decision_trees/html/{lang}.html"
    predictor_var = "core_args"

    tree2html(
        model,
        dt_df,
        full_df,
        predictor_var,
        html_file,
        max_rows=25,
        meta={"Language": lang},
    )
