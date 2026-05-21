import os
import warnings

import numpy as np
import plotly.graph_objects as go
import scipy
import seaborn as sns

from matplotlib.colors import to_hex

from .entropy import order_entropy, calculate_base_entropy, calculate_tree_entropy
from .utils import get_all_orders
from .html.html_tree import create_html


def clean_rule(rule):
    rule = (
        rule.replace("num__", "")
        .replace("cat__", "")
        .replace("sibling-deprel", "sibling")
        .replace("_missing", "nan")
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
    elif "child" in rule_elements[1]:
        if rule_elements[1] == "child-feat" and len(rule_elements) >= 5:
            rule = f"{rule_elements[0]}'s {rule_elements[2]} child: {rule_elements[3]} = {rule_elements[4]}"
        else:
            rule = f"{rule_elements[0]} has {rule_elements[2]} child?"
    else:
        rule = "_".join(rule_elements[:-1]) + f" = {rule_elements[-1]}"

    return rule


def get_correlated_features(prep, clf, dt_df):
    X = prep.transform(dt_df)

    if isinstance(X, scipy.sparse._csr.csr_matrix):
        return {}

    if X.dtype == object:
        try:
            X = X.astype(np.float64)
        except (ValueError, TypeError) as e:
            warnings.warn(
                f"Cannot convert object array to float: {e}. Skipping correlation computation.",
                UserWarning,
                stacklevel=2,
            )
            return {}

    X = X.astype(np.float64)
    feature_names = prep.get_feature_names_out()
    rule_names = [clean_rule(f) for f in feature_names]

    tree = clf.tree_
    node_indicator = clf.decision_path(X)

    result = {}

    for node_id in range(tree.node_count):
        feat_idx = tree.feature[node_id]
        if feat_idx == -2:  # leaf
            continue

        row_indices = node_indicator[:, node_id].nonzero()[0]
        if len(row_indices) < 2:
            result[node_id] = []
            continue

        X_node = X[row_indices]
        feat_col = X_node[:, feat_idx]
        feat_centered = feat_col - feat_col.mean()
        feat_std = feat_centered.std()

        if feat_std == 0:
            result[node_id] = []
            continue

        others_centered = X_node - X_node.mean(axis=0)
        others_std = others_centered.std(axis=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            correlations = np.dot(feat_centered, others_centered) / (
                len(row_indices) * feat_std * others_std
            )

        result[node_id] = [
            rule_names[jdx]
            for jdx, c in enumerate(correlations)
            if jdx != feat_idx and np.abs(c) > 0.99
        ]

    return result


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
    keep_columns = ["sen_str"]
    keep_columns.extend([col for col in full_df.columns if "_form" in col])
    keep_columns.extend([
        "treebank_link",
        "sent_id",
    ])
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


def tree2html(
    pipeline_model,
    dt_df,
    full_df,
    predictor_var,
    target,
    out_file,
    max_rows=100,
    meta=None,
    only_show_real_orders=False,
    correlate_features=True,
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

    # Pad to full classes permutation set so legend & nodes always show all classes.
    # Build display_classes in canonical class order; append any extras.
    
    all_orders = get_all_orders(predictor_var, target)

    if only_show_real_orders:
        all_orders = [order for order in all_orders if order in full_df[predictor_var].unique()]

    classes = list(all_orders)
    for c in model_classes:
        if c not in classes:
            classes.append(c)

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

    accuracy = pipeline_model.score(dt_df, dt_df[predictor_var])
    meta["accuracy"] = f"{accuracy * 100:.1f}%"

    base_ent = calculate_base_entropy(dt_df, predictor_var, binary=True)
    reduced_ent = calculate_tree_entropy(pipeline_model, dt_df, predictor_var, binary=True)
    meta["base entropy"] = f"{base_ent:.3f}"
    meta["reduced entropy"] = f"{reduced_ent:.3f}"

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

    if correlate_features:
        correlated_features = get_correlated_features(prep, clf, dt_df)
    else:
        correlated_features = {}

    # Always generate palette across full SVO space so colours are stable
    palette = sns.color_palette("husl", n_colors=max(n_classes, len(all_orders)))
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
            corr = correlated_features.get(i, [])
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
        meta,
    )


def write_html(
    fig,
    node_samples,
    node_data,
    out_file,
    classes,
    hex_colors,
    root_dist_counts,
    meta,
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

    meta['legend_items'] = legend_items
    meta['rows'] = meta_rows

    html += create_html(meta, node_samples, node_data, hex_colors, classes, div_id)
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)
