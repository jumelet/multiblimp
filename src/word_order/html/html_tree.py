import json
import numpy as np


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


def create_html(meta, node_samples, node_data, hex_colors, classes, div_id):
    return f"""
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
        transition: width 0.3s ease;
      }}

      .info-panel.minimized {{
        width: auto;
      }}

      .info-panel-header {{
        padding: 10px 14px;
        background: var(--text-primary);
        color: white;
        position: relative;
      }}

      .minimize-button {{
        position: absolute;
        top: 10px;
        right: 10px;
        width: 20px;
        height: 20px;
        border: none;
        background: rgba(255,255,255,0.2);
        color: white;
        border-radius: 4px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        line-height: 1;
        transition: background 0.2s;
        padding: 0;
      }}

      .minimize-button:hover {{
        background: rgba(255,255,255,0.3);
      }}

      .minimize-button:active {{
        background: rgba(255,255,255,0.4);
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
        transition: opacity 0.3s ease, max-height 0.3s ease;
        max-height: 500px;
        overflow: hidden;
      }}

      .info-panel.minimized .info-panel-meta,
      .info-panel.minimized .info-panel-legend,
      .info-panel.minimized .toggle-row,
      .info-panel.minimized .branch-legend {{
        opacity: 0;
        max-height: 0;
        padding-top: 0;
        padding-bottom: 0;
        border: none;
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
        transition: opacity 0.3s ease, max-height 0.3s ease, padding 0.3s ease;
        max-height: 500px;
        overflow: hidden;
      }}

      .toggle-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 14px;
        border-top: 1px solid var(--border-subtle);
        cursor: pointer;
        transition: opacity 0.3s ease, max-height 0.3s ease, padding 0.3s ease;
        max-height: 100px;
        overflow: hidden;
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
        transition: opacity 0.3s ease, max-height 0.3s ease, padding 0.3s ease;
        max-height: 100px;
        overflow: hidden;
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
        width: 19px;
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
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 150px;
      }}

      .node-table td:has(details[open]) {{
        white-space: normal;
        overflow: visible;
        max-width: none;
      }}

      .node-table td details[open] {{
        word-break: break-all;
        white-space: normal;
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

      <div class="info-panel" id="info-panel">
        <div class="info-panel-header">
          <button class="minimize-button" id="minimize-btn" onclick="toggleMinimize()" title="Minimize panel">−</button>
          <div class="language-name">{meta.get('Language', 'Decision Tree').replace('_', ' ')}</div>
          <div class="accuracy-badge">
            <span class="accuracy-dot"></span>
            {meta['accuracy']} accuracy
          </div>
        </div>
        {"<div class='info-panel-meta'>" + meta['rows'] + "</div>" if meta.get('rows') else ""}
        <div class="info-panel-legend">
          <div class="legend-title">Classes · dataset distribution</div>
          {meta['legend_items']}
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

    // ── Info panel minimize/expand ──
    function toggleMinimize() {{
      const panel = document.getElementById("info-panel");
      const btn = document.getElementById("minimize-btn");
      const isMinimized = panel.classList.toggle("minimized");
      btn.textContent = isMinimized ? "+" : "−";
      btn.title = isMinimized ? "Expand panel" : "Minimize panel";
    }}

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
            const raw = r[c] == null ? "" : String(r[c]);
            let content;
            if (c.endsWith("_features")) {{
              content = "<details><summary>features...</summary>" + raw.split(",").join("<br>") + "</details>";
            }} else {{
              content = raw;
            }}
            html += "<td class='" + cls + "'>" + content + "</td>";
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