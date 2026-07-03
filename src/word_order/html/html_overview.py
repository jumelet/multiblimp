def create_html(panels_html, all_data_json):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MultiBLiMP v2 - Word Order Overview</title>
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
            max-width: 1400px;
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
            gap: 2rem;
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
        .description {{
            color: #78716c;
            font-size: 0.95rem;
            line-height: 1.6;
            max-width: 700px;
            margin: 0.5rem 0 0;
        }}
        .controls {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            flex-shrink: 0;
            padding-top: 0.25rem;
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
        select:hover {{ border-color: var(--accent); }}
        select:focus {{
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
        }}
        .panel {{
            border: 1px solid var(--border);
            border-radius: 10px;
            background: #fafaf9;
            overflow: hidden;
            transition: box-shadow 0.15s ease, border-color 0.15s ease;
            cursor: pointer;
        }}
        .panel:hover {{
            box-shadow: 0 4px 16px rgba(0,0,0,0.10);
            border-color: #c7c4c0;
        }}
        .mini-plot {{
            width: 100%;
            height: 260px;
        }}
        .panel-label {{
            padding: 0.6rem 1rem 0.75rem;
            text-align: center;
            border-top: 1px solid var(--border);
            background: var(--card);
        }}
        .panel-label a {{
            color: var(--accent);
            text-decoration: none;
            font-weight: 600;
            font-size: 0.9rem;
            letter-spacing: 0.01em;
        }}
        .panel-label a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title-section">
                <h1>MultiBLiMP v2 &mdash; Word Order Overview</h1>
                <p class="description">
                    This page gives an overview of word order predictability across dependency relations and languages.
                    Each panel shows the base entropy vs. reduced entropy (after fitting a decision tree) for a specific
                    dependency relation across all available languages. Click a panel to explore a dependency relation
                    in detail, or click a point to go directly to a specific language.
                </p>
            </div>
            <div class="controls">
                <label for="entropyType">Entropy type:</label>
                <select id="entropyType">
                    <option value="six" selected>Six-class</option>
                    <option value="binary">Binary (majority vs. rest)</option>
                </select>
            </div>
        </div>

        <div class="grid">
{panels_html}
        </div>
    </div>

    <script>
        const allData = {all_data_json};

        let currentType = 'six';

        function renderMiniPlot(el, points) {{
            const url = el.dataset.url;

            const baseLine = (() => {{
                const vals = points.map(d => d.base);
                const mn = Math.min(...vals), mx = Math.max(...vals);
                return {{ x: [mn, mx], y: [mn, mx] }};
            }})();

            const traceLine = {{
                x: baseLine.x,
                y: baseLine.y,
                mode: 'lines',
                type: 'scatter',
                line: {{ color: '#d4d0cb', width: 1, dash: 'dash' }},
                hoverinfo: 'skip',
                showlegend: false,
            }};

            const tracePoints = {{
                x: points.map(d => d.base),
                y: points.map(d => d.reduced),
                mode: 'markers',
                type: 'scatter',
                text: points.map(d => d.name),
                customdata: points.map(d => [d.url, d.n_items]),
                hovertemplate: '<b>%{{text}}</b><br>Base: %{{x:.3f}}<br>Reduced: %{{y:.3f}}<br>N: %{{customdata[1]:,}}<extra></extra>',
                marker: {{
                    size: points.map(d => d.n_items),
                    sizemode: 'area',
                    sizeref: 2 * Math.max(...points.map(d => d.n_items)) / (20 ** 2),
                    sizemin: 3,
                    color: points.map(d => d.color ?? '#2563eb'),
                    opacity: 0.75,
                    line: {{ 
                        color: points.map(d => d.color ?? '#2563eb'),
                        width: 0.5 
                        }}               
                     }},
                showlegend: false,
            }};

            const layout = {{
                margin: {{ t: 12, r: 12, b: 40, l: 44 }},
                xaxis: {{
                    title: {{ text: 'Base entropy', font: {{ size: 10 }} }},
                    gridcolor: '#e7e5e4',
                    zeroline: false,
                    tickfont: {{ size: 9 }},
                }},
                yaxis: {{
                    title: {{ text: 'Reduced entropy', font: {{ size: 10 }} }},
                    gridcolor: '#e7e5e4',
                    zeroline: false,
                    tickfont: {{ size: 9 }},
                }},
                plot_bgcolor: '#ffffff',
                paper_bgcolor: '#fafaf9',
                font: {{ family: 'DM Sans, system-ui, sans-serif', color: '#1c1917' }},
                hovermode: 'closest',
            }};

            const config = {{
                responsive: true,
                displayModeBar: false,
            }};

            Plotly.newPlot(el, [traceLine, tracePoints], layout, config);

            // Plotly fires its event synchronously before the DOM click bubbles to
            // the panel, so setting this flag here prevents the panel handler below
            // from also navigating when the user clicked a specific data point.
            el.on('plotly_click', function(evt) {{
                const pt = evt.points[0];
                if (pt && pt.customdata && pt.customdata[0]) {{
                    el.parentElement._pointClicked = true;
                    window.location.href = pt.customdata[0];
                }}
            }});
        }}

        function renderAll(type) {{
            document.querySelectorAll('.mini-plot').forEach(el => {{
                const deprel = el.dataset.deprel;
                const data = allData[deprel];
                if (data && data[type]) {{
                    renderMiniPlot(el, data[type]);
                }}
            }});
        }}

        // Set up panel-level navigation once — clicking the panel background
        // navigates to the deprel index, unless a scatter point was clicked.
        document.querySelectorAll('.panel').forEach(panel => {{
            panel.addEventListener('click', function() {{
                if (panel._pointClicked) {{
                    panel._pointClicked = false;
                    return;
                }}
                const url = panel.querySelector('.mini-plot').dataset.url;
                if (url) window.location.href = url;
            }});
        }});

        renderAll('six');

        document.getElementById('entropyType').addEventListener('change', (e) => {{
            currentType = e.target.value;
            renderAll(currentType);
        }});
    </script>
</body>
</html>"""
