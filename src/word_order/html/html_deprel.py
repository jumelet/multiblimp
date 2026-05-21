def create_html(rows_six, rows_binary, plot_data_six_json, plot_data_binary_json):
    return f"""<!DOCTYPE html>
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
            .back-btn {{
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.45rem 0.875rem;
                border: 1px solid var(--border);
                border-radius: 6px;
                background: white;
                color: #57534e;
                font-family: 'DM Sans', system-ui, sans-serif;
                font-size: 0.875rem;
                font-weight: 500;
                text-decoration: none;
                transition: border-color 0.15s, color 0.15s, background 0.15s;
            }}
            .back-btn:hover {{
                border-color: var(--accent);
                color: var(--accent);
                background: #eff6ff;
                text-decoration: none;
            }}
            .back-btn svg {{
                flex-shrink: 0;
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
                    <a href="../" class="back-btn">
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M10 12L6 8L10 4" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                        Overview
                    </a>
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
                    customdata: data.map(d => [d.url, d.n_items]),
                    hovertemplate: '<b>%{{text}}</b><br>' +
                                'Base Entropy: %{{x:.3f}}<br>' +
                                'Reduced Entropy: %{{y:.3f}}<br>' +
                                'N Items: %{{customdata[1]:,}}<br>' +
                                '<extra></extra>',
                    marker: {{
                        size: data.map(d => d.n_items),
                        sizemode: 'area',
                        sizeref: 2 * Math.max(...data.map(d => d.n_items)) / (40 ** 2),
                        sizemin: 4,
                        color: '#2563eb',
                        opacity: 0.6,
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
                    const url = data.points[0].customdata[0];
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
