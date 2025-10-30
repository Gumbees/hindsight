"""
Interactive HTML graph visualization of memory system.

Uses Cytoscape.js to create a performant, interactive network graph that can be
explored in the browser. Shows all memory units and their links with weights.
"""
import psycopg2
from dotenv import load_dotenv
import os
import json

load_dotenv()


def create_interactive_graph():
    """Create an interactive HTML graph visualization using Cytoscape.js."""

    # Connect to database
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cursor = conn.cursor()

    # Get all memory units (no agent_id filter)
    cursor.execute("""
        SELECT id, text, event_date, context
        FROM memory_units
        ORDER BY event_date
    """)
    units = cursor.fetchall()

    # Get all links with weights (no agent_id filter)
    cursor.execute("""
        SELECT
            ml.from_unit_id,
            ml.to_unit_id,
            ml.link_type,
            ml.weight,
            e.canonical_name as entity_name
        FROM memory_links ml
        LEFT JOIN entities e ON ml.entity_id = e.id
        ORDER BY ml.link_type, ml.weight DESC
    """)
    links = cursor.fetchall()

    # Get entity information (no agent_id filter)
    cursor.execute("""
        SELECT ue.unit_id, e.canonical_name, e.entity_type
        FROM unit_entities ue
        JOIN entities e ON ue.entity_id = e.id
        ORDER BY ue.unit_id
    """)
    unit_entities = cursor.fetchall()

    cursor.close()
    conn.close()

    # Build entity mapping
    entity_map = {}
    for unit_id, entity_name, entity_type in unit_entities:
        if unit_id not in entity_map:
            entity_map[unit_id] = []
        entity_map[unit_id].append(f"{entity_name} ({entity_type})")

    # Build Cytoscape.js graph data
    cy_nodes = []
    cy_edges = []

    # Add nodes
    for unit_id, text, event_date, context in units:
        entities = entity_map.get(unit_id, [])
        entity_count = len(entities)

        # Color by entity count
        if entity_count == 0:
            color = "#e0e0e0"
        elif entity_count == 1:
            color = "#90caf9"
        else:
            color = "#42a5f5"

        cy_nodes.append({
            "data": {
                "id": str(unit_id),
                "label": text[:50] + "..." if len(text) > 50 else text,
                "text": text,
                "context": context,
                "date": str(event_date.date()),
                "entities": ", ".join(entities) if entities else "None",
                "color": color
            }
        })

    # Add edges
    for from_id, to_id, link_type, weight, entity_name in links:
        # Set color based on link type
        if link_type == 'temporal':
            color = "#00bcd4"
            line_style = "dashed"
        elif link_type == 'semantic':
            color = "#ff69b4"
            line_style = "solid"
        elif link_type == 'entity':
            color = "#ffd700"
            line_style = "solid"
        else:
            color = "#999999"
            line_style = "solid"

        cy_edges.append({
            "data": {
                "id": f"{from_id}-{to_id}-{link_type}",
                "source": str(from_id),
                "target": str(to_id),
                "weight": weight,
                "linkType": link_type,
                "entityName": entity_name or "",
                "color": color,
                "lineStyle": line_style
            }
        })

    graph_data = {"nodes": cy_nodes, "edges": cy_edges}

    # Build table rows for table view
    table_rows = []
    for unit_id, text, event_date, context in units:
        entities = entity_map.get(unit_id, [])
        entity_str = ", ".join(entities) if entities else "None"
        table_rows.append(f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">{str(unit_id)[:8]}...</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{text}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{context}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{event_date.date()}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{entity_str}</td>
            </tr>
        """)

    # Generate HTML with Cytoscape.js
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Memory Graph - Interactive Visualization</title>
    <meta charset="utf-8">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
    <style>
        body {{
            font-family: Tahoma, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
        }}

        .tab-container {{
            background: white;
        }}

        .tab-buttons {{
            background: #f0f0f0;
            border-bottom: 2px solid #333;
            padding: 0;
            margin: 0;
        }}

        .tab-button {{
            background: #e0e0e0;
            border: none;
            padding: 12px 24px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            border-top: 2px solid transparent;
            border-left: 2px solid transparent;
            border-right: 2px solid transparent;
            transition: background 0.2s;
        }}

        .tab-button:hover {{
            background: #d0d0d0;
        }}

        .tab-button.active {{
            background: white;
            border-top: 2px solid #333;
            border-left: 2px solid #333;
            border-right: 2px solid #333;
            border-bottom: 2px solid white;
            margin-bottom: -2px;
        }}

        .tab-content {{
            display: none;
            background: white;
        }}

        .tab-content.active {{
            display: block;
        }}

        #cy {{
            width: 100%;
            height: 800px;
            background: #ffffff;
        }}

        #graph-tab {{
            position: relative;
        }}

        #table-tab {{
            padding: 20px;
        }}

        .legend {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: white;
            padding: 15px;
            border: 2px solid #333;
            border-radius: 8px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
            z-index: 1000;
            max-width: 250px;
        }}

        .legend h3 {{
            margin-top: 0;
            border-bottom: 2px solid #333;
            padding-bottom: 5px;
        }}

        .legend-item {{
            margin: 8px 0;
            display: flex;
            align-items: center;
        }}

        .legend-line {{
            width: 30px;
            height: 2px;
            margin-right: 10px;
        }}

        .legend-node {{
            width: 20px;
            height: 20px;
            margin-right: 10px;
            border: 1px solid #999;
            border-radius: 3px;
        }}

        #table-filter {{
            width: 100%;
            max-width: 600px;
            padding: 10px;
            margin-bottom: 15px;
            border: 2px solid #ccc;
            border-radius: 4px;
            font-size: 14px;
            box-sizing: border-box;
        }}

        #memory-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            max-width: 1400px;
        }}

        #memory-table th {{
            padding: 10px;
            text-align: left;
            border: 1px solid #ddd;
            background: #f0f0f0;
        }}

        #memory-table td {{
            padding: 8px;
            border: 1px solid #ddd;
        }}

        .tooltip {{
            position: absolute;
            background: white;
            border: 2px solid #333;
            border-radius: 4px;
            padding: 10px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
            max-width: 300px;
            font-size: 12px;
            pointer-events: none;
            z-index: 9999;
        }}
    </style>
</head>
<body>
    <div class="tab-container">
        <div class="tab-buttons">
            <button class="tab-button active" onclick="switchTab('graph')">Graph View</button>
            <button class="tab-button" onclick="switchTab('table')">Table View</button>
        </div>

        <div id="graph-tab" class="tab-content active">
            <div style="padding: 15px; background: #f9f9f9; border-bottom: 2px solid #333;">
                <div style="display: flex; gap: 15px; align-items: center; flex-wrap: wrap;">
                    <div>
                        <label style="font-weight: bold; margin-right: 5px;">Limit nodes:</label>
                        <input type="number" id="node-limit" value="50" min="10" max="1000" step="10"
                               style="width: 80px; padding: 5px; border: 1px solid #ccc; border-radius: 4px;">
                    </div>
                    <div>
                        <label style="font-weight: bold; margin-right: 5px;">Layout:</label>
                        <select id="layout-select" style="padding: 5px; border: 1px solid #ccc; border-radius: 4px;">
                            <option value="circle">Circle (fast)</option>
                            <option value="grid">Grid (fast)</option>
                            <option value="cose">Force-directed (slow)</option>
                        </select>
                    </div>
                    <button onclick="reloadGraph()" style="padding: 6px 15px; background: #42a5f5; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">
                        Apply
                    </button>
                    <span id="node-count" style="color: #666; font-size: 14px;"></span>
                </div>
            </div>
            <div id="cy"></div>
            <div class="legend">
                <h3>Legend</h3>
                <h4 style="margin: 10px 0 5px 0;">Link Types:</h4>
                <div class="legend-item">
                    <div class="legend-line" style="background: #00bcd4; border-top: 1px dashed #00bcd4;"></div>
                    <span><b>Temporal</b></span>
                </div>
                <div class="legend-item">
                    <div class="legend-line" style="background: #ff69b4;"></div>
                    <span><b>Semantic</b></span>
                </div>
                <div class="legend-item">
                    <div class="legend-line" style="background: #ffd700;"></div>
                    <span><b>Entity</b></span>
                </div>
                <h4 style="margin: 15px 0 5px 0;">Nodes:</h4>
                <div class="legend-item">
                    <div class="legend-node" style="background: #e0e0e0;"></div>
                    <span>No entities</span>
                </div>
                <div class="legend-item">
                    <div class="legend-node" style="background: #90caf9;"></div>
                    <span>1 entity</span>
                </div>
                <div class="legend-item">
                    <div class="legend-node" style="background: #42a5f5;"></div>
                    <span>2+ entities</span>
                </div>
            </div>
        </div>

        <div id="table-tab" class="tab-content">
            <h2>Memory Units ({len(units)})</h2>
            <input type="text" id="table-filter" placeholder="Filter by text, context, or entities...">
            <div style="overflow-x: auto;">
                <table id="memory-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Text</th>
                            <th>Context</th>
                            <th>Date</th>
                            <th>Entities</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(table_rows)}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        // Graph data
        const allGraphData = {json.dumps(graph_data)};
        let cy = null;

        // Initialize graph with filtering
        function initGraph(nodeLimit, layoutName) {{
            // Filter nodes to limit
            const limitedNodes = allGraphData.nodes.slice(0, nodeLimit);
            const nodeIds = new Set(limitedNodes.map(n => n.data.id));

            // Filter edges to only include those between visible nodes
            const limitedEdges = allGraphData.edges.filter(e =>
                nodeIds.has(e.data.source) && nodeIds.has(e.data.target)
            );

            // Update count display
            document.getElementById('node-count').textContent =
                `Showing ${{limitedNodes.length}} of ${{allGraphData.nodes.length}} nodes`;

            // Destroy existing graph if any
            if (cy) {{
                cy.destroy();
            }}

            // Layout configurations
            const layouts = {{
                'circle': {{
                    name: 'circle',
                    animate: false,
                    radius: 300,
                    spacingFactor: 1.5
                }},
                'grid': {{
                    name: 'grid',
                    animate: false,
                    rows: Math.ceil(Math.sqrt(limitedNodes.length)),
                    cols: Math.ceil(Math.sqrt(limitedNodes.length)),
                    spacingFactor: 2
                }},
                'cose': {{
                    name: 'cose',
                    animate: false,
                    nodeRepulsion: 15000,
                    idealEdgeLength: 150,
                    edgeElasticity: 100,
                    nestingFactor: 1.2,
                    gravity: 1,
                    numIter: 1000,
                    initialTemp: 200,
                    coolingFactor: 0.95,
                    minTemp: 1.0
                }}
            }};

            // Initialize Cytoscape
            cy = cytoscape({{
                container: document.getElementById('cy'),

                elements: [
                    ...limitedNodes.map(n => ({{ data: n.data }})),
                    ...limitedEdges.map(e => ({{ data: e.data }}))
                ],

                style: [
                    {{
                        selector: 'node',
                        style: {{
                            'background-color': 'data(color)',
                            'label': 'data(label)',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'font-size': '10px',
                            'font-weight': 'bold',
                            'text-wrap': 'wrap',
                            'text-max-width': '100px',
                            'width': 40,
                            'height': 40,
                            'border-width': 2,
                            'border-color': '#333'
                        }}
                    }},
                    {{
                        selector: 'edge',
                        style: {{
                            'width': 1,
                            'line-color': 'data(color)',
                            'line-style': 'data(lineStyle)',
                            'target-arrow-shape': 'triangle',
                            'target-arrow-color': 'data(color)',
                            'curve-style': 'bezier',
                            'opacity': 0.7
                        }}
                    }},
                    {{
                        selector: 'node:selected',
                        style: {{
                            'border-width': 4,
                            'border-color': '#000'
                        }}
                    }}
                ],

                layout: layouts[layoutName] || layouts['circle']
            }});

            // Simple tooltip on hover
            let tooltip = null;

            cy.on('mouseover', 'node', function(evt) {{
                const node = evt.target;
                const data = node.data();
                const renderedPosition = node.renderedPosition();

                // Create tooltip
                tooltip = document.createElement('div');
                tooltip.className = 'tooltip';
                tooltip.innerHTML = `
                    <b>Text:</b> ${{data.text}}<br>
                    <b>Context:</b> ${{data.context}}<br>
                    <b>Date:</b> ${{data.date}}<br>
                    <b>Entities:</b> ${{data.entities}}
                `;
                tooltip.style.left = renderedPosition.x + 20 + 'px';
                tooltip.style.top = renderedPosition.y + 'px';
                document.body.appendChild(tooltip);
            }});

            cy.on('mouseout', 'node', function(evt) {{
                if (tooltip) {{
                    tooltip.remove();
                    tooltip = null;
                }}
            }});
        }}

        // Reload graph with current settings
        function reloadGraph() {{
            const nodeLimit = parseInt(document.getElementById('node-limit').value) || 50;
            const layoutName = document.getElementById('layout-select').value;
            initGraph(nodeLimit, layoutName);
        }}

        // Initialize with default settings (50 nodes, circle layout)
        initGraph(50, 'circle');

        // Tab switching
        function switchTab(tabName) {{
            document.querySelectorAll('.tab-content').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.querySelectorAll('.tab-button').forEach(btn => {{
                btn.classList.remove('active');
            }});

            if (tabName === 'graph') {{
                document.getElementById('graph-tab').classList.add('active');
                document.querySelectorAll('.tab-button')[0].classList.add('active');
                cy.resize();  // Resize graph when switching to it
            }} else if (tabName === 'table') {{
                document.getElementById('table-tab').classList.add('active');
                document.querySelectorAll('.tab-button')[1].classList.add('active');
            }}
        }}

        // Table filtering
        document.getElementById('table-filter').addEventListener('input', function() {{
            const filterValue = this.value.toLowerCase();
            const rows = document.querySelectorAll('#memory-table tbody tr');

            rows.forEach(row => {{
                const text = row.textContent.toLowerCase();
                if (text.includes(filterValue)) {{
                    row.style.display = '';
                }} else {{
                    row.style.display = 'none';
                }}
            }});
        }});
    </script>
</body>
</html>
"""

    # Write HTML file
    output_file = "memory_graph_interactive.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # Print summary
    print(f"\n{'='*80}")
    print("INTERACTIVE GRAPH GENERATED (Cytoscape.js)")
    print(f"{'='*80}")
    print(f"\nFile: {output_file}")
    print(f"Units: {len(units)}")
    print(f"Links: {len(links)}")
    print("\nFeatures:")
    print("  • Tab 1: Graph View - Fast interactive network (Cytoscape.js)")
    print("    - Limit nodes (default: 50) for better performance")
    print("    - Choose layout: Circle (fast), Grid (fast), or Force-directed")
    print("    - Drag nodes, zoom, pan")
    print("    - Hover for details")
    print("  • Tab 2: Table View - Searchable memory units")
    print("    - Filter by text, context, or entities")
    print("    - Case-insensitive search")
    print("    - Shows ALL nodes")
    print(f"\n{'='*80}")
    print(f"✓ Open {output_file} in your browser to explore!")
    print(f"  TIP: Start with 50 nodes and Circle layout for best performance")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    create_interactive_graph()
