import pandas as pd
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx

# Read BOM file and extract U components
def extract_u_components_from_bom(bom_file):
    """Extract components with designators starting with 'U' from BOM"""
    df = pd.read_excel(bom_file)
    u_components = {}
    
    for _, row in df.iterrows():
        designators = str(row['Designators'])
        mpn = str(row['MPN'])
        description = str(row.get('param_Description', ''))
        
        # Split designators by comma
        des_list = [d.strip() for d in designators.split(',')]
        
        for des in des_list:
            # Check if designator starts with 'U' followed by a number
            if re.match(r'^U\d+', des):
                u_components[des] = {
                    'mpn': mpn,
                    'description': description
                }
    
    return u_components

# Parse netlist file
def parse_netlist(netlist_file):
    """Parse netlist file to extract connections"""
    connections = defaultdict(list)
    current_signal = None
    
    with open(netlist_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Check if it's a signal definition
            if line.startswith('*SIGNAL*'):
                current_signal = line.replace('*SIGNAL*', '').strip()
            elif current_signal and line and not line.startswith('*'):
                # Extract component connections
                parts = line.split()
                for part in parts:
                    if '.' in part:
                        component, pin = part.rsplit('.', 1)
                        connections[current_signal].append({
                            'component': component,
                            'pin': pin
                        })
    
    return connections

# Identify power nets
def identify_power_nets(connections):
    """Identify power-related signals"""
    power_keywords = ['VP_', 'VCC', 'VDD', 'V+', 'POWER', 'PWR', '+']
    power_nets = {}
    
    for signal, components in connections.items():
        # Check if signal name contains power keywords
        if any(keyword in signal.upper() for keyword in power_keywords):
            power_nets[signal] = components
    
    return power_nets

# Build power tree
def build_power_tree(u_components, power_nets):
    """Build power tree structure"""
    power_tree = defaultdict(lambda: {'sources': [], 'loads': [], 'voltage': None})
    
    # Power supply components (regulators, converters, etc.)
    power_supply_keywords = ['DC DC', 'CONVERTER', 'REGULATOR', 'POWER SUPPLY', 'MODULE']
    
    for net_name, components in power_nets.items():
        u_comps_in_net = []
        
        for comp_info in components:
            comp = comp_info['component']
            if comp in u_components:
                u_comps_in_net.append({
                    'designator': comp,
                    'pin': comp_info['pin'],
                    'mpn': u_components[comp]['mpn'],
                    'description': u_components[comp]['description']
                })
        
        if u_comps_in_net:
            # Determine if component is a source or load
            for comp in u_comps_in_net:
                desc = comp['description'].upper()
                is_source = any(keyword in desc for keyword in power_supply_keywords)
                
                if is_source:
                    power_tree[net_name]['sources'].append(comp)
                else:
                    power_tree[net_name]['loads'].append(comp)
            
            # Extract voltage from net name
            voltage_match = re.search(r'(\d+\.?\d*)', net_name)
            if voltage_match:
                power_tree[net_name]['voltage'] = voltage_match.group(1) + 'V'
    
    return power_tree

# Visualize power tree
def visualize_power_tree(power_tree, u_components, output_file='power_tree.png'):
    """Create a visual representation of the power tree"""
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Color mapping
    source_color = '#4CAF50'  # Green for sources
    load_color = '#2196F3'    # Blue for loads
    net_color = '#FF9800'     # Orange for power nets
    
    node_colors = {}
    node_labels = {}
    
    # Add nodes and edges
    for net_name, net_info in power_tree.items():
        # Add net node
        net_label = f"{net_name}\n{net_info['voltage'] or ''}"
        G.add_node(net_name, node_type='net')
        node_colors[net_name] = net_color
        node_labels[net_name] = net_label
        
        # Add source nodes
        for source in net_info['sources']:
            node_id = source['designator']
            G.add_node(node_id, node_type='source')
            node_colors[node_id] = source_color
            node_labels[node_id] = f"{node_id}\n{source['mpn'][:20]}"
            G.add_edge(node_id, net_name)
        
        # Add load nodes
        for load in net_info['loads']:
            node_id = load['designator']
            G.add_node(node_id, node_type='load')
            node_colors[node_id] = load_color
            node_labels[node_id] = f"{node_id}\n{load['mpn'][:20]}"
            G.add_edge(net_name, node_id)
    
    # Create layout
    plt.figure(figsize=(20, 14))
    
    # Use hierarchical layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes
    for node in G.nodes():
        color = node_colors.get(node, '#CCCCCC')
        nx.draw_networkx_nodes(G, pos, nodelist=[node], 
                              node_color=color, 
                              node_size=3000,
                              alpha=0.9)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=20, 
                          arrowstyle='->', width=2,
                          connectionstyle='arc3,rad=0.1')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, node_labels, font_size=8, font_weight='bold')
    
    # Add legend
    source_patch = mpatches.Patch(color=source_color, label='Power Sources (Converters/Regulators)')
    load_patch = mpatches.Patch(color=load_color, label='Power Loads (ICs)')
    net_patch = mpatches.Patch(color=net_color, label='Power Nets')
    plt.legend(handles=[source_patch, load_patch, net_patch], loc='upper left', fontsize=10)
    
    plt.title('Power Tree Diagram', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Power tree diagram saved to {output_file}")
    
    return G

# Generate text report
def generate_report(power_tree, u_components):
    """Generate a detailed text report of the power tree"""
    report = []
    report.append("=" * 80)
    report.append("POWER TREE ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary
    total_nets = len(power_tree)
    total_sources = sum(len(net['sources']) for net in power_tree.values())
    total_loads = sum(len(net['loads']) for net in power_tree.values())
    
    report.append(f"Summary:")
    report.append(f"  - Total Power Nets: {total_nets}")
    report.append(f"  - Total Power Sources: {total_sources}")
    report.append(f"  - Total Power Loads: {total_loads}")
    report.append("")
    
    # Detailed breakdown
    for net_name, net_info in sorted(power_tree.items()):
        report.append("-" * 80)
        report.append(f"Power Net: {net_name}")
        if net_info['voltage']:
            report.append(f"Voltage: {net_info['voltage']}")
        report.append("")
        
        if net_info['sources']:
            report.append("  SOURCES (Power Suppliers):")
            for source in net_info['sources']:
                report.append(f"    • {source['designator']} (Pin {source['pin']})")
                report.append(f"      MPN: {source['mpn']}")
                report.append(f"      Description: {source['description']}")
                report.append("")
        
        if net_info['loads']:
            report.append("  LOADS (Power Consumers):")
            for load in net_info['loads']:
                report.append(f"    • {load['designator']} (Pin {load['pin']})")
                report.append(f"      MPN: {load['mpn']}")
                report.append(f"      Description: {load['description']}")
                report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)

# Main execution
def main():
    print("Starting Power Tree Analysis...")
    print("-" * 80)
    
    # Step 1: Extract U components from BOM
    print("\n1. Extracting U components from BOM...")
    u_components = extract_u_components_from_bom('bom.xlsx')
    print(f"   Found {len(u_components)} U components")
    
    # Step 2: Parse netlist
    print("\n2. Parsing netlist...")
    connections = parse_netlist('netlist.net')
    print(f"   Found {len(connections)} signal connections")
    
    # Step 3: Identify power nets
    print("\n3. Identifying power nets...")
    power_nets = identify_power_nets(connections)
    print(f"   Found {len(power_nets)} power nets")
    
    # Step 4: Build power tree
    print("\n4. Building power tree...")
    power_tree = build_power_tree(u_components, power_nets)
    print(f"   Power tree built with {len(power_tree)} power domains")
    
    # Step 5: Generate report
    print("\n5. Generating report...")
    report = generate_report(power_tree, u_components)
    
    # Save report to file
    with open('power_tree_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("   Report saved to power_tree_report.txt")
    
    # Step 6: Visualize power tree
    print("\n6. Creating visualization...")
    visualize_power_tree(power_tree, u_components)
    
    print("\n" + "=" * 80)
    print("Power Tree Analysis Complete!")
    print("=" * 80)
    
    # Print report to console
    print("\n" + report)

if __name__ == "__main__":
    main()
