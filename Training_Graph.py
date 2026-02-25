import streamlit as st
import json
from utils import get_models, get_lineage, get_weights

def render_page():
    st.set_page_config(page_title="Training Graph", layout="wide")

    st.title("Training Graph")
    st.markdown("Visualize the lineage of model weights trained on specific datasets.")

    models = get_models()
    if not models:
        st.info("No models found. Please create models and run training jobs first.")
    else:
        selected_model = st.selectbox("Select Model to view lineage", models)

        if selected_model:
            lineage_data = get_lineage(selected_model)

            # If there's no actual lineage data, we can build a mock one based on existing weights for demonstration
            if not lineage_data:
                st.info("No real training lineage found for this model. Building a mock JSON from existing weights for demonstration...")
                weights = get_weights(selected_model)
                if weights:
                    # Mock lineage: Random Init -> v1 -> v2 -> v3
                    sorted_weights = sorted([w['Filename'] for w in weights])
                    for i, w in enumerate(sorted_weights):
                        source = sorted_weights[i-1] if i > 0 else "Random Initialization"
                        lineage_data.append({
                            "source_weight": source,
                            "dataset": "Mock_Dataset_A" if i % 2 == 0 else "Mock_Dataset_B",
                            "target_weight": w,
                            "timestamp": "N/A"
                        })
                else:
                    st.warning("No weights available to build a graph.")

            if lineage_data:
                st.subheader("Training Lineage Graph")

                # Generate DOT format string for Graphviz
                dot = ["digraph Lineage {"]
                dot.append('  rankdir="LR";') # Left to right layout
                dot.append('  node [shape=box, style=filled, fillcolor=lightblue];')
                dot.append('  edge [color=gray50, fontcolor=blue, fontsize=10];')

                for entry in lineage_data:
                    src = entry['source_weight']
                    dst = entry['target_weight']
                    edge_label = entry['dataset']

                    # Clean up node names for DOT syntax (quotes handle special chars)
                    dot.append(f'  "{src}" -> "{dst}" [label="{edge_label}"];')

                dot.append("}")
                dot_string = "\n".join(dot)

                # Render using Streamlit's built-in graphviz chart
                st.graphviz_chart(dot_string, use_container_width=True)
