import streamlit as st
import time
import tempfile
import os
from utils import get_models, get_weights, get_datasets, save_model_weight, record_training_lineage, get_lot_codes

class MockUploadedFile:
    def __init__(self, name, path):
        self.name = name
        self.path = path
    def getbuffer(self):
        with open(self.path, "rb") as f:
            return f.read()

def render_page():
    st.set_page_config(page_title="Training")

    st.title("Model Training")
    st.markdown("Configure and start a new training job.")

    lot_codes = get_lot_codes()
    st.markdown("### 🎯 Active Workspace")
    selected_lot_env = st.selectbox("Select Lot to operate within:", lot_codes)
    
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Select Model")
        models = get_models()
        if not models:
            st.warning("No models available.")
            selected_model = None
        else:
            selected_model = st.selectbox("Choose a model architecture", models)
            use_pretrained = False
            selected_weight = None
            if selected_model:
                weights = get_weights(selected_model)
                lot_weights = [w for w in weights if w['Lot'] == selected_lot_env]
                
                use_pretrained = st.checkbox("Initialize from existing weights")
                if use_pretrained and lot_weights:
                    def format_weight(w):
                        return f"{w['Filename']} ({w['Version']})"
                    selected_weight_obj = st.selectbox("Choose starting weights", lot_weights, format_func=format_weight)
                    selected_weight = selected_weight_obj['Filename'] if selected_weight_obj else None
                elif use_pretrained and not lot_weights:
                    st.warning(f"No pre-trained weights available for this model in Lot `{selected_lot_env}`.")

    with col2:
        st.subheader("2. Select Dataset")
        datasets = get_datasets()
        if not datasets:
            st.warning("No datasets available.")
            selected_dataset = None
        else:
            dataset_names = [ds['Name'] for ds in datasets]
            selected_dataset = st.selectbox("Choose a dataset", dataset_names)

    st.divider()

    st.subheader("3. Training Configuration")
    config_col1, config_col2 = st.columns(2)
    with config_col1:
        epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=10)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=512, value=32)
        lr = st.number_input("Learning Rate", min_value=1e-6, max_value=1.0, value=0.001, format="%.5f")

    with config_col2:
        st.write("**Early Stopping**")
        use_early_stopping = st.checkbox("Enable Early Stopping", value=False)
        if use_early_stopping:
            patience = st.number_input("Patience (epochs)", min_value=1, max_value=100, value=5)
            min_delta = st.number_input("Min Delta", min_value=0.0, max_value=1.0, value=0.001, format="%.4f")

    st.divider()
    st.divider()
    
    try:
        default_index = lot_codes.index(selected_lot_env)
    except ValueError:
        default_index = 0
        
    save_to_lot = st.selectbox("Save new weight to Lot", lot_codes, index=default_index)

    st.divider()

    if st.button("Start Training", type="primary", disabled=(not models or not datasets)):
        st.info(f"Starting training job for model **{selected_model}** on dataset **{selected_dataset}**...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        best_loss = float('inf')
        wait = 0

        for i in range(epochs):
            # mock training delay
            time.sleep(0.5)

            # Mock loss getting smaller
            current_loss = max(0.05, 1.0 - (i*0.08)) 

            progress = (i + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {i+1}/{epochs} - Loss: {current_loss:.4f}")

            # Early Stopping Logic Simulation
            if use_early_stopping:
                if current_loss < best_loss - min_delta:
                    best_loss = current_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        st.warning(f"Early stopping triggered at epoch {i+1}!")
                        progress_bar.progress(1.0)
                        break


        # Mock saving a new weight 
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
            tmp.write(b"mock weight data")
            tmp_path = tmp.name

        base_name = selected_weight if (use_pretrained and selected_weight) else None
        dummy_file = MockUploadedFile("trained_weight.pth", tmp_path)

        # Save the weight logic assigns the _v# properly
        new_weight_path = save_model_weight(selected_model, save_to_lot, dummy_file)
        new_weight_filename = os.path.basename(new_weight_path)

        # Record Lineage Graph
        record_training_lineage(selected_model, base_name, selected_dataset, new_weight_filename)
        os.remove(tmp_path)

        st.success(f"Training completed successfully! New weight `{new_weight_filename}` saved and graph updated.")
