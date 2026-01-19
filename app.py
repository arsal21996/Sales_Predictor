# app.py
import streamlit as st
import joblib
import pandas as pd
import torch

st.title("🎲 Synthetic Sales Data Generator")
st.write("Generate realistic synthetic sales data instantly!")

# Load your pre-trained model (CPU-compatible)
@st.cache_resource
def load_pretrained_model():
    # Force CPU loading even if model was trained on GPU
    return joblib.load('tvae_model_joblib.pkl')

# Monkey-patch torch.load to always use CPU
original_torch_load = torch.load
torch.load = lambda *args, **kwargs: original_torch_load(*args, **{**kwargs, 'map_location': 'cpu'})

model = load_pretrained_model()
st.success("✅ Pre-trained model loaded and ready!")

# User controls
st.subheader("Generate Synthetic Data")
num_samples = st.slider("Number of samples to generate:", 10, 10000, 1000)

if st.button("🚀 Generate Now"):
    with st.spinner("Generating synthetic data..."):
        synthetic_data = model.sample(num_samples)
    
    st.success(f"✅ Generated {len(synthetic_data)} synthetic samples!")
    
    # Show preview
    st.dataframe(synthetic_data.head(20))
    
    # Download button
    csv = synthetic_data.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="synthetic_sales_data.csv",
        mime="text/csv"
    )
