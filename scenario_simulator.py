# scenario_simulator.py
import streamlit as st
import joblib
import pandas as pd
import torch
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Sales Scenario Simulator", page_icon="🎯", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎯 Sales Scenario Simulator</h1>', unsafe_allow_html=True)
st.markdown("### Experiment with different business scenarios and predict sales outcomes")

# Monkey-patch torch.load for CPU compatibility
original_torch_load = torch.load
torch.load = lambda *args, **kwargs: original_torch_load(*args, **{**kwargs, 'map_location': 'cpu'})

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load('tvae_model_joblib.pkl')
    except FileNotFoundError:
        st.error("❌ Model file 'tvae_model_joblib.pkl' not found! Please ensure it's in the same directory.")
        st.stop()

model = load_model()

# Sidebar for scenario configuration
st.sidebar.header("🎛️ Scenario Configuration")

scenario_type = st.sidebar.selectbox(
    "Choose Scenario Type:",
    ["Single Scenario", "Batch Comparison", "Sensitivity Analysis"]
)

st.sidebar.markdown("---")

if scenario_type == "Single Scenario":
    st.header("📊 Single Scenario Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        marketing_spend = st.slider("Marketing Spend ($)", 500, 5000, 2000, 100)
        temperature = st.slider("Temperature (°C)", 0, 40, 20, 1)
        is_holiday = st.selectbox("Holiday?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        day_of_week = st.selectbox("Day of Week", ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        product_category = st.selectbox("Product Category", ['Electronics', 'Clothing', 'Food'])
    
    if st.button("🚀 Predict Sales", type="primary"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'marketing_spend': [marketing_spend],
            'temperature': [temperature],
            'is_holiday': [is_holiday],
            'day_of_week': [day_of_week],
            'product_category': [product_category],
            'sales': [0]  # Placeholder
        })
        
        # Generate synthetic data with similar conditions (sampling approach)
        with st.spinner("Running simulation..."):
            synthetic_samples = model.sample(1000)
            
            # Filter samples close to our conditions
            filtered = synthetic_samples[
                (synthetic_samples['marketing_spend'].between(marketing_spend - 200, marketing_spend + 200)) &
                (synthetic_samples['temperature'].between(temperature - 3, temperature + 3)) &
                (synthetic_samples['is_holiday'] == is_holiday) &
                (synthetic_samples['day_of_week'] == day_of_week) &
                (synthetic_samples['product_category'] == product_category)
            ]
            
            if len(filtered) > 0:
                predicted_sales = filtered['sales'].mean()
                sales_std = filtered['sales'].std()
                sales_min = filtered['sales'].min()
                sales_max = filtered['sales'].max()
            else:
                # Fallback: use broader criteria
                filtered = synthetic_samples[
                    (synthetic_samples['is_holiday'] == is_holiday) &
                    (synthetic_samples['product_category'] == product_category)
                ]
                predicted_sales = filtered['sales'].mean()
                sales_std = filtered['sales'].std()
                sales_min = filtered['sales'].min()
                sales_max = filtered['sales'].max()
    
        with col2:
            st.subheader("Predicted Results")
            
            # Metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Predicted Sales", f"${predicted_sales:,.0f}")
            with col_b:
                st.metric("Best Case", f"${sales_max:,.0f}")
            with col_c:
                st.metric("Worst Case", f"${sales_min:,.0f}")
            
            # Distribution chart
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=filtered['sales'],
                name='Similar Scenarios',
                marker_color='#667eea',
                opacity=0.7
            ))
            fig.add_vline(x=predicted_sales, line_dash="dash", line_color="red", 
                         annotation_text=f"Predicted: ${predicted_sales:,.0f}")
            fig.update_layout(
                title="Sales Distribution for Similar Scenarios",
                xaxis_title="Sales ($)",
                yaxis_title="Frequency",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"📊 Prediction based on {len(filtered)} similar scenarios from synthetic data")

elif scenario_type == "Batch Comparison":
    st.header("📈 Compare Multiple Scenarios")
    
    st.markdown("Compare sales predictions across different scenarios side-by-side")
    
    num_scenarios = st.slider("Number of scenarios to compare:", 2, 5, 3)
    
    scenarios = []
    cols = st.columns(num_scenarios)
    
    for i, col in enumerate(cols):
        with col:
            st.subheader(f"Scenario {i+1}")
            marketing = st.number_input(f"Marketing ($) #{i+1}", 500, 5000, 1500 + i*500, 100, key=f"mkt_{i}")
            temp = st.number_input(f"Temp (°C) #{i+1}", 0, 40, 15 + i*5, 1, key=f"temp_{i}")
            holiday = st.selectbox(f"Holiday #{i+1}", [0, 1], key=f"hol_{i}")
            day = st.selectbox(f"Day #{i+1}", ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], key=f"day_{i}")
            cat = st.selectbox(f"Category #{i+1}", ['Electronics', 'Clothing', 'Food'], key=f"cat_{i}")
            
            scenarios.append({
                'name': f'Scenario {i+1}',
                'marketing_spend': marketing,
                'temperature': temp,
                'is_holiday': holiday,
                'day_of_week': day,
                'product_category': cat
            })
    
    if st.button("🔍 Compare All Scenarios", type="primary"):
        results = []
        
        with st.spinner("Analyzing scenarios..."):
            synthetic_samples = model.sample(2000)
            
            for scenario in scenarios:
                filtered = synthetic_samples[
                    (synthetic_samples['marketing_spend'].between(scenario['marketing_spend'] - 300, scenario['marketing_spend'] + 300)) &
                    (synthetic_samples['temperature'].between(scenario['temperature'] - 5, scenario['temperature'] + 5)) &
                    (synthetic_samples['is_holiday'] == scenario['is_holiday']) &
                    (synthetic_samples['day_of_week'] == scenario['day_of_week']) &
                    (synthetic_samples['product_category'] == scenario['product_category'])
                ]
                
                if len(filtered) > 0:
                    results.append({
                        'Scenario': scenario['name'],
                        'Predicted Sales': filtered['sales'].mean(),
                        'Marketing Spend': scenario['marketing_spend'],
                        'Temperature': scenario['temperature'],
                        'Category': scenario['product_category']
                    })
        
        # Display results
        st.markdown("### 📊 Comparison Results")
        
        results_df = pd.DataFrame(results)
        
        # Bar chart
        fig = px.bar(results_df, x='Scenario', y='Predicted Sales', 
                     color='Predicted Sales',
                     color_continuous_scale='Viridis',
                     title='Sales Predictions Comparison')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.dataframe(results_df.style.format({
            'Predicted Sales': '${:,.0f}',
            'Marketing Spend': '${:,.0f}',
            'Temperature': '{:.1f}°C'
        }), use_container_width=True)
        
        # Best scenario
        best_scenario = results_df.loc[results_df['Predicted Sales'].idxmax()]
        st.success(f"🏆 Best Scenario: **{best_scenario['Scenario']}** with predicted sales of **${best_scenario['Predicted Sales']:,.0f}**")

else:  # Sensitivity Analysis
    st.header("🔬 Sensitivity Analysis")
    st.markdown("See how changes in one variable affect sales predictions")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Base Scenario")
        base_marketing = st.slider("Base Marketing ($)", 500, 5000, 2000, 100)
        base_temp = st.slider("Base Temperature (°C)", 0, 40, 20, 1)
        base_holiday = st.selectbox("Holiday?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        base_day = st.selectbox("Day", ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        base_category = st.selectbox("Category", ['Electronics', 'Clothing', 'Food'])
        
        variable_to_test = st.selectbox(
            "Variable to Analyze:",
            ["Marketing Spend", "Temperature"]
        )
    
    if st.button("📊 Run Sensitivity Analysis", type="primary"):
        with col2:
            with st.spinner("Running analysis..."):
                synthetic_samples = model.sample(3000)
                
                if variable_to_test == "Marketing Spend":
                    test_values = range(500, 5001, 250)
                    results = []
                    
                    for mkt in test_values:
                        filtered = synthetic_samples[
                            (synthetic_samples['marketing_spend'].between(mkt - 200, mkt + 200)) &
                            (synthetic_samples['is_holiday'] == base_holiday) &
                            (synthetic_samples['day_of_week'] == base_day) &
                            (synthetic_samples['product_category'] == base_category)
                        ]
                        if len(filtered) > 0:
                            results.append({
                                'Marketing Spend': mkt,
                                'Predicted Sales': filtered['sales'].mean()
                            })
                    
                    results_df = pd.DataFrame(results)
                    
                    fig = px.line(results_df, x='Marketing Spend', y='Predicted Sales',
                                 title='Sales Sensitivity to Marketing Spend',
                                 markers=True)
                    fig.add_vline(x=base_marketing, line_dash="dash", line_color="red",
                                 annotation_text="Current")
                    
                else:  # Temperature
                    test_values = range(0, 41, 2)
                    results = []
                    
                    for temp in test_values:
                        filtered = synthetic_samples[
                            (synthetic_samples['temperature'].between(temp - 3, temp + 3)) &
                            (synthetic_samples['is_holiday'] == base_holiday) &
                            (synthetic_samples['day_of_week'] == base_day) &
                            (synthetic_samples['product_category'] == base_category)
                        ]
                        if len(filtered) > 0:
                            results.append({
                                'Temperature': temp,
                                'Predicted Sales': filtered['sales'].mean()
                            })
                    
                    results_df = pd.DataFrame(results)
                    
                    fig = px.line(results_df, x='Temperature', y='Predicted Sales',
                                 title='Sales Sensitivity to Temperature',
                                 markers=True)
                    fig.add_vline(x=base_temp, line_dash="dash", line_color="red",
                                 annotation_text="Current")
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Insights
                st.markdown("### 💡 Insights")
                if variable_to_test == "Marketing Spend":
                    roi = (results_df['Predicted Sales'].iloc[-1] - results_df['Predicted Sales'].iloc[0]) / (results_df['Marketing Spend'].iloc[-1] - results_df['Marketing Spend'].iloc[0])
                    st.info(f"📈 Estimated ROI: For every $1 increase in marketing, sales increase by ${roi:.2f}")
                else:
                    optimal_temp = results_df.loc[results_df['Predicted Sales'].idxmax(), 'Temperature']
                    st.info(f"🌡️ Optimal temperature for sales: {optimal_temp:.0f}°C")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** This simulator uses synthetic data generated by TVAE to predict outcomes for different business scenarios.")
