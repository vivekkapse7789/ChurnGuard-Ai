import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="ChurnGuard AI Dashboard",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_assets():
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('model.joblib')
    explainer = joblib.load('explainer.joblib')
    feature_names = joblib.load('feature_names.joblib')
    return preprocessor, model, explainer, feature_names

preprocessor, model, explainer, feature_names = load_assets()

st.title("🛡️ ChurnGuard AI")
st.caption("Customer Retention & Predictive Risk Management Platform")

tabs = st.tabs(["Single Assessment", "Batch Processing & Analytics", "Retention Playbooks"])

with tabs[0]:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Customer Profile Input")
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 150.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges))
        contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
        internet = st.selectbox("Internet Service", ['Fiber optic', 'DSL', 'No'])
        payment = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])

        run_assessment = st.button("Assess Churn Risk", type="primary")

    with col2:
        if run_assessment:
            input_df = pd.DataFrame([{
                'Tenure': tenure,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges,
                'Contract': contract,
                'InternetService': internet,
                'PaymentMethod': payment
            }])

            processed = preprocessor.transform(input_df)
            processed_df = pd.DataFrame(processed, columns=feature_names)

            prob = model.predict_proba(processed_df)[0][1]
            shap_vals = explainer.shap_values(processed_df)[0]

            st.subheader("Risk Score")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#FF4B4B" if prob > 0.5 else "#00CC96"},
                    'steps': [
                        {'range': [0, 35], 'color': "lightgreen"},
                        {'range': [35, 65], 'color': "yellow"},
                        {'range': [65, 100], 'color': "salmon"}
                    ]
                }
            ))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Key Risk Drivers (SHAP)")
            impact_df = pd.DataFrame({'Feature': feature_names, 'Impact': shap_vals})
            impact_df = impact_df.reindex(impact_df.Impact.abs().sort_values(ascending=False).index).head(5)

            fig_bar = px.bar(
                impact_df,
                x='Impact',
                y='Feature',
                orientation='h',
                color='Impact',
                color_continuous_scale=['#00CC96', '#FF4B4B']
            )
            st.plotly_chart(fig_bar, use_container_width=True)

with tabs[1]:
    st.subheader("Batch File Processing")
    uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        required_cols = {'Tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 'PaymentMethod'}
        if required_cols.issubset(batch_df.columns):
            processed_batch = preprocessor.transform(batch_df[list(required_cols)])
            batch_df['Churn_Probability'] = model.predict_proba(
                pd.DataFrame(processed_batch, columns=feature_names)
            )[:, 1]
            
            batch_df['Risk_Level'] = pd.cut(
                batch_df['Churn_Probability'],
                bins=[-1, 0.35, 0.65, 1.0],
                labels=['Low', 'Medium', 'High']
            )

            c1, c2 = st.columns(2)
            with c1:
                st.write("### Predicted Risk Distribution")
                st.plotly_chart(px.pie(batch_df, names='Risk_Level', color='Risk_Level', 
                                       color_discrete_map={'Low':'green', 'Medium':'orange', 'High':'red'}),
                               use_container_width=True)
            with c2:
                st.write("### High Risk Segment Preview")
                st.dataframe(batch_df[batch_df['Risk_Level'] == 'High'].head(10))
        else:
            st.error(f"CSV missing required columns: {required_cols - set(batch_df.columns)}")

with tabs[2]:
    st.subheader("Automated Action Playbooks")
    st.markdown("""
    | Risk Tier | Risk Threshold | Recommended Playbook Action | Trigger Channel |
    | :--- | :--- | :--- | :--- |
    | **High Risk** | > 65% | Offer 20% discount on 1-year contract extension | Direct Agent Call / Automated Email |
    | **Medium Risk** | 35% - 65% | Send feature usage onboarding guide & offer free speed upgrade | In-App Notification |
    | **Low Risk** | < 35% | Enrollment in loyalty/rewards program | Monthly Newsletter |
    """)
