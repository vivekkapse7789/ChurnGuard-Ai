import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
import shap

def generate_synthetic_telco_data(num_samples=1000):
    np.random.seed(42)
    tenure = np.random.randint(1, 72, size=num_samples)
    monthly_charges = np.random.uniform(18.25, 118.75, size=num_samples)
    total_charges = tenure * monthly_charges + np.random.normal(0, 10, size=num_samples)
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=num_samples, p=[0.55, 0.25, 0.20])
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], size=num_samples, p=[0.35, 0.45, 0.20])
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
        size=num_samples
    )
    
    churn_prob = (
        0.3 * (contract == 'Month-to-month') +
        0.3 * (internet_service == 'Fiber optic') -
        0.005 * tenure +
        0.002 * monthly_charges
    )
    churn_prob = 1 / (1 + np.exp(-churn_prob))
    churn = np.random.binomial(1, churn_prob)

    return pd.DataFrame({
        'Tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': np.maximum(0, total_charges),
        'Contract': contract,
        'InternetService': internet_service,
        'PaymentMethod': payment_method,
        'Churn': churn
    })

def train_pipeline():
    df = generate_synthetic_telco_data()
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    num_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = ['Contract', 'InternetService', 'PaymentMethod']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    feature_names = num_cols + list(cat_feature_names)

    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    model.fit(X_train_df, y_train)

    preds = model.predict(X_test_df)
    probs = model.predict_proba(X_test_df)[:, 1]
    print("--- Model Performance ---")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")

    explainer = shap.TreeExplainer(model)

    joblib.dump(preprocessor, 'preprocessor.joblib')
    joblib.dump(model, 'model.joblib')
    joblib.dump(explainer, 'explainer.joblib')
    joblib.dump(feature_names, 'feature_names.joblib')
    print("\nArtifacts successfully saved: preprocessor.joblib, model.joblib, explainer.joblib, feature_names.joblib")

if __name__ == '__main__':
    train_pipeline()
