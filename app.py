import numpy as np
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ---------- SETTINGS ----------
st.set_page_config(page_title="Boston Housing Price Predictor", layout="wide")
st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    h1, h2, h3, h4 { color: #2C3E50; }
</style>
""", unsafe_allow_html=True)

# ---------- FUNCTIONS ----------
def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DataFrame columns to Arrow-compatible types."""
    return df.convert_dtypes().infer_objects()

@st.cache_data
def load_data():
    df = pd.read_csv('data/boston.csv')
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

# ---------- LOAD DATA & MODEL ----------
with st.spinner("Loading dataset..."):
    df = load_data()

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Train-test split for evaluation
X = df.drop('MEDV', axis=1)
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- HEADER ----------
st.title("Boston Housing Price Prediction Dashboard")
st.caption("Explore the dataset, visualize trends, and predict housing prices using Machine Learning.")

# ---------- TABS ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", 
    "Data Exploration", 
    "Visualisation", 
    "Prediction", 
    "Performance"
])

# ---------- TAB 1: OVERVIEW ----------
with tab1:
    st.header("Project Overview")
    st.markdown("""
    This app predicts **Boston house prices** using a trained **Random Forest** model.

    **Dataset**: Boston Housing Prices  
    **Features**: 13 predictors (CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT)  
    **Target**: MEDV — Median home value in $1000's  

    ### Features of the App:
    - Explore dataset with interactive filtering
    - View multiple visualisations
    - Make predictions with confidence intervals
    - Compare model performance  
    """)

# ---------- TAB 2: DATA EXPLORATION ----------
with tab2:
    st.header("Data Overview")
    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", df.columns.tolist())
    st.write("**Data Types:**", df.dtypes)
    st.subheader("Sample Data")
    st.dataframe(make_arrow_safe(df.head()), use_container_width=True)

    st.subheader("Filter Data")
    col = st.selectbox("Select column to filter", df.columns)
    col_data = pd.to_numeric(df[col], errors='coerce')
    if pd.api.types.is_numeric_dtype(col_data):
        min_val, max_val = float(col_data.min()), float(col_data.max())
        if min_val == max_val:
            st.write(f"Column {col} has constant value: {min_val}")
            filtered_df = df.copy()
        else:
            val = st.slider(f"Filter {col}", min_val, max_val, (min_val, max_val))
            filtered_df = df[(col_data >= val[0]) & (col_data <= val[1])]
    else:
        unique_vals = df[col].dropna().unique().tolist()
        selected_vals = st.multiselect(
            f"Select {col} values", unique_vals, default=unique_vals
        )
        filtered_df = df[df[col].isin(selected_vals)]
    st.dataframe(make_arrow_safe(filtered_df), use_container_width=True)

# ---------- TAB 3: VISUALISATION ----------
with tab3:
    st.header("Visualisations")
    col_choice = st.selectbox("Select column for histogram", df.columns)
    st.subheader(f"Histogram of {col_choice}")
    st.bar_chart(df[col_choice])

    st.subheader("Correlation Heatmap")
    threshold = st.slider("Correlation threshold", 0.0, 1.0, 0.5)
    corr = df.corr(numeric_only=True)
    mask = (abs(corr) >= threshold)
    fig, ax = plt.subplots()
    sns.heatmap(corr.where(mask), ax=ax, cmap="coolwarm", annot=True, fmt=".2f")
    st.pyplot(fig)

    st.subheader("Scatter Plot")
    x_col = st.selectbox("X axis", df.columns)
    y_col = st.selectbox("Y axis", df.columns, index=len(df.columns)-1)
    fig2, ax2 = plt.subplots()
    ax2.scatter(df[x_col], df[y_col])
    ax2.set_xlabel(x_col)
    ax2.set_ylabel(y_col)
    st.pyplot(fig2)

# ---------- TAB 4: PREDICTION ----------
with tab4:
    st.header("Predict House Price")
    inputs = {}
    for feature in X.columns:
        if feature == 'CHAS':
            inputs[feature] = st.selectbox(feature, [0, 1], help="0 = No, 1 = Yes")
        else:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            default_val = float(df[feature].mean())
            inputs[feature] = st.number_input(
                feature, min_value=min_val, max_value=max_val, value=default_val
            )
    if st.button('Predict'):
        with st.spinner("Making prediction..."):
            features = np.array([[inputs[f] for f in X.columns]])
            prediction = model.predict(features)
            residuals = y_train - model.predict(X_train)
            std_dev = np.std(residuals)
            lower_bound = prediction[0] - 1.96 * std_dev
            upper_bound = prediction[0] + 1.96 * std_dev
            st.success(f'Predicted Price: ${prediction[0]:.2f}k')
            st.info(f"95% Confidence Interval: ${lower_bound:.2f}k - ${upper_bound:.2f}k")

# ---------- TAB 5: PERFORMANCE ----------
with tab5:
    st.header("Model Performance")
    with st.spinner("Evaluating model..."):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    st.write(f"**Mean Squared Error:** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    st.subheader("Actual vs Predicted")
    fig3, ax3 = plt.subplots()
    ax3.scatter(y_test, y_pred)
    ax3.set_xlabel("Actual MEDV")
    ax3.set_ylabel("Predicted MEDV")
    st.pyplot(fig3)

    st.subheader("Residuals Histogram")
    residuals = y_test - y_pred
    fig4, ax4 = plt.subplots()
    ax4.hist(residuals, bins=20)
    ax4.set_xlabel("Residual")
    ax4.set_ylabel("Frequency")
    st.pyplot(fig4)

    st.subheader("Model Comparison")
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42)
    }
    comparison = []
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        pred = mdl.predict(X_test)
        comparison.append({
            "Model": name,
            "MSE": mean_squared_error(y_test, pred),
            "R²": r2_score(y_test, pred)
        })
    st.dataframe(make_arrow_safe(pd.DataFrame(comparison)), use_container_width=True)