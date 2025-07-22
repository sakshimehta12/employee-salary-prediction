import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------------------
# 🎛 Page Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Employee Salary / Income Predictor",
    page_icon="💼",
    layout="centered"
)

# ------------------------------------------------------------
# 💅 Custom CSS styling with color names
# ------------------------------------------------------------
st.markdown("""
    <style>
    .main {
        background-color: white !important;
        color: black !important;
    }

    h1, h2, h3 {
        color: white !important;
    }

    .stButton > button {
        background-color: lightblue !important;
        color: black !important;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.4rem 1rem;
    }

    .about-box {
        background: white;
        padding: 1.25rem;
        border-radius: 10px;
        border: 1px solid lightgray;
        margin-top: 2rem;
        text-align: center;
        color: black !important;
    }

    .about-box p {
        color: black !important;
        font-size: 16px;
        line-height: 1.6;
    }

    .about-box h3 {
        color: darkblue !important;
        margin-bottom: 1rem;
    }

    .about-box a {
        color: #0077cc;
        font-weight: bold;
        text-decoration: none;
    }

    .about-box a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)



# ------------------------------------------------------------
# 📥 Data Loader
# ------------------------------------------------------------
@st.cache_data
def load_default_data():
    df = pd.read_csv("adult 3.csv")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ------------------------------------------------------------
# 📦 Sidebar: Data Source
# ------------------------------------------------------------
st.sidebar.header("Data Source")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
df = clean_data(pd.read_csv(uploaded)) if uploaded else clean_data(load_default_data())

st.title("💼 Employee Income Prediction App")
st.write("Predict whether an employee's annual income is **>50K** or **<=50K** using demographic and work-related features.")


# ------------------------------------------------------------
# 👀 Preview Dataset
# ------------------------------------------------------------
with st.expander("Preview Cleaned Dataset"):
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    st.dataframe(df.head(50))


# ------------------------------------------------------------
# 📊 Distribution Plot
# ------------------------------------------------------------
st.subheader("Income Class Distribution")
income_counts = df['income'].value_counts()
fig, ax = plt.subplots()
ax.bar(income_counts.index, income_counts.values, color=['mediumseagreen', 'salmon'])
ax.set_xlabel("Income Category")
ax.set_ylabel("Count")
ax.set_title("Income Class Distribution")
st.pyplot(fig)


# ------------------------------------------------------------
# 🧠 Feature Setup
# ------------------------------------------------------------
FEATURES = ['age', 'education', 'gender', 'occupation', 'hours-per-week']
df.columns = df.columns.str.strip()
missing = [c for c in FEATURES + ['income'] if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}. Please upload a compatible dataset.")
    st.stop()

X_full = pd.get_dummies(df[FEATURES], drop_first=True)
y_full = df['income'].apply(lambda x: 1 if str(x).strip() == '>50K' else 0)

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)


# ------------------------------------------------------------
# 🤖 Train Model
# ------------------------------------------------------------
@st.cache_resource
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)


# ------------------------------------------------------------
# 📈 Model Performance
# ------------------------------------------------------------
st.subheader("Model Performance")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.metric("Accuracy", f"{acc:.2%}")

with st.expander("Detailed Classification Report"):
    st.text(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))

with st.expander("Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(cm, index=['Actual <=50K', 'Actual >50K'], columns=['Pred <=50K', 'Pred >50K']))


# ------------------------------------------------------------
# 🧮 User Prediction
# ------------------------------------------------------------
st.header("🔮 Try Your Own Prediction")
education_opt = sorted(df['education'].unique())
gender_opt = sorted(df['gender'].unique())
occupation_opt = sorted(df['occupation'].unique())

age_in = st.slider("Age", int(df['age'].min()), int(df['age'].max()), 30)
education_in = st.selectbox("Education Level", education_opt)
gender_in = st.selectbox("Gender", gender_opt)
occupation_in = st.selectbox("Occupation", occupation_opt)
hpw_in = st.slider("Hours per Week", int(df['hours-per-week'].min()), int(df['hours-per-week'].max()), 40)

user_df = pd.DataFrame([[age_in, education_in, gender_in, occupation_in, hpw_in]], columns=FEATURES)
user_enc = pd.get_dummies(user_df).reindex(columns=X_full.columns, fill_value=0)

if st.button("Predict Income"):
    pred_class = model.predict(user_enc)[0]
    pred_proba = model.predict_proba(user_enc)[0][1]
    if pred_class == 1:
        st.success(f"Predicted: **>50K** income. Confidence: {pred_proba:.1%}")
    else:
        st.info(f"Predicted: **<=50K** income. Confidence (for >50K): {pred_proba:.1%}")

    st.subheader("Where You Fall vs Dataset")
    fig2, ax2 = plt.subplots()
    ax2.bar(income_counts.index, income_counts.values, alpha=0.6, color='lightslategray')
    ax2.bar(['>50K' if pred_class == 1 else '<=50K'], [max(income_counts.values)*0.1], color='crimson', alpha=0.9)
    ax2.set_title("Dataset Income Distribution (Prediction Highlighted)")
    st.pyplot(fig2)


# ------------------------------------------------------------
# 👩‍💻 About the Creator
# ------------------------------------------------------------
st.markdown("""
<div class="about-box">
    <h3>About the Creator</h3>
    <p>Hi! I'm <strong>Sakshi Mehta</strong> — a BCA student, data enthusiast and aspiring Python programmer.</p>
    <p>This app predicts whether an individual's annual income is greater than ₹50K based on demographics & work features.</p>
</div>
""", unsafe_allow_html=True)