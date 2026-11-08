import streamlit as st, pandas as pd, joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix,
    ConfusionMatrixDisplay, classification_report)

TARGET = "price_range"
FILES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "support_vector_machine",
}

st.set_page_config(page_title="Phone Price Classifier", layout="wide")
st.title("Phone Price Range - Model Comparison App")
st.write("Upload the test CSV, pick a model, and view how well it performs.")

encoder = joblib.load("model/label_encoder.joblib")
feature_cols = joblib.load("model/feature_columns.joblib")

st.sidebar.header("Controls")
model_name = st.sidebar.selectbox("Choose a model", list(FILES.keys()))
upload = st.sidebar.file_uploader("Upload test CSV", type="csv")

if upload is None:
    st.info("Please upload test_data.csv using the sidebar.")
else:
    df = pd.read_csv(upload)
    missing = [c for c in feature_cols + [TARGET] if c not in df.columns]
    if missing:
        st.error("These columns are missing: " + str(missing))
        st.stop()

    st.subheader("Data preview")
    st.dataframe(df.head())

    model = joblib.load("model/" + FILES[model_name] + ".joblib")
    X = df[feature_cols]
    y = encoder.transform(df[TARGET])
    pred = model.predict(X)
    proba = model.predict_proba(X)
    n = len(encoder.classes_)
    avg = "binary" if n == 2 else "weighted"
    if n == 2:
        auc = roc_auc_score(y, proba[:, 1])
    else:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="weighted")

    results = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, average=avg, zero_division=0),
        "Recall": recall_score(y, pred, average=avg, zero_division=0),
        "F1": f1_score(y, pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }

    st.subheader("Evaluation metrics - " + model_name)
    cols = st.columns(6)
    for c, (k, v) in zip(cols, results.items()):
        c.metric(k, round(v, 3))

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix(y, pred),
                               display_labels=encoder.classes_).plot(ax=ax)
        st.pyplot(fig)
    with right:
        st.subheader("Classification report")
        rep = classification_report(y, pred, output_dict=True,
                                    target_names=[str(c) for c in encoder.classes_])
        st.dataframe(pd.DataFrame(rep).T.round(3))