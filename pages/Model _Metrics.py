import streamlit as st

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import json
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


linear_model_history_data = json.load(open("./models/linear_nn/train_history.json"))

st.set_page_config(page_title="Model Metrics", layout="wide")

st.title('Model Metrics Dashboard')


svm_prediction_data = pd.read_csv("test_data/prediction_svm.csv")
linear_prediction_data = pd.read_csv("test_data/prediction_linear.csv")


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


tab1, tab2 = st.tabs(["Category Classification", "Subcategory Classification"])

ml_model = st.radio("Select Model", ["SVM", "Linear Neural Network"])

if ml_model == "SVM":
    df = svm_prediction_data
else:
    df = linear_prediction_data


with tab1:
    st.subheader("Category Classification Performance Metrics")
    
    cat_metrics = calculate_metrics(df['category'], df['predicted_category'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Accuracy", value=f"{cat_metrics['Accuracy']:.2%}")
    with col2:
        st.metric(label="Precision", value=f"{cat_metrics['Precision']:.2%}")
    with col3:
        st.metric(label="Recall", value=f"{cat_metrics['Recall']:.2%}")
    with col4:
        st.metric(label="F1 Score", value=f"{cat_metrics['F1 Score']:.2%}")

with tab2:
    st.subheader("Subcategory Classification Performance Metrics")
    
    subcat_metrics = calculate_metrics(df['subcategory'], df['predicted_subcategory'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Accuracy", value=f"{subcat_metrics['Accuracy']:.2%}")
    with col2:
        st.metric(label="Precision", value=f"{subcat_metrics['Precision']:.2%}")
    with col3:
        st.metric(label="Recall", value=f"{subcat_metrics['Recall']:.2%}")
    with col4:
        st.metric(label="F1 Score", value=f"{subcat_metrics['F1 Score']:.2%}")


st.markdown("---")
st.title("Model Training Metrics")

tab1, tab2 = st.tabs(["Combined Plot", "Individual Metrics"])

with tab1:
    st.header("Training and Validation Metrics")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=list(range(1, len(linear_model_history_data["train_loss"])+1)), y=linear_model_history_data["train_loss"], name="Training Loss"),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=list(range(1, len(linear_model_history_data["val_loss"])+1)), y=linear_model_history_data["val_loss"], name="Validation Loss"),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=list(range(1, len(linear_model_history_data["val_accuracy"])+1)), y=linear_model_history_data["val_accuracy"], name="Validation Accuracy (%)"),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Epoch")
    
    fig.update_yaxes(title_text="Loss", secondary_y=False)
    fig.update_yaxes(title_text="Accuracy (%)", secondary_y=True)
    
    fig.update_layout(
        title="Model Training Metrics Over Epochs",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Individual Metrics")
    
    metric = st.selectbox("Select Metric to Display", ["Training Loss", "Validation Loss", "Validation Accuracy"])
    
    if metric == "Training Loss":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, len(linear_model_history_data["train_loss"])+1)), y=linear_model_history_data["train_loss"], mode="lines+markers"))
        fig.update_layout(title="Training Loss Over Epochs", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig, use_container_width=True)
        
    elif metric == "Validation Loss":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, len(linear_model_history_data["val_loss"])+1)), y=linear_model_history_data["val_loss"], mode="lines+markers"))
        fig.update_layout(title="Validation Loss Over Epochs", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, len(linear_model_history_data["val_accuracy"])+1)), y=linear_model_history_data["val_accuracy"], mode="lines+markers"))
        fig.update_layout(title="Validation Accuracy Over Epochs", xaxis_title="Epoch", yaxis_title="Accuracy (%)")
        st.plotly_chart(fig, use_container_width=True)


st.header("Key Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Final Training Loss", value=f"{linear_model_history_data['train_loss'][-1]:.4f}")
with col2:
    st.metric(label="Final Validation Loss", value=f"{linear_model_history_data['val_loss'][-1]:.4f}")
with col3:
    st.metric(label="Best Validation Accuracy", value=f"{max(linear_model_history_data['val_accuracy']):.2f}%")