import pandas as pd
import numpy as np
import streamlit as st
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit başlığı
st.title("Diabetes Prediction with Naive Bayes (Selected Features)")

# Veri yükleme
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes_pima_indians.csv")
    cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)
    df[cols_to_replace] = df[cols_to_replace].fillna(df[cols_to_replace].median())
    return df

df = load_data()
st.write("Dataset Preview:", df.head())

# Sadece belirli özellikleri seç
selected_features = ['Pregnancies', 'Insulin', 'SkinThickness']
X = df[selected_features]
y = df["Outcome"]

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Naive Bayes modeli
model = GaussianNB()
model.fit(X_train_scaled, y_train)

# Kullanıcıdan giriş al
st.sidebar.header("Input Features")
def user_input_features():
    input_data = {}
    for feature in selected_features:
        input_data[feature] = st.sidebar.slider(
            feature, float(X[feature].min()), float(X[feature].max()), float(X[feature].mean())
        )
    return pd.DataFrame(input_data, index=[0])

input_df = user_input_features()

# Tahmin yap butonu
if st.button("Show Prediction Results"):
    # Tahmin yap
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    # Tahmin sonuçlarını göster
    st.subheader("Prediction")
    st.write("Positive (Diabetes)" if prediction[0] == 1 else "Negative (No Diabetes)")

    st.subheader("Prediction Probability")
    st.write(f"Probability of No Diabetes: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Diabetes: {prediction_proba[0][1]:.2f}")

    # Eğitim seti sonuçlarını görselleştir
    st.subheader("Model Evaluation on Test Set")
    y_pred = model.predict(X_test_scaled)
    st.write("Accuracy:", accuracy_score(y_test, y_pred))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig)