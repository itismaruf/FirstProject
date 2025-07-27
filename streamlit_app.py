# app.py

import streamlit as st
import pandas as pd
from functions import (
    load_data, clean_data, encode_data,
    plot_survival_by_sex, plot_survival_by_class, plot_age_distribution,
    train_model, predict_single_passenger
)

st.set_page_config(page_title="🚢 Titanic Classifier", layout="wide")
st.title("🚢 Titanic Classifier - Предсказание выживаемости пассажиров")

# === 📥 Загрузка и предобработка данных ===
st.subheader("📦 Загрузка и обработка данных")
raw_df = load_data()
df = clean_data(raw_df)
df_encoded = encode_data(df)

st.dataframe(df.head(10), use_container_width=True)

# === 📊 Визуализация данных ===
st.subheader("📊 Визуализация данных")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(plot_survival_by_sex(df), use_container_width=True)
    st.plotly_chart(plot_survival_by_class(df), use_container_width=True)
with col2:
    st.plotly_chart(plot_age_distribution(df), use_container_width=True)

# === 🤖 Обучение модели ===
st.subheader("🤖 Обучение модели RandomForest")

model, train_acc, test_acc = train_model(df_encoded)

st.success(f"Train Accuracy: {train_acc:.2f}")
st.success(f"Test Accuracy: {test_acc:.2f}")

# === 🧍 Предсказание по параметрам пользователя ===
st.sidebar.header("🔮 Ввод данных пассажира")

sex = st.sidebar.selectbox("Пол", ["male", "female"])
pclass = st.sidebar.selectbox("Класс", [1, 2, 3])
age = st.sidebar.slider("Возраст", 0, 80, 30)
fare = st.sidebar.slider("Тариф", 0.0, 600.0, 50.0)
sibsp = st.sidebar.slider("Число братьев/сестер или супругов на борту", 0, 8, 0)
parch = st.sidebar.slider("Число родителей/детей на борту", 0, 6, 0)
embarked = st.sidebar.selectbox("Порт посадки", ["C", "Q", "S"])

user_input = {
    "Sex": sex,
    "Pclass": pclass,
    "Age": age,
    "Fare": fare,
    "SibSp": sibsp,
    "Parch": parch,
    "Embarked": embarked
}

# 🔮 Предсказание
st.sidebar.subheader("🧠 Результат предсказания")
prediction, proba = predict_single_passenger(model, df_encoded.drop("Survived", axis=1), user_input)

label = "✅ Выжил" if prediction == 1 else "❌ Не выжил"
st.sidebar.markdown(f"### Предсказание: {label}")

st.sidebar.write("Вероятности:")
st.sidebar.progress(proba[1])  # вероятность выживания
st.sidebar.markdown(f"Выживание: **{proba[1]*100:.1f}%**")
st.sidebar.markdown(f"Не выжил: **{proba[0]*100:.1f}%**")
