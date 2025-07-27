import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# === Загрузка и очистка данных ===

def load_data(path="https://github.com/itismaruf/FirstProject/blob/master/titanic.csv"):
    """Загрузка данных из CSV"""
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Базовая очистка данных"""
    df = df.copy()
    
    # Удалим ненужные столбцы
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors='ignore', inplace=True)

    # Заполнение пропусков
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    return df

def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot кодирование категориальных признаков"""
    df_encoded = pd.get_dummies(df, columns=["Sex", "Embarked", "Pclass"], drop_first=True)
    return df_encoded


# === Визуализация ===

def plot_survival_by_sex(df: pd.DataFrame):
    return px.histogram(df, x="Sex", color="Survived", barmode="group",
                        title="Выживаемость по полу", labels={"Survived": "Выжил"})

def plot_survival_by_class(df: pd.DataFrame):
    return px.histogram(df, x="Pclass", color="Survived", barmode="group",
                        title="Выживаемость по классу", labels={"Survived": "Выжил"})

def plot_age_distribution(df: pd.DataFrame):
    return px.histogram(df, x="Age", color="Survived", nbins=30,
                        title="Распределение возраста", labels={"Survived": "Выжил"})


# === Модель ===

def train_model(df: pd.DataFrame):
    """Обучение Random Forest модели"""
    df = df.copy()
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    return model, train_acc, test_acc


# === Предсказание одного пассажира ===

def predict_single_passenger(model, df_template: pd.DataFrame, user_input: dict) -> int:
    """Подготовка пользовательских данных и предсказание"""
    user_df = pd.DataFrame([user_input])
    user_df_encoded = pd.get_dummies(user_df)

    # Добавим недостающие столбцы из шаблона
    for col in df_template.columns:
        if col not in user_df_encoded.columns:
            user_df_encoded[col] = 0

    user_df_encoded = user_df_encoded[df_template.columns]  # Убедимся в порядке колонок

    prediction = model.predict(user_df_encoded)[0]
    proba = model.predict_proba(user_df_encoded)[0]

    return prediction, proba
