import os
import time

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import autogluon
from autogluon.tabular import TabularDataset, TabularPredictor

st.set_page_config(
    page_title="SUML Sales Regression",
    layout="wide"
)

DATA_PATH = os.path.join("data", "sales.csv")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# Preprocessing + trenowanie
FEATURE_COLS = [
    "Age_Group",
    "Customer_Gender",
    "Country",
    "State",
    "Product_Category",
    "Sub_Category",
    "Product",
]
TARGET_COL = "Order_Quantity"


@st.cache_data
def preprocess_data(df_raw: pd.DataFrame):
    df = df_raw.copy()

    df = df[FEATURE_COLS + [TARGET_COL]].dropna()

    encoders: dict[str, LabelEncoder] = {}
    categorical_cols = [
        "Age_Group",
        "Customer_Gender",
        "Country",
        "State",
        "Product_Category",
        "Sub_Category",
        "Product",
    ]

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


@st.cache_resource
def train_model(df_processed: pd.DataFrame):
    X = df_processed[FEATURE_COLS]
    y = df_processed[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }

    return model, (X_train, X_test, y_train, y_test, y_pred), metrics

@st.cache_resource
def train_autogluon(df_processed: pd.DataFrame):
    X = df_processed[FEATURE_COLS]
    y = df_processed[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    train_data = X_train.copy()
    train_data["Order Quantity"] = y_train

    predictor = TabularPredictor(label="Order Quantity").fit(train_data, time_limit=40)
    predictor.save("models/")

    y_pred = predictor.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }

    model = predictor.model_best
    return model, (X_train, X_test, y_train, y_test, y_pred), metrics


# Sidebar
st.sidebar.title("Sales App")
page = st.sidebar.selectbox(
    "Wybierz sekcję:",
    [
        "Strona główna",
        "Eksploracja danych",
        "Przygotowanie danych",
        "Model i metryki",
        "Prognoza zamówień",
    ],
)

# Próba wczytania danych

try:
    sales_raw = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Nie znaleziono pliku `{DATA_PATH}`. Upewnij się, że istnieje.")
    st.stop()

# Preprocessing + model
sales_processed, encoders = preprocess_data(sales_raw)
model, splits, metrics = train_model(sales_processed)
model_auto, splits_auto, metrics_auto = train_autogluon(sales_processed)
X_train, X_test, y_train, y_test, y_pred = splits
X_train_auto, X_test_auto, y_train_auto, y_test_auto, y_pred_auto = splits_auto

# strina głowna

if page == "Strona główna":
    st.title("Sales Regression")
    st.write(
        """
        Aplikacja wykorzystuje regresję liniową do przewidywania `Order_Quantity`
        na podstawie danych sprzedażowych.
        """
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Co mamy?")
        st.markdown(
            """
            - Eksplorację danych (proporcje klas, outliery, korelacje)  
            - Przygotowanie danych – encoding, skalowanie  
            - Model – metryki jakości  
            - Prognozy – interaktywny formularz przewidywania `Order_Quantity`  
            """
        )

    with col2:
        with st.expander("Podejrzyj próbkę danych"):
            st.dataframe(sales_raw.head())

    with st.container():
        st.subheader("Szybkie statystyki")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Liczba rekordów", f"{len(sales_raw):,}")
        with col_b:
            st.metric("Liczba cech (po obróbce)", len(FEATURE_COLS))
        with col_c:
            st.metric("MAE", f"{metrics['MAE']:.2f}")
        with col_d:
            st.metric("R²", f"{metrics['R2']:.3f}")

# Eksplorację danych

elif page == "Eksploracja danych":
    st.title("Eksploracja danych")

    tab1, tab2, tab3 = st.tabs(
        ["Proporcje klas", "Outliery", "Korelacje"]
    )

    # Proporcje klas
    with tab1:
        st.subheader("Proporcje wybranych kategorii")
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x="Age_Group", data=sales_raw, ax=ax)
            ax.set_title("Age_Group")
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x="Customer_Gender", data=sales_raw, ax=ax)
            ax.set_title("Customer_Gender")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x="Product_Category", data=sales_raw, ax=ax)
            ax.set_title("Product_Category")
            st.pyplot(fig)

            if "Year" in sales_raw.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(x="Year", data=sales_raw, ax=ax)
                ax.set_title("Year")
                st.pyplot(fig)

    # Outliery
    with tab2:
        st.subheader("Outliery")

        col1, col2 = st.columns(2)

        with col1:
            if "Order_Quantity" in sales_raw.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(x="Order_Quantity", data=sales_raw, ax=ax)
                ax.set_title("Order_Quantity boxplot")
                st.pyplot(fig)
            else:
                st.info("Brak kolumny Order_Quantity.")

        with col2:
            if "Customer_Age" in sales_raw.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(x="Customer_Age", data=sales_raw, ax=ax)
                ax.set_title("Customer_Age boxplot")
                st.pyplot(fig)
            else:
                st.info("Kolumna Customer_Age została usunięta.")

    # Korelacje
    with tab3:
        st.subheader("Macierz korelacji (cechy numeryczne)")

        numeric_sales = sales_raw.select_dtypes(include="number")
        corr = numeric_sales.corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation matrix")
        st.pyplot(fig)

# Przygotowanie danych
elif page == "Przygotowanie danych":
    st.title("Przygotowanie danych")

    st.write("Poniżej pokazane są etapy przetwarzania danych.")

    with st.expander("Dane surowe"):
        st.dataframe(sales_raw.head())

    with st.expander("Dane po drop kolumn"):
        st.write("Usunięto m.in. `Unit_Cost`, `Unit_Price`, `Profit`, `Cost`, `Revenue`, `Customer_Age`, `Date`.")
        cols_after_drop = FEATURE_COLS + [TARGET_COL]
        st.dataframe(sales_processed[cols_after_drop].head())

    with st.expander("Label encoding + skalowanie"):
        st.markdown(
            """
            - **LabelEncoder** zastosowano dla:
              `Month`, `Age_Group`, `Customer_Gender`, `Country`, `State`,
              `Product_Category`, `Sub_Category`, `Product`  
            - **MinMaxScaler** zastosowano dla: `Year`
            """
        )
        st.dataframe(sales_processed.head())

    st.subheader("Rozkład cechy docelowej `Order_Quantity`")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(sales_processed[TARGET_COL], kde=True, ax=ax)
    st.pyplot(fig)


# Model i metryki
elif page == "Model i metryki":
    st.title("Model – Linear Regression (nasz)")

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Metryki")
            st.write(f"**MAE**: {metrics['MAE']:.2f}")
            st.write(f"**MSE**: {metrics['MSE']:.2f}")
            st.write(f"**RMSE**: {metrics['RMSE']:.2f}")
            st.write(f"**R²**: {metrics['R2']:.3f}")

        with col2:
            st.subheader("Parametry modelu")
            st.write("Model: `sklearn.linear_model.LinearRegression`")
            st.write(f"Liczba cech: {len(FEATURE_COLS)}")
            st.write(f"Liczba próbek treningowych: {len(X_train)}")

    st.subheader("Porównanie: y_test vs y_pred")
    comparison_df = pd.DataFrame(
        {
            "y_test": y_test.values,
            "y_pred": y_pred,
        }
    )

    chart = alt.Chart(comparison_df).mark_circle(size=60).encode(
        x="y_test",
        y="y_pred",
        tooltip=["y_test", "y_pred"],
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

#Autogluon
    st.title("Autogluon - AutoML")

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Metryki")
            st.write(f"**MAE**: {metrics_auto['MAE']:.2f}")
            st.write(f"**MSE**: {metrics_auto['MSE']:.2f}")
            st.write(f"**RMSE**: {metrics_auto['RMSE']:.2f}")
            st.write(f"**R²**: {metrics_auto['R2']:.3f}")

        with col2:
            st.subheader("Parametry modelu")
            st.write(f"Model: `{model_auto}`")
            st.write(f"Liczba cech: {len(FEATURE_COLS)}")
            st.write(f"Liczba próbek treningowych: {len(X_train_auto)}")

    st.subheader("Porównanie: y_test vs y_pred")
    comparison_df = pd.DataFrame(
        {
            "y_test": y_test_auto.values,
            "y_pred": y_pred_auto,
        }
    )

    chart = alt.Chart(comparison_df).mark_circle(size=60).encode(
        x="y_test",
        y="y_pred",
        tooltip=["y_test", "y_pred"],
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

# Prognoza zamówień
elif page == "Prognoza zamówień":
    st.title("Prognoza `Order_Quantity`")

    st.write("Wybierz wartości cech, a model zwróci przewidywaną liczbę zamówionych sztuk.")

    age_group = st.selectbox(
        "Age_Group", sorted(sales_raw["Age_Group"].unique().tolist())
    )
    gender = st.selectbox(
        "Customer_Gender", sorted(sales_raw["Customer_Gender"].unique().tolist())
    )
    country = st.selectbox(
        "Country", sorted(sales_raw["Country"].unique().tolist())
    )
    state = st.selectbox("State", sorted(sales_raw["State"].unique().tolist()))
    product_category = st.selectbox(
        "Product_Category", sorted(sales_raw["Product_Category"].unique().tolist())
    )
    sub_category = st.selectbox(
        "Sub_Category", sorted(sales_raw["Sub_Category"].unique().tolist())
    )
    product = st.selectbox(
        "Product", sorted(sales_raw["Product"].unique().tolist())
    )

if st.button("Przewiduj Order_Quantity"):
    with st.spinner("Model liczy prognozę..."):
        time.sleep(0.8)

        sample_dict = {
            "Age_Group": age_group,
            "Customer_Gender": gender,
            "Country": country,
            "State": state,
            "Product_Category": product_category,
            "Sub_Category": sub_category,
            "Product": product,
        }
        sample_df = pd.DataFrame([sample_dict])

        # Encoding
        for col, le in encoders.items():
            sample_df[col] = le.transform(sample_df[col])

        # Kolumny w odpowiedniej kolejności
        sample_df = sample_df[FEATURE_COLS]

        # Predykcja
        pred = model.predict(sample_df)[0]

        st.success(f"Przewidywana wartość **Order_Quantity**: `{pred:.2f}`")

        with st.expander("Zobacz dane wejściowe użyte do prognozy"):
            st.write(sample_dict)
