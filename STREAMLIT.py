import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    roc_auc_score, 
    precision_recall_curve, 
    f1_score
)
import pickle
import numpy as np

import plotly.express as px
import random  # Para generar valores aleatorios

# Configuración de la página
st.set_page_config(
    page_title="Detección de Fraude Bancario",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Título de la aplicación
st.title("Dashboard de Detección de Fraude Bancario")

# Introducción
st.markdown("""
Este dashboard presenta el análisis y la evaluación de un modelo predictivo diseñado para detectar fraudes bancarios. 
El objetivo es facilitar la comprensión de los resultados a personas sin experiencia en análisis de datos.
""")

# Definición de tipos de datos para optimizar la carga
dtype_dict = {
    'income': 'float32',
    'velocity_24h': 'float32',
    'credit_risk_score': 'float32',
    'foreign_request': 'int8',
    'fraud_bool': 'int8',
    'customer_age': 'int8',
    'phone_mobile_valid': 'int8',
    'payment_type': 'category',
    'employment_status': 'category',
    'bank_months_count': 'int16',
    'session_length_in_minutes': 'float32'
}

# Cargar datos optimizados
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath, dtype=dtype_dict)
    return df

df = load_data("Base.csv")

# Cargar modelo
@st.cache_resource
def load_model():
    with open('stacking_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Cargar y_val y y_pred_val
@st.cache_data
def load_pickle_file(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

y_val = load_pickle_file('y_val.pkl')
y_pred_val = load_pickle_file('y_pred_val.pkl')

# Cargar y_test y y_pred_test si es necesario
y_test = load_pickle_file('y_test.pkl')
y_pred_test = load_pickle_file('y_pred_test.pkl')

# Cargar y_prob_val y y_prob_test si es necesario
y_prob_val = load_pickle_file('y_prob_val.pkl')
y_prob_test = load_pickle_file('y_prob_test.pkl')

# Navegación en la barra lateral
st.sidebar.title("Navegación")
seccion = st.sidebar.radio("Ir a", [
    "Descripción de los Datos",
    "Preprocesamiento de Datos",
    "Evaluación del Modelo",
    "Predicciones Interactivas",
    "Referencias"
])

if seccion == "Predicciones Interactivas":


    # Cargar modelo
    @st.cache_resource
    def load_model():
        with open('stacking_model.pkl', 'rb') as file:
            return pickle.load(file)

    model = load_model()

    # Columnas relevantes y genéricas
    required_columns = [
        'income', 'velocity_24h', 'credit_risk_score', 'foreign_request',
        'customer_age', 'phone_mobile_valid', 'bank_months_count', 'session_length_in_minutes'
    ]
    irrelevant_columns = [col for col in model.feature_names_in_ if col not in required_columns]

    # Inputs interactivos
    st.markdown("Ingrese los datos de una transacción para predecir si es fraude o no.")

    # 1. Income y Velocity (24h) con barras lineales
    income = st.slider("Income (Ingreso en dólares):", min_value=0, max_value=200000, value=50000)
    velocity_24h = st.slider("Velocity 24h (Velocidad en 24h):", min_value=0, max_value=100, value=10)

    # 2. Customer Age con botones por rangos
    st.markdown("### Seleccione la categoría de edad del cliente:")
    age_group = st.radio("Rango de edad:", ["Joven (18-25)", "Adulto (26-40)", "Maduro (41-60)", "Senior (61-90)"])
    if age_group == "Joven (18-25)":
        customer_age = random.randint(18, 25)
    elif age_group == "Adulto (26-40)":
        customer_age = random.randint(26, 40)
    elif age_group == "Maduro (41-60)":
        customer_age = random.randint(41, 60)
    else:  # Senior (61-90)
        customer_age = random.randint(61, 90)

    st.write(f"Edad seleccionada para predicción: {customer_age} años")


    # Foreign Request y Phone Mobile Valid con botones de Sí/No
    st.markdown("### Configuración de valores binarios:")
    foreign_request = st.radio(
        "¿Es una solicitud extranjera?",
        options=["No", "Sí"],
        index=0
    )
    foreign_request = 1 if foreign_request == "Sí" else 0

    phone_mobile_valid = st.radio(
        "¿El teléfono móvil es válido?",
        options=["No", "Sí"],
        index=0
    )
    phone_mobile_valid = 1 if phone_mobile_valid == "Sí" else 0



    # 3. Otros inputs interactivos
    credit_risk_score = st.slider("Credit Risk Score (Puntaje de Riesgo Crediticio):", min_value=0, max_value=100, value=50)
    bank_months_count = st.slider("Bank Months Count (Meses en el Banco):", min_value=0, max_value=240, value=12)
    session_length_in_minutes = st.slider("Session Length (Duración de la Sesión en Minutos):", min_value=0, max_value=300, value=15)

    # Construir DataFrame de entrada
    input_data = {
        'income': income,
        'velocity_24h': velocity_24h,
        'credit_risk_score': credit_risk_score,
        'foreign_request': foreign_request,
        'customer_age': customer_age,
        'phone_mobile_valid': phone_mobile_valid,
        'bank_months_count': bank_months_count,
        'session_length_in_minutes': session_length_in_minutes,
    }

    # Agregar valores genéricos para las columnas irrelevantes
    for column in irrelevant_columns:
        input_data[column] = 1  # Asignar valor genérico

    input_df = pd.DataFrame([input_data])

    # Predicción
    if st.button("Predecir"):
        prob = model.predict_proba(input_df)[0][1]
        pred = int(prob >= 0.5)
        st.write(f"### Probabilidad de Fraude: {prob:.2f}")
        st.write(f"### Predicción: {'Fraude' if pred else 'No Fraude'}")


elif seccion == "Descripción de los Datos":
    st.header("Descripción de los Datos")
    st.markdown("""
    A continuación, se muestra una visión general de los datos utilizados para entrenar el modelo de detección de fraude.
    """)

    # Mostrar muestra de datos
    if st.checkbox("Mostrar muestra de datos"):
        st.write(df.sample(100))  # Muestra 100 filas aleatorias

    # Estadísticas descriptivas
    st.write("### Estadísticas Descriptivas")
    st.write(df.describe())

    # Gráfica de la distribución de fraude
    st.write("### Distribución de Fraudes")
    fig, ax = plt.subplots()
    sns.countplot(x='fraud_bool', data=df, ax=ax)
    ax.set_title('Distribución de Fraude (0: No, 1: Sí)')
    st.pyplot(fig)

elif seccion == "Preprocesamiento de Datos":
    st.header("Preprocesamiento de Datos")
    st.markdown("""
    El preprocesamiento de datos incluyó:
    - Codificación de variables categóricas.
    - Escalado de variables numéricas.
    - Manejo de desequilibrio de clases mediante SMOTE y NearMiss.
    """)

    st.subheader("Codificación de Variables Categóricas")
    st.write("Se utilizó One-Hot Encoding para convertir las variables categóricas en numéricas.")
    categorical_cols = ['payment_type', 'employment_status']
    st.write(df[categorical_cols].head())

    st.subheader("Escalado de Variables Numéricas")
    st.write("Se aplicó StandardScaler para normalizar las características numéricas.")

    st.subheader("Manejo de Desequilibrio de Clases")
    st.write("""
    Se aplicaron técnicas de sobremuestreo (SMOTE) y submuestreo (NearMiss) para equilibrar las clases.
    Esto mejora la capacidad del modelo para detectar fraudes sin sesgarse hacia la clase mayoritaria.
    """)

elif seccion == "Evaluación del Modelo":
    st.header("Evaluación del Modelo")
    st.markdown("""
    Se evaluó el modelo utilizando conjuntos de validación y prueba, analizando métricas clave como precisión, recall, F1-score y AUC.
    """)

    st.subheader("Reporte de Clasificación - Conjunto de Validación")
    report_val = classification_report(y_val, y_pred_val, output_dict=True)
    df_report_val = pd.DataFrame(report_val).transpose()
    st.dataframe(df_report_val)

    # Confusión Matrix
    st.subheader("Matriz de Confusión - Conjunto de Validación")
    conf_matrix_val = confusion_matrix(y_val, y_pred_val)
    conf_matrix_val_percent = conf_matrix_val.astype('float') / conf_matrix_val.sum(axis=1)[:, np.newaxis] * 100  # Normalización por fila
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix_val_percent, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=['Predicción No Fraude', 'Predicción Fraude'],
                yticklabels=['No Fraude', 'Fraude'])
    ax.set_xlabel('Predicciones')
    ax.set_ylabel('Valores Reales')
    ax.set_title('Matriz de Confusión - Conjunto de Validación (%)')
    st.pyplot(fig)

    # Curva ROC
    st.subheader("Curva ROC - Conjunto de Validación")
    fpr, tpr, _ = roc_curve(y_val, y_prob_val)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('Curva ROC - Conjunto de Validación')
    ax.legend(loc="lower right")
    st.pyplot(fig)

elif seccion == "Referencias":
    st.header("Trabajo para el curso de DATA ANALYSIS")
    st.markdown("""
    
    Group menbers 
    
    - io Aarón Torres Mariño 20194608
    - Carlos Francisco Jaime Alza 20192231
    - HannahIsabelle van den Brink- 2024618
    - Carlos Eduardo Zegarra Barrenechea 20216177
                
    DATASET 
                
    - Bank Account Fraud Dataset Suite (NeurIPS 2022)
     https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022?select=
                
    """)


