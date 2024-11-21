import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración inicial de la app
st.set_page_config(page_title="Exploración de Datos", layout="wide")

st.title("Exploración de Datos Interactiva")

# Cargar Dataset
uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Crear las pestañas principales
    tabs = st.tabs(["Inicio", "Vista de Datos", "Exploración Estadística", "Visualizaciones", "Filtros Dinámicos"])

    # Pestaña Inicio
    with tabs[0]:
        st.header("Bienvenida")
        st.write("Esta aplicación permite explorar datasets de forma interactiva. Puedes cargar un archivo CSV y navegar por las diferentes secciones para analizarlo.")
        st.info("Selecciona una pestaña para empezar.")

    # Pestaña Vista de Datos
    with tabs[1]:
        st.header("Vista de Datos")
        st.write("Aquí puedes explorar el contenido del dataset.")
        st.write("Dimensiones: ", data.shape)
        st.write("Vista previa del dataset:")
        st.dataframe(data.head(10))
        st.write("Información de los datos:")
        st.write(data.info())

    # Pestaña Exploración Estadística
    with tabs[2]:
        st.header("Exploración Estadística")
        st.write("Resumen estadístico de los datos:")
        st.write(data.describe())

        st.write("Distribución de valores nulos por columna:")
        st.bar_chart(data.isnull().sum())

    # Pestaña Visualizaciones
    with tabs[3]:
        st.header("Visualizaciones")
        st.write("Elige las columnas para graficar:")
        column_x = st.selectbox("Selecciona la columna X:", data.columns)
        column_y = st.selectbox("Selecciona la columna Y:", data.columns)

        fig, ax = plt.subplots()
        sns.scatterplot(x=data[column_x], y=data[column_y], ax=ax)
        st.pyplot(fig)

    # Pestaña Filtros Dinámicos
    with tabs[4]:
        st.header("Filtros Dinámicos")
        st.write("Filtra los datos según las columnas seleccionadas.")

        filter_column = st.selectbox("Selecciona una columna para filtrar:", data.columns)
        unique_values = data[filter_column].unique()
        selected_value = st.selectbox("Selecciona un valor:", unique_values)

        filtered_data = data[data[filter_column] == selected_value]
        st.write("Datos filtrados:", filtered_data)

else:
    st.warning("Por favor, sube un archivo CSV para comenzar.")
