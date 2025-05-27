import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import streamlit as st

# Configuración inicial de Streamlit
st.set_page_config(layout="wide")
st.title("Simulación del Consumo de Sustancias y Tolerancia")
st.markdown("""
Este modelo simula el efecto de diferentes patrones de consumo sobre el cuerpo humano, incluyendo la tolerancia desarrollada con el tiempo.
Selecciona una sustancia para cargar sus parámetros automáticos y modifica la forma de consumo desde el panel lateral.
""")

# Menú de selección de sustancia
st.sidebar.markdown("## Sustancia")
sustancia = st.sidebar.selectbox("Selecciona una sustancia", ["Simulacion General", "Alcohol", "Nicotina", "Marihuana"])

# Valores predeterminados
parametros = {
    "Simulacion General": {"ke": 0.5, "alpha": 0.3, "beta": 0.1},
    "Alcohol": {"ke": 0.3, "alpha": 0.2, "beta": 0.05},
    "Nicotina": {"ke": 0.7, "alpha": 0.4, "beta": 0.15},
    "Marihuana": {"ke": 0.4, "alpha": 0.25, "beta": 0.1},
}

# Asignar valores según la sustancia seleccionada
ke_default = parametros[sustancia]["ke"]
alpha_default = parametros[sustancia]["alpha"]
beta_default = parametros[sustancia]["beta"]

# Sliders con unidades
st.sidebar.markdown("## Parámetros fisiológicos")
ke = st.sidebar.slider("Tasa de eliminación ke [1/hora]", 0.1, 1.0, ke_default, step=0.05)
alpha = st.sidebar.slider("Aumento de tolerancia α [1/mg]", 0.0, 1.0, alpha_default, step=0.05)
beta = st.sidebar.slider("Reducción de tolerancia β [1/hora]", 0.0, 1.0, beta_default, step=0.05)

# Tiempo
t_max = st.sidebar.slider("Tiempo máximo de simulación [horas]", 10, 100, 50)
t = np.linspace(0, t_max, 500)

# Funciones de consumo
def u_singular(t): return 0
def u_constante(t, R0=1.0): return R0
def u_lineal(t, a=0.2): return a * t
def u_periodica(t, D=5, T=5): return D * sum(np.isclose(t, n * T, atol=0.1) for n in range(int(t // T) + 1))

# EDO
def modelo(y, t, ke, alpha, beta, u_func):
    C, T = y
    u_t = u_func(t)
    dCdt = -ke * C + u_t
    dTdt = alpha * u_t - beta * T
    return [dCdt, dTdt]

# Condiciones iniciales
y0_singular = [10, 2]
y0_general = [0, 0]

# Selección de tipo de consumo
st.sidebar.markdown("## Tipo de consumo")
tipo = st.sidebar.radio("", ["Dosis única", "Consumo continuo", "Consumo lineal", "Consumo periódico"])

# Configuración adicional según tipo
if tipo == "Consumo periódico":
    D = st.sidebar.slider("Dosis por toma [mg]", 1, 10, 5)
    T_per = st.sidebar.slider("Intervalo entre dosis [horas]", 1, 20, 5)
    u_func = lambda t: u_periodica(t, D, T_per)
    y0 = y0_general
elif tipo == "Consumo continuo":
    R0 = st.sidebar.slider("Tasa constante de consumo [mg/hora]", 0.1, 5.0, 1.0)
    u_func = lambda t: u_constante(t, R0)
    y0 = y0_general
elif tipo == "Consumo lineal":
    a = st.sidebar.slider("Incremento de consumo [mg/hora²]", 0.01, 1.0, 0.2)
    u_func = lambda t: u_lineal(t, a)
    y0 = y0_general
else:
    u_func = u_singular
    y0 = y0_singular

# Solución del modelo
sol = odeint(modelo, y0, t, args=(ke, alpha, beta, u_func))

# Gráfico principal
st.subheader(f"Simulación para: {sustancia}")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, sol[:, 0], label='C(t): Sustancia [mg]')
ax.plot(t, sol[:, 1], label='T(t): Tolerancia [adimensional]')
ax.set_xlabel("Tiempo [horas]")
ax.set_ylabel("Cantidad")
ax.set_title(f"Simulación: {tipo}")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Resultados finales
st.markdown(f"**C(t_final):** {sol[-1,0]:.2f} mg | **T(t_final):** {sol[-1,1]:.2f} (tolerancia)")

# Sección de comparación
st.markdown("---")
st.subheader("Comparativa entre dos sustancias")

# Selección de sustancias para comparar
col1, col2 = st.columns(2)
with col1:
    sust1 = st.selectbox("Sustancia A", ["Alcohol", "Nicotina", "Marihuana"], index=0)
with col2:
    sust2 = st.selectbox("Sustancia B", ["Alcohol", "Nicotina", "Marihuana"], index=1)

# Parámetros de ambas sustancias
p1 = parametros[sust1]
p2 = parametros[sust2]

# Función de consumo fijo para comparación
u_comparacion = lambda t: u_constante(t, 1.0)
y0_comp = [0, 0]

# Solución para ambas
sol1 = odeint(modelo, y0_comp, t, args=(p1["ke"], p1["alpha"], p1["beta"], u_comparacion))
sol2 = odeint(modelo, y0_comp, t, args=(p2["ke"], p2["alpha"], p2["beta"], u_comparacion))

# Gráfica comparativa
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(t, sol1[:, 0], label=f'C(t) {sust1}', linestyle='--')
ax2.plot(t, sol2[:, 0], label=f'C(t) {sust2}', linestyle='-')
ax2.plot(t, sol1[:, 1], label=f'T(t) {sust1}', linestyle='--')
ax2.plot(t, sol2[:, 1], label=f'T(t) {sust2}', linestyle='-')
ax2.set_xlabel("Tiempo [horas]")
ax2.set_ylabel("Cantidad")
ax2.set_title("Comparativa de C(t) y T(t) con consumo constante de 1 mg/h")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

