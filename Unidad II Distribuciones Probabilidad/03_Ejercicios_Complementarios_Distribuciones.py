# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Introducción
# MAGIC %md
# MAGIC # 📊 Ejercicios Complementarios: Distribuciones de Probabilidad
# MAGIC ## Aplicaciones en Contabilidad, Economía y Administración
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Objetivos de Aprendizaje
# MAGIC
# MAGIC Estos ejercicios complementarios están diseñados para reforzar los conceptos de distribuciones de probabilidad aplicados a contextos económicos y empresariales.
# MAGIC
# MAGIC Cada ejercicio sigue la estructura:
# MAGIC
# MAGIC 1. **Contexto económico real** (finanzas, control de calidad, control presupuestario, proyección de ingresos)
# MAGIC 2. **Conceptos teóricos clave** de la distribución aplicada
# MAGIC 3. **Solución detallada** con cálculos paso a paso
# MAGIC 4. **Visualizaciones profesionales** para interpretación
# MAGIC 5. **Conclusiones y recomendaciones** para aplicación práctica
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📚 Distribuciones Cubiertas
# MAGIC
# MAGIC #### **Ejercicio A: Distribución Binomial**
# MAGIC * **Contexto:** Control de calidad en producción de chips electrónicos
# MAGIC * **Aplicaciones:** 
# MAGIC   - Probabilidades de defectos en lotes
# MAGIC   - Establecimiento de umbrales de rechazo
# MAGIC   - Simulación de procesos de inspección
# MAGIC   - Diseño de políticas de control de calidad
# MAGIC
# MAGIC #### **Ejercicio B: Distribución Normal y Estandarización**
# MAGIC * **Contexto:** Análisis de costos operativos de sucursales minoristas
# MAGIC * **Aplicaciones:**
# MAGIC   - Control presupuestario y benchmarking
# MAGIC   - Identificación de sucursales con costos atípicos (outliers)
# MAGIC   - Comparación entre regiones usando Z-scores
# MAGIC   - Establecimiento de intervalos de control (68%, 95%, 99.7%)
# MAGIC   - Eficiencia operativa y mejora continua
# MAGIC
# MAGIC #### **Ejercicio C: Teorema Central del Límite (TCL)**
# MAGIC * **Contexto:** Proyección de ingresos y estimación de ventas
# MAGIC * **Aplicaciones:**
# MAGIC   - Estimación de ingresos promedio con muestras
# MAGIC   - Diseño de tamaño de muestra para proyecciones confiables
# MAGIC   - Intervalos de confianza para medias
# MAGIC   - Verificación experimental: poblaciones NO normales → medias normales
# MAGIC   - Justificación de métodos estadísticos en auditoría y muestreo
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🛠️ Herramientas Utilizadas
# MAGIC
# MAGIC * **Python**: `numpy`, `scipy.stats`, `matplotlib`
# MAGIC * **Visualizaciones**: Histogramas, gráficos de densidad, CDF, Q-Q plots, comparaciones
# MAGIC * **Métodos**: Cálculos analíticos, simulaciones Monte Carlo, verificación experimental
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💼 Relevancia para Ciencias Económicas
# MAGIC
# MAGIC Estos ejercicios cubren aplicaciones directas en:
# MAGIC
# MAGIC * ✅ **Contabilidad y Auditoría**: Control presupuestario, identificación de anomalías, muestreo
# MAGIC * ✅ **Finanzas**: Proyección de ingresos, análisis de riesgo, intervalos de confianza
# MAGIC * ✅ **Gestión de Operaciones**: Control de calidad, dimensionamiento de capacidad, eficiencia
# MAGIC * ✅ **Economía Aplicada**: Diseño de encuestas, inferencia estadística, política pública
# MAGIC * ✅ **Administración**: Toma de decisiones bajo incertidumbre, benchmarking, mejora continua
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🚀 ¡Comencemos!

# COMMAND ----------

# DBTITLE 1,Importar librerías
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# Configuración para gráficos con fondo blanco y sin recuadro
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['figure.figsize'] = (10, 6)
# Eliminar recuadro superior y derecho
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

print("✓ Bibliotecas importadas correctamente")
print("✓ Configuración de gráficos lista (fondo blanco, sin recuadro superior/derecho)")

# COMMAND ----------

# DBTITLE 1,Ejercicio A: Problema
# MAGIC %md
# MAGIC ## 📝 Ejercicio Complementario A: Control de Calidad en Producción
# MAGIC
# MAGIC ### Contexto de Negocio
# MAGIC
# MAGIC Una fábrica de componentes electrónicos produce chips con una **tasa de defectos del 3%**. Los chips se empaquetan en lotes de **200 unidades** para envío a clientes.
# MAGIC
# MAGIC ### Problema
# MAGIC
# MAGIC Como responsable de control de calidad, necesitas:
# MAGIC
# MAGIC 1. **Calcular la probabilidad** de que un lote contenga:
# MAGIC    - Exactamente 5 chips defectuosos
# MAGIC    - Más de 10 chips defectuosos
# MAGIC    - Entre 3 y 8 chips defectuosos (inclusive)
# MAGIC
# MAGIC 2. **Determinar el número esperado** de chips defectuosos por lote y su variabilidad
# MAGIC
# MAGIC 3. **Establecer un umbral de rechazo**: ¿Qué número de defectos debería hacer que rechacemos un lote completo? (considera el percentil 95)
# MAGIC
# MAGIC 4. **Evaluar política de inspección**: Si inspeccionamos 5 lotes al día, ¿cuál es la probabilidad de encontrar al menos un lote con más de 10 defectos?
# MAGIC
# MAGIC ### Distribución a Usar
# MAGIC
# MAGIC **Binomial**: X ~ Binomial(n=200, p=0.03)
# MAGIC
# MAGIC * n = 200 chips por lote
# MAGIC * p = 0.03 (probabilidad de defecto)
# MAGIC * X = número de chips defectuosos en un lote

# COMMAND ----------

# DBTITLE 1,📚 Conceptos Clave
# MAGIC %md
# MAGIC ### 📚 Conceptos Clave para el Ejercicio A
# MAGIC
# MAGIC **1. Distribución Binomial en Control de Calidad:**
# MAGIC * Cada chip es un ensayo independiente (defectuoso/no defectuoso)
# MAGIC * Probabilidad constante p = 0.03
# MAGIC * n = 200 ensayos (tamaño del lote)
# MAGIC
# MAGIC **2. Parámetros importantes:**
# MAGIC * **E(X) = np**: Número esperado de defectos
# MAGIC * **Var(X) = np(1-p)**: Variabilidad esperada
# MAGIC * **σ(X) = √[np(1-p)]**: Desviación estándar
# MAGIC
# MAGIC **3. Probabilidades acumuladas:**
# MAGIC * P(X ≤ k): CDF (Función de distribución acumulada)
# MAGIC * P(X > k) = 1 - P(X ≤ k): Función de supervivencia
# MAGIC * P(a ≤ X ≤ b) = P(X ≤ b) - P(X ≤ a-1)
# MAGIC
# MAGIC **4. Percentiles para umbrales:**
# MAGIC * Percentil 95: Valor k tal que P(X ≤ k) = 0.95
# MAGIC * Sirve para definir límites de aceptación/rechazo
# MAGIC
# MAGIC **5. Aproximación Normal:**
# MAGIC Cuando n es grande y p no es extremo, Binomial ≈ Normal
# MAGIC * Condición: np > 5 y n(1-p) > 5
# MAGIC * En este caso: np = 6 (justo en el límite, usaremos Binomial exacta)

# COMMAND ----------

# DBTITLE 1,💻 Solución Ejercicio A
# ==========================================
# EJERCICIO A: CONTROL DE CALIDAD
# ==========================================

# Parámetros del problema
n = 200  # tamaño del lote
p = 0.03  # tasa de defectos (3%)

# Crear la distribución Binomial
X = stats.binom(n, p)

print("="*70)
print("EJERCICIO A: CONTROL DE CALIDAD EN PRODUCCIÓN")
print("="*70)
print(f"Distribución: X ~ Binomial(n={n}, p={p})")
print(f"\nEscenario: Lotes de {n} chips con tasa de defectos del {p*100}%")
print("="*70)

# ========================================
# 1. PROBABILIDADES ESPECÍFICAS
# ========================================
print("\n1. PROBABILIDADES DE DEFECTOS EN UN LOTE:")
print("-"*70)

# a) Exactamente 5 defectos
prob_5 = X.pmf(5)
print(f"   a) P(X = 5 defectos) = {prob_5:.6f} ({prob_5*100:.4f}%)")
print(f"      Interpretación: {prob_5*100:.4f}% de los lotes tendrán exactamente 5 chips defectuosos")

# b) Más de 10 defectos
prob_mas_10 = 1 - X.cdf(10)  # P(X > 10) = 1 - P(X <= 10)
print(f"\n   b) P(X > 10 defectos) = {prob_mas_10:.6f} ({prob_mas_10*100:.4f}%)")
print(f"      Interpretación: {prob_mas_10*100:.4f}% de los lotes tendrán más de 10 defectos")
print(f"      Esto es aproximadamente 1 de cada {int(1/prob_mas_10)} lotes")

# c) Entre 3 y 8 defectos (inclusive)
prob_3_a_8 = X.cdf(8) - X.cdf(2)  # P(3 <= X <= 8) = P(X <= 8) - P(X <= 2)
print(f"\n   c) P(3 ≤ X ≤ 8 defectos) = {prob_3_a_8:.6f} ({prob_3_a_8*100:.2f}%)")
print(f"      Interpretación: {prob_3_a_8*100:.2f}% de los lotes tendrán entre 3 y 8 defectos")

# ========================================
# 2. PARÁMETROS ESTADÍSTICOS
# ========================================
print("\n2. PARÁMETROS ESPERADOS:")
print("-"*70)

# Esperanza (media)
mu = X.mean()
print(f"   E(X) = np = {n} × {p} = {mu:.2f} defectos esperados por lote")

# Varianza y desviación estándar
var = X.var()
std = X.std()
print(f"   Var(X) = np(1-p) = {var:.2f}")
print(f"   σ(X) = √[np(1-p)] = {std:.2f} defectos")
print(f"\n   Interpretación: En promedio esperamos {mu:.2f} defectos por lote")
print(f"   con una variabilidad típica de ±{std:.2f} defectos")

# ========================================
# 3. UMBRAL DE RECHAZO (PERCENTIL 95)
# ========================================
print("\n3. UMBRAL DE RECHAZO (PERCENTIL 95):")
print("-"*70)

umbral_95 = X.ppf(0.95)  # Percentil 95
print(f"   Percentil 95: {umbral_95:.0f} defectos")
print(f"   P(X ≤ {umbral_95:.0f}) = {X.cdf(umbral_95):.4f}")
print(f"\n   ✅ RECOMENDACIÓN: Rechazar lotes con más de {umbral_95:.0f} defectos")
print(f"   Esto significa que solo el 5% de los lotes 'normales' serán rechazados")
print(f"   (falsos positivos), mientras capturamos lotes problemáticos.")

# ========================================
# 4. INSPECCIÓN MÚTIPLE DE LOTES
# ========================================
print("\n4. POLÍTICA DE INSPECCIÓN DIARIA (5 lotes):")
print("-"*70)

# Probabilidad de que UN lote tenga más de 10 defectos
p_lote_malo = prob_mas_10

# Probabilidad de que NINGÚN lote (de 5) tenga más de 10 defectos
p_todos_buenos = (1 - p_lote_malo) ** 5

# Probabilidad de encontrar AL MENOS UN lote con más de 10 defectos
p_al_menos_uno = 1 - p_todos_buenos

print(f"   Probabilidad de que un lote tenga > 10 defectos: {p_lote_malo:.6f}")
print(f"   Probabilidad de que los 5 lotes sean aceptables: {p_todos_buenos:.6f}")
print(f"\n   🚨 P(al menos 1 lote con > 10 defectos) = {p_al_menos_uno:.6f} ({p_al_menos_uno*100:.4f}%)")
print(f"\n   Interpretación: En {p_al_menos_uno*100:.2f}% de los días encontraremos")
print(f"   al menos un lote problemático (con más de 10 defectos)")
print(f"   Esto es aproximadamente {p_al_menos_uno*100*30:.0f} días al mes.")

print("\n" + "="*70)

# COMMAND ----------

# DBTITLE 1,📊 Visualizaciones Ejercicio A
# ==========================================
# VISUALIZACIONES - EJERCICIO A
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Valores para graficar (rango razonable de defectos)
k_vals = np.arange(0, 20)
pmf_vals = X.pmf(k_vals)
cdf_vals = X.cdf(k_vals)

# ------------------------------------------
# GRÁFICO 1: DISTRIBUCIÓN DE PROBABILIDAD
# ------------------------------------------
ax1 = axes[0, 0]
ax1.bar(k_vals, pmf_vals, color='steelblue', alpha=0.7, edgecolor='black', width=0.8)

# Marcar la media
ax1.axvline(mu, color='red', linestyle='--', linewidth=2, label=f'E(X) = {mu:.2f}')

# Marcar la región de 1 desviación estándar
ax1.axvspan(mu - std, mu + std, alpha=0.2, color='red', label=f'±1σ [{mu-std:.1f}, {mu+std:.1f}]')

ax1.set_xlabel('Número de chips defectuosos (k)', fontsize=11)
ax1.set_ylabel('Probabilidad P(X = k)', fontsize=11)
ax1.set_title('Distribución de Probabilidad\nBinomial(n=200, p=0.03)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3, axis='y')

# ------------------------------------------
# GRÁFICO 2: PROBABILIDADES ACUMULADAS
# ------------------------------------------
ax2 = axes[0, 1]
ax2.step(k_vals, cdf_vals, where='post', color='darkblue', linewidth=2, label='CDF')

# Marcar percentil 95
ax2.axhline(0.95, color='red', linestyle='--', linewidth=1.5, label='Percentil 95')
ax2.axvline(umbral_95, color='red', linestyle='--', linewidth=1.5)
ax2.plot(umbral_95, 0.95, 'ro', markersize=10, label=f'Umbral = {umbral_95:.0f}')

ax2.set_xlabel('Número de chips defectuosos (k)', fontsize=11)
ax2.set_ylabel('Probabilidad acumulada P(X ≤ k)', fontsize=11)
ax2.set_title('Función de Distribución Acumulada (CDF)\nUmbral de Rechazo al 95%', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# ------------------------------------------
# GRÁFICO 3: ZONAS DE DECISIÓN
# ------------------------------------------
ax3 = axes[1, 0]

# Definir zonas
zona_aceptable = k_vals <= umbral_95
zona_rechazo = k_vals > umbral_95

ax3.bar(k_vals[zona_aceptable], pmf_vals[zona_aceptable], 
        color='green', alpha=0.6, edgecolor='black', width=0.8, label=f'Aceptable (≤ {umbral_95:.0f})')
ax3.bar(k_vals[zona_rechazo], pmf_vals[zona_rechazo], 
        color='red', alpha=0.6, edgecolor='black', width=0.8, label=f'Rechazo (> {umbral_95:.0f})')

# Marcar casos específicos del problema
ax3.plot(5, X.pmf(5), 'bo', markersize=12, label='5 defectos', zorder=5)
ax3.plot(10, X.pmf(10), 'mo', markersize=12, label='10 defectos', zorder=5)

ax3.axvline(umbral_95, color='red', linestyle='--', linewidth=2, alpha=0.7)

ax3.set_xlabel('Número de chips defectuosos (k)', fontsize=11)
ax3.set_ylabel('Probabilidad P(X = k)', fontsize=11)
ax3.set_title('Zonas de Decisión\nCriterio de Aceptación/Rechazo', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3, axis='y')

# ------------------------------------------
# GRÁFICO 4: SIMULACIÓN DE 1000 LOTES
# ------------------------------------------
ax4 = axes[1, 1]

# Simular 1000 lotes
np.random.seed(42)
lotes_simulados = X.rvs(size=1000)

# Histograma de la simulación
ax4.hist(lotes_simulados, bins=range(0, 20), density=True, 
         color='lightblue', alpha=0.7, edgecolor='black', label='Simulación (1000 lotes)')

# Superponer la distribución teórica
ax4.plot(k_vals, pmf_vals, 'r-', linewidth=2, marker='o', markersize=6, label='Teórico')

# Estadísticas de la simulación
media_sim = lotes_simulados.mean()
std_sim = lotes_simulados.std()
ax4.axvline(media_sim, color='green', linestyle='--', linewidth=2, 
            label=f'Media simulada = {media_sim:.2f}')

# Contar lotes rechazados en simulación
lotes_rechazados = np.sum(lotes_simulados > umbral_95)
porcentaje_rechazados = (lotes_rechazados / 1000) * 100

ax4.set_xlabel('Número de chips defectuosos', fontsize=11)
ax4.set_ylabel('Densidad de probabilidad', fontsize=11)
ax4.set_title(f'Simulación vs Teórico\n{lotes_rechazados} lotes rechazados ({porcentaje_rechazados:.1f}%) de 1000', 
              fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ------------------------------------------
# CONCLUSIONES
# ------------------------------------------
print("\n✅ CONCLUSIONES Y RECOMENDACIONES:")
print("="*70)
print(f"\n1. ESPERADO: En promedio, cada lote tendrá {mu:.2f} chips defectuosos")
print(f"   con variabilidad típica de ±{std:.2f} chips.")

print(f"\n2. UMBRAL: Establecer {umbral_95:.0f} defectos como límite de rechazo.")
print(f"   - Solo {(1-X.cdf(umbral_95))*100:.2f}% de lotes normales serán rechazados.")
print(f"   - Esto balancea control de calidad con eficiencia operativa.")

print(f"\n3. INSPECCIÓN DIARIA: Con 5 lotes/día, esperamos encontrar")
print(f"   lotes problemáticos (>10 defectos) aproximadamente {p_al_menos_uno*100*30:.0f} días al mes.")

print(f"\n4. SIMULACIÓN: Validación con 1000 lotes muestra que")
print(f"   - Media simulada: {media_sim:.2f} vs teórica: {mu:.2f} ✓")
print(f"   - {porcentaje_rechazados:.1f}% rechazados vs teórico: {(1-X.cdf(umbral_95))*100:.1f}% ✓")

print(f"\n5. POLÍTICA SUGERIDA:")
print(f"   - Lotes con ≤ {umbral_95:.0f} defectos: ACEPTAR y enviar")
print(f"   - Lotes con > {umbral_95:.0f} defectos: RECHAZAR e investigar causa raíz")
print(f"   - Lotes con > 10 defectos: ALERTA para revisión de proceso")

print("\n" + "="*70)

# COMMAND ----------

# DBTITLE 1,Ejercicio C: Problema
# MAGIC %md
# MAGIC ## 📝 Ejercicio Complementario B: Análisis de Costos Operativos
# MAGIC
# MAGIC ### Contexto Económico - Gestión de Costos Empresariales
# MAGIC
# MAGIC Una cadena de tiendas minoristas analiza los **costos operativos mensuales** de sus sucursales. Los datos históricos muestran que los costos siguen una distribución normal:
# MAGIC
# MAGIC * **Media (μ)**: $320,000 pesos/mes por sucursal
# MAGIC * **Desviación estándar (σ)**: $45,000 pesos
# MAGIC
# MAGIC El directorio necesita establecer políticas de control presupuestario y identificar sucursales con costos atípicos.
# MAGIC
# MAGIC ### Problema
# MAGIC
# MAGIC Como gerente financiero, necesitas:
# MAGIC
# MAGIC 1. **Análisis de rangos de costos**:
# MAGIC    - ¿Qué porcentaje de sucursales tiene costos mensuales entre $280,000 y $360,000?
# MAGIC    - ¿Qué porcentaje excede los $400,000 (alerta de costos altos)?
# MAGIC    - ¿Qué porcentaje está por debajo de $250,000 (posible subreporte)?
# MAGIC
# MAGIC 2. **Estandarización y comparación de sucursales**:
# MAGIC    - Calcular Z-scores para sucursales con costos de $280k, $320k y $410k
# MAGIC    - Comparar dos regiones con diferentes estructuras:
# MAGIC      * Región Norte: μ=$320k, σ=$45k, Sucursal A reporta $385k
# MAGIC      * Región Sur: μ=$280k, σ=$55k, Sucursal B reporta $365k
# MAGIC    - ¿Cuál sucursal tiene costos más atípicos (relativamente)?
# MAGIC
# MAGIC 3. **Intervalos de control presupuestario**:
# MAGIC    - Determinar el intervalo del 95% central de costos (política estándar)
# MAGIC    - Calcular el rango del 68% (variación normal)
# MAGIC    - ¿Cuál es el costo máximo para estar en el top 10% de eficiencia (menores costos)?
# MAGIC
# MAGIC 4. **Identificación de outliers**:
# MAGIC    - Establecer umbrales para costos atípicos usando |Z| > 2
# MAGIC    - ¿Cuántas sucursales (de 50 totales) esperarías encontrar con costos atípicos?
# MAGIC
# MAGIC ### Distribución a Usar
# MAGIC
# MAGIC **Normal**: X ~ N(μ, σ²)
# MAGIC
# MAGIC * Región estándar: X ~ N(320,000, 45,000²)
# MAGIC * Comparaciones con Z-scores
# MAGIC
# MAGIC **Aplicación:** Control presupuestario, benchmarking de sucursales, identificación de ineficiencias

# COMMAND ----------

# DBTITLE 1,📚 Conceptos Clave B
# MAGIC %md
# MAGIC ### 📚 Conceptos Clave para el Ejercicio B
# MAGIC
# MAGIC **1. Distribución Normal en Análisis de Costos:**
# MAGIC * Los costos operativos suelen seguir distribución normal
# MAGIC * μ = costo promedio (punto de referencia)
# MAGIC * σ = variabilidad de costos (dispersión entre sucursales)
# MAGIC
# MAGIC **2. Interpretación de probabilidades:**
# MAGIC * P(a ≤ X ≤ b): Porcentaje de sucursales con costos en ese rango
# MAGIC * P(X > c): Proporción de sucursales con costos altos
# MAGIC * P(X < d): Proporción de sucursales con costos bajos
# MAGIC
# MAGIC **3. Z-scores en análisis empresarial:**
# MAGIC * Z = (X - μ) / σ mide cuántas desviaciones estándar está un costo del promedio
# MAGIC * |Z| > 2: Outlier (requiere investigación)
# MAGIC * Z negativo: Por debajo del promedio (más eficiente)
# MAGIC * Z positivo: Por encima del promedio (menos eficiente)
# MAGIC
# MAGIC **4. Intervalos de control:**
# MAGIC * 68% de sucursales: μ ± 1σ (variación normal)
# MAGIC * 95% de sucursales: μ ± 2σ (rango aceptable)
# MAGIC * 99.7% de sucursales: μ ± 3σ (casi todas)
# MAGIC
# MAGIC **5. Benchmarking:**
# MAGIC * Usar percentiles para identificar sucursales eficientes (P10, P25)
# MAGIC * Establecer metas basadas en percentiles superiores
# MAGIC * Comparar sucursales usando Z-scores (ajusta por contexto regional)

# COMMAND ----------

# DBTITLE 1,💻 Solución Ejercicio B
# ==========================================
# EJERCICIO B: ANÁLISIS DE COSTOS OPERATIVOS
# ==========================================

# Parámetros de la distribución estándar
mu = 320000  # costo promedio mensual (pesos)
sigma = 45000  # desviación estándar

print("="*75)
print("EJERCICIO B: ANÁLISIS DE COSTOS OPERATIVOS")
print("="*75)
print(f"Distribución: X ~ N(μ={mu:,}, σ={sigma:,})")
print(f"\nEscenario: Costos operativos mensuales de sucursales minoristas")
print("="*75)

# Crear distribución Normal
X = stats.norm(loc=mu, scale=sigma)

# ========================================
# 1. ANÁLISIS DE RANGOS DE COSTOS
# ========================================
print("\n1. ANÁLISIS DE RANGOS DE COSTOS:")
print("-"*75)

# a) Entre $280k y $360k
costo_bajo = 280000
costo_alto = 360000
prob_rango = X.cdf(costo_alto) - X.cdf(costo_bajo)

z_bajo = (costo_bajo - mu) / sigma
z_alto = (costo_alto - mu) / sigma

print(f"\n   a) P(${costo_bajo:,} ≤ X ≤ ${costo_alto:,}) = {prob_rango:.4f} ({prob_rango*100:.2f}%)")
print(f"      Z-scores: [{z_bajo:.2f}, {z_alto:.2f}]")
print(f"      Interpretación: {prob_rango*100:.1f}% de las sucursales tienen costos")
print(f"      en este rango (operación estándar)")

# b) Más de $400k (alerta de costos altos)
umbral_alto = 400000
prob_alto = 1 - X.cdf(umbral_alto)
z_umbral_alto = (umbral_alto - mu) / sigma

print(f"\n   b) P(X > ${umbral_alto:,}) = {prob_alto:.4f} ({prob_alto*100:.2f}%)")
print(f"      Z-score: z = {z_umbral_alto:.2f}")
print(f"      Interpretación: {prob_alto*100:.1f}% de sucursales superan ${umbral_alto:,}")
print(f"      De 50 sucursales, ~{prob_alto*50:.0f} requerirían auditoría de costos")

# c) Menos de $250k (posible subreporte)
umbral_bajo = 250000
prob_bajo = X.cdf(umbral_bajo)
z_umbral_bajo = (umbral_bajo - mu) / sigma

print(f"\n   c) P(X < ${umbral_bajo:,}) = {prob_bajo:.4f} ({prob_bajo*100:.2f}%)")
print(f"      Z-score: z = {z_umbral_bajo:.2f}")
print(f"      Interpretación: {prob_bajo*100:.2f}% de sucursales con costos muy bajos")
print(f"      Posible subreporte o alta eficiencia (investigar)")

# ========================================
# 2. ESTANDARIZACIÓN Y COMPARACIÓN
# ========================================
print("\n2. ESTANDARIZACIÓN Y COMPARACIÓN DE SUCURSALES:")
print("-"*75)

# Z-scores para diferentes niveles de costos
costos_eval = [280000, 320000, 410000]
print("\n   Z-scores para diferentes sucursales:")
for costo in costos_eval:
    z = (costo - mu) / sigma
    percentil = X.cdf(costo)
    
    if abs(z) < 1:
        categoria = "NORMAL"
    elif abs(z) < 2:
        categoria = "Moderadamente atípico"
    else:
        categoria = "OUTLIER (investigar)"
    
    print(f"\n   Costo ${costo:,}: Z = {z:+.2f} (percentil {percentil*100:.1f}%)")
    print(f"      {abs(z):.1f} desviaciones estándar {'sobre' if z > 0 else 'bajo'} el promedio")
    print(f"      Categoría: {categoria}")

# Comparación entre regiones
print("\n   COMPARACIÓN ENTRE REGIONES:")

# Región Norte
mu_norte = 320000
sigma_norte = 45000
costo_norte = 385000
z_norte = (costo_norte - mu_norte) / sigma_norte

print(f"\n   Región Norte: μ=${mu_norte:,}, σ=${sigma_norte:,}")
print(f"      Sucursal A: ${costo_norte:,}")
print(f"      Z_A = ({costo_norte:,} - {mu_norte:,}) / {sigma_norte:,} = {z_norte:.2f}")
print(f"      {abs(z_norte):.2f} desviaciones estándar sobre el promedio regional")

# Región Sur
mu_sur = 280000
sigma_sur = 55000
costo_sur = 365000
z_sur = (costo_sur - mu_sur) / sigma_sur

print(f"\n   Región Sur: μ=${mu_sur:,}, σ=${sigma_sur:,}")
print(f"      Sucursal B: ${costo_sur:,}")
print(f"      Z_B = ({costo_sur:,} - {mu_sur:,}) / {sigma_sur:,} = {z_sur:.2f}")
print(f"      {abs(z_sur):.2f} desviaciones estándar sobre el promedio regional")

if abs(z_norte) > abs(z_sur):
    mas_atipica = "Norte (Sucursal A)"
    diferencia = abs(z_norte) - abs(z_sur)
else:
    mas_atipica = "Sur (Sucursal B)"
    diferencia = abs(z_sur) - abs(z_norte)

print(f"\n   🎯 CONCLUSIÓN: Región {mas_atipica} tiene costos MÁS ATÍPICOS")
print(f"      (mayor Z-score = mayor desviación relativa del promedio)")
print(f"      Diferencia: {diferencia:.2f} desviaciones estándar")

# ========================================
# 3. INTERVALOS DE CONTROL PRESUPUESTARIO
# ========================================
print("\n3. INTERVALOS DE CONTROL PRESUPUESTARIO:")
print("-"*75)

# Intervalo 95% central
limite_inf_95 = X.ppf(0.025)
limite_sup_95 = X.ppf(0.975)

print(f"\n   Intervalo del 95% central (política estándar):")
print(f"   [${limite_inf_95:,.0f}, ${limite_sup_95:,.0f}]")
print(f"   Fórmula: μ ± 1.96σ = [${mu - 1.96*sigma:,.0f}, ${mu + 1.96*sigma:,.0f}]")
print(f"\n   💼 Interpretación para control presupuestario:")
print(f"   - 95% de sucursales operan en este rango")
print(f"   - Fuera de este rango: requiere auditoría/explicación")

# Intervalo 68% (±1σ)
rango_1s_inf = mu - sigma
rango_1s_sup = mu + sigma

print(f"\n   Rango del 68% (variación normal):")
print(f"   [${rango_1s_inf:,.0f}, ${rango_1s_sup:,.0f}]")
print(f"   Aproximadamente 2/3 de sucursales en este rango")

# Top 10% eficiencia (costos MÁS BAJOS)
percentil_10 = X.ppf(0.10)

print(f"\n   Top 10% de eficiencia (menores costos):")
print(f"   Costo máximo: ${percentil_10:,.0f}/mes (percentil 10)")
print(f"   Sucursales con costos ≤ ${percentil_10:,.0f} son las más eficientes")
print(f"   Esto es {((mu - percentil_10)/mu)*100:.1f}% menos que el promedio")

# ========================================
# 4. IDENTIFICACIÓN DE OUTLIERS
# ========================================
print("\n4. IDENTIFICACIÓN DE OUTLIERS (|Z| > 2):")
print("-"*75)

# Umbrales usando |Z| > 2
umbral_outlier_bajo = mu - 2*sigma
umbral_outlier_alto = mu + 2*sigma

print(f"\n   Umbrales para outliers (±2σ):")
print(f"   Costo muy bajo: < ${umbral_outlier_bajo:,.0f} (Z < -2)")
print(f"   Costo muy alto: > ${umbral_outlier_alto:,.0f} (Z > +2)")

# Probabilidad de ser outlier
prob_outlier_bajo = X.cdf(umbral_outlier_bajo)
prob_outlier_alto = 1 - X.cdf(umbral_outlier_alto)
prob_outlier_total = prob_outlier_bajo + prob_outlier_alto

print(f"\n   Probabilidades:")
print(f"   P(outlier bajo) = {prob_outlier_bajo:.4f} ({prob_outlier_bajo*100:.2f}%)")
print(f"   P(outlier alto) = {prob_outlier_alto:.4f} ({prob_outlier_alto*100:.2f}%)")
print(f"   P(outlier total) = {prob_outlier_total:.4f} ({prob_outlier_total*100:.2f}%)")

# Expectativa en 50 sucursales
num_sucursales = 50
outliers_esperados_bajos = prob_outlier_bajo * num_sucursales
outliers_esperados_altos = prob_outlier_alto * num_sucursales
outliers_esperados_total = prob_outlier_total * num_sucursales

print(f"\n   De {num_sucursales} sucursales, se esperan aproximadamente:")
print(f"   - {outliers_esperados_bajos:.1f} con costos muy bajos (< ${umbral_outlier_bajo:,.0f})")
print(f"   - {outliers_esperados_altos:.1f} con costos muy altos (> ${umbral_outlier_alto:,.0f})")
print(f"   - {outliers_esperados_total:.1f} outliers en total (~{prob_outlier_total*100:.0f}%)")

print("\n" + "="*75)

# COMMAND ----------

# DBTITLE 1,📊 Visualizaciones Ejercicio B
# ==========================================
# VISUALIZACIONES - EJERCICIO B
# ==========================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Valores para graficar
x_vals = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
pdf_vals = X.pdf(x_vals)

# ------------------------------------------
# GRÁFICO 1: DISTRIBUCIÓN COMPLETA CON INTERVALOS
# ------------------------------------------
ax1 = axes[0, 0]

ax1.plot(x_vals, pdf_vals, 'b-', linewidth=2, label='N(320k, 45k)')
ax1.fill_between(x_vals, 0, pdf_vals, alpha=0.2, color='blue')

# Intervalos clave
ax1.axvspan(rango_1s_inf, rango_1s_sup, alpha=0.2, color='green', label='68% (±1σ)')
ax1.axvspan(limite_inf_95, limite_sup_95, alpha=0.1, color='orange', label='95% (±1.96σ)')
ax1.axvline(mu, color='red', linestyle='--', linewidth=2, label=f'μ = ${mu/1000:.0f}k')

ax1.set_xlabel('Costo mensual ($)', fontsize=11)
ax1.set_ylabel('Densidad de probabilidad', fontsize=11)
ax1.set_title('Distribución de Costos Operativos\nIntervalos de Control', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
ax1.set_xlim(mu - 4*sigma, mu + 4*sigma)

# Formatear eje X en miles
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))

# ------------------------------------------
# GRÁFICO 2: REGLA EMPIRICA (68-95-99.7)
# ------------------------------------------
ax2 = axes[0, 1]

ax2.plot(x_vals, pdf_vals, 'b-', linewidth=2)

# 68% (±1σ)
ax2.fill_between(x_vals, 0, pdf_vals, 
                 where=(x_vals >= mu - sigma) & (x_vals <= mu + sigma),
                 alpha=0.3, color='green', label='68% (±1σ)')

# 95% (±2σ)
ax2.fill_between(x_vals, 0, pdf_vals,
                 where=((x_vals >= mu - 2*sigma) & (x_vals < mu - sigma)) |
                       ((x_vals > mu + sigma) & (x_vals <= mu + 2*sigma)),
                 alpha=0.3, color='orange', label='27% (1-2σ)')

# 99.7% (±3σ)
ax2.fill_between(x_vals, 0, pdf_vals,
                 where=((x_vals >= mu - 3*sigma) & (x_vals < mu - 2*sigma)) |
                       ((x_vals > mu + 2*sigma) & (x_vals <= mu + 3*sigma)),
                 alpha=0.3, color='red', label='4.3% (2-3σ)')

ax2.axvline(mu, color='black', linestyle='--', linewidth=1.5)

for i in [-3, -2, -1, 1, 2, 3]:
    ax2.axvline(mu + i*sigma, color='gray', linestyle=':', linewidth=1, alpha=0.5)

ax2.set_xlabel('Costo mensual ($)', fontsize=11)
ax2.set_ylabel('Densidad', fontsize=11)
ax2.set_title('Regla Empírica (68-95-99.7)\nZonas de Variabilidad', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))

# ------------------------------------------
# GRÁFICO 3: ZONAS DE ALERTA
# ------------------------------------------
ax3 = axes[0, 2]

ax3.plot(x_vals, pdf_vals, 'b-', linewidth=2)

# Zona normal
ax3.fill_between(x_vals, 0, pdf_vals,
                 where=(x_vals >= umbral_outlier_bajo) & (x_vals <= umbral_outlier_alto),
                 alpha=0.3, color='green', label=f'Normal (95.4%)')

# Zona baja (eficiencia excepcional o subreporte)
ax3.fill_between(x_vals, 0, pdf_vals,
                 where=(x_vals < umbral_outlier_bajo),
                 alpha=0.4, color='blue', label=f'Muy bajo ({prob_outlier_bajo*100:.1f}%)')

# Zona alta (costos excesivos)
ax3.fill_between(x_vals, 0, pdf_vals,
                 where=(x_vals > umbral_outlier_alto),
                 alpha=0.4, color='red', label=f'Muy alto ({prob_outlier_alto*100:.1f}%)')

ax3.axvline(umbral_outlier_bajo, color='blue', linestyle='--', linewidth=2)
ax3.axvline(umbral_outlier_alto, color='red', linestyle='--', linewidth=2)

ax3.set_xlabel('Costo mensual ($)', fontsize=11)
ax3.set_ylabel('Densidad', fontsize=11)
ax3.set_title('Zonas de Alerta\nOutliers (|Z| > 2)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))

# ------------------------------------------
# GRÁFICO 4: FUNCIÓN ACUMULADA (CDF)
# ------------------------------------------
ax4 = axes[1, 0]

cdf_vals = X.cdf(x_vals)
ax4.plot(x_vals, cdf_vals, 'b-', linewidth=2, label='CDF')

# Marcar percentiles clave
percentiles_interes = [0.10, 0.50, 0.90, 0.95]
for p in percentiles_interes:
    x_p = X.ppf(p)
    ax4.plot(x_p, p, 'ro', markersize=8)
    ax4.axhline(p, color='gray', linestyle=':', alpha=0.5)
    ax4.axvline(x_p, color='gray', linestyle=':', alpha=0.5)
    ax4.text(x_p, p + 0.03, f'P{int(p*100)}\n${x_p/1000:.0f}k', 
             ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax4.set_xlabel('Costo mensual ($)', fontsize=11)
ax4.set_ylabel('Probabilidad acumulada', fontsize=11)
ax4.set_title('Función de Distribución Acumulada\nPercentiles Clave', fontsize=12, fontweight='bold')
ax4.grid(alpha=0.3)
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))

# ------------------------------------------
# GRÁFICO 5: COMPARACIÓN DE REGIONES (Z-SCORES)
# ------------------------------------------
ax5 = axes[1, 1]

# Distribución Norte
X_norte = stats.norm(loc=mu_norte, scale=sigma_norte)
x_norte_vals = np.linspace(mu_norte - 4*sigma_norte, mu_norte + 4*sigma_norte, 500)
ax5.plot(x_norte_vals, X_norte.pdf(x_norte_vals), 'b-', linewidth=2, label=f'Norte: N({mu_norte/1000:.0f}k, {sigma_norte/1000:.0f}k)')
ax5.axvline(mu_norte, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
ax5.axvline(costo_norte, color='blue', linestyle='-', linewidth=3, label=f'Sucursal A: ${costo_norte/1000:.0f}k (Z={z_norte:.2f})')

# Distribución Sur
X_sur = stats.norm(loc=mu_sur, scale=sigma_sur)
x_sur_vals = np.linspace(mu_sur - 4*sigma_sur, mu_sur + 4*sigma_sur, 500)
ax5.plot(x_sur_vals, X_sur.pdf(x_sur_vals), 'r-', linewidth=2, label=f'Sur: N({mu_sur/1000:.0f}k, {sigma_sur/1000:.0f}k)')
ax5.axvline(mu_sur, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax5.axvline(costo_sur, color='red', linestyle='-', linewidth=3, label=f'Sucursal B: ${costo_sur/1000:.0f}k (Z={z_sur:.2f})')

ax5.set_xlabel('Costo mensual ($)', fontsize=11)
ax5.set_ylabel('Densidad', fontsize=11)
ax5.set_title('Comparación entre Regiones\nEstandarización con Z-scores', fontsize=12, fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)
ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))

# ------------------------------------------
# GRÁFICO 6: ESTANDARIZACIÓN (DISTRIBUCIÓN Z)
# ------------------------------------------
ax6 = axes[1, 2]

# Distribución estándar normal Z ~ N(0, 1)
Z_dist = stats.norm(0, 1)
z_vals = np.linspace(-4, 4, 500)
z_pdf = Z_dist.pdf(z_vals)

ax6.plot(z_vals, z_pdf, 'k-', linewidth=2, label='Z ~ N(0,1)')
ax6.fill_between(z_vals, 0, z_pdf, where=(np.abs(z_vals) <= 1), alpha=0.2, color='green', label='68% (|Z| ≤ 1)')
ax6.fill_between(z_vals, 0, z_pdf, where=(np.abs(z_vals) <= 2), alpha=0.1, color='orange', label='95% (|Z| ≤ 2)')

# Marcar los Z-scores de las sucursales evaluadas
for i, costo in enumerate(costos_eval):
    z = (costo - mu) / sigma
    color = ['green', 'black', 'red'][i]
    ax6.axvline(z, color=color, linestyle='--', linewidth=2, alpha=0.7,
                label=f'${costo/1000:.0f}k: Z={z:.2f}')
    ax6.plot(z, 0, 'o', color=color, markersize=10)

# Zonas de outliers
ax6.axvline(-2, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
ax6.axvline(2, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
ax6.text(-2, 0.35, 'Outlier\n(<-2)', ha='center', fontsize=8, color='red')
ax6.text(2, 0.35, 'Outlier\n(>+2)', ha='center', fontsize=8, color='red')

ax6.set_xlabel('Z-score (desviaciones estándar)', fontsize=11)
ax6.set_ylabel('Densidad', fontsize=11)
ax6.set_title('Distribución Estandarizada\nInterpretación de Z-scores', fontsize=12, fontweight='bold')
ax6.legend(fontsize=8, loc='upper right')
ax6.grid(alpha=0.3)
ax6.set_xlim(-4, 4)

plt.tight_layout()
plt.show()

# ------------------------------------------
# CONCLUSIONES
# ------------------------------------------
print("\n✅ CONCLUSIONES Y RECOMENDACIONES:")
print("="*75)

print(f"\n1. CONTROL PRESUPUESTARIO:")
print(f"   - Intervalo aceptable (95%): [${limite_inf_95:,.0f}, ${limite_sup_95:,.0f}]")
print(f"   - Sucursales fuera de este rango requieren auditoría")
print(f"   - De 50 sucursales, ~{(1-0.95)*50:.0f} estarán fuera del rango (esperado)")

print(f"\n2. EFICIENCIA OPERATIVA:")
print(f"   - Top 10% más eficientes: costos ≤ ${percentil_10:,.0f}")
print(f"   - Usar estas sucursales como benchmarks")
print(f"   - Ahorro potencial: ${(mu - percentil_10):,.0f}/mes por sucursal")

print(f"\n3. ALERTAS DE COSTOS:")
print(f"   - Alerta inmediata si costo > ${umbral_alto:,.0f} (Z > {z_umbral_alto:.1f})")
print(f"   - Investigación si costo < ${umbral_bajo:,.0f} (Z < {z_umbral_bajo:.1f})")
print(f"   - Outliers (|Z| > 2): requieren explicación obligatoria")

print(f"\n4. BENCHMARKING REGIONAL:")
print(f"   - Región {mas_atipica} muestra mayor desviación")
print(f"   - Comparar usando Z-scores (ajusta por contexto)")
print(f"   - No comparar valores absolutos entre regiones diferentes")

print(f"\n5. POLÍTICA DE GESTIÓN:")
print(f"   - Monitoreo mensual de Z-scores por sucursal")
print(f"   - Establecer metas de reducción: mover μ hacia percentil 25")
print(f"   - Reducir σ: mejorar homogeneidad operativa")
print(f"   - Compartir mejores prácticas de sucursales eficientes")

print("\n" + "="*75)

# COMMAND ----------

# DBTITLE 1,Ejercicio C: Problema
# MAGIC %md
# MAGIC ## 📝 Ejercicio Complementario C: Estimación de Ventas con TCL
# MAGIC
# MAGIC ### Contexto Económico - Proyección de Ingresos
# MAGIC
# MAGIC Una empresa analiza las **transacciones diarias** de sus clientes para proyectar ingresos. Los montos de transacciones individuales **NO** siguen distribución normal (tienen sesgo positivo - pocas ventas muy grandes):
# MAGIC
# MAGIC * **Distribución individual**: Gamma con parámetros shape=2, scale=50
# MAGIC * **Media poblacional (μ)**: $100 por transacción
# MAGIC * **Desv. Est. poblacional (σ)**: $70.71 por transacción
# MAGIC * **Sesgo**: Positivo (cola derecha larga)
# MAGIC
# MAGIC Sin embargo, según el **Teorema Central del Límite**, el promedio de múltiples transacciones **SÍ** se distribuye normalmente para muestras grandes.
# MAGIC
# MAGIC ### Problema
# MAGIC
# MAGIC Como analista financiero, necesitas:
# MAGIC
# MAGIC 1. **Verificar el TCL experimentalmente**:
# MAGIC    - Simular 10,000 transacciones individuales de la población (Gamma)
# MAGIC    - Tomar 1,000 muestras de tamaño n = 5, 10, 30, 50
# MAGIC    - Calcular la media de cada muestra
# MAGIC    - Observar cómo la distribución de medias se vuelve normal
# MAGIC
# MAGIC 2. **Calcular errores estándar**:
# MAGIC    - Para cada n, calcular: SE = σ / √n
# MAGIC    - Comparar SE teórico vs. observado
# MAGIC    - Cuantificar reducción de variabilidad con n
# MAGIC
# MAGIC 3. **Aplicación en proyección de ingresos**:
# MAGIC    - Si proyectas ventas promedio con 30 transacciones, ¿cuál es el SE?
# MAGIC    - ¿Cuál es el intervalo de confianza 95% para la media?
# MAGIC    - ¿Cuántas transacciones necesitas para SE ≤ $10?
# MAGIC
# MAGIC 4. **Visualización y conclusiones**:
# MAGIC    - Comparar histogramas de medias muestrales (diferentes n)
# MAGIC    - Superponer curva normal teórica
# MAGIC    - Verificar convergencia a normalidad
# MAGIC
# MAGIC ### Distribución a Usar
# MAGIC
# MAGIC **Población (transacciones individuales)**: X ~ Gamma(shape=2, scale=50)
# MAGIC
# MAGIC * E(X) = shape × scale = 2 × 50 = $100
# MAGIC * Var(X) = shape × scale² = 2 × 50² = 5,000
# MAGIC * σ(X) = √5,000 ≈ $70.71
# MAGIC * **NO es normal** (sesgo positivo)
# MAGIC
# MAGIC **Distribución de medias muestrales** (n transacciones):
# MAGIC
# MAGIC Según el TCL: X̄ ~ N(μ, σ²/n) para n ≥ 30
# MAGIC
# MAGIC * E(X̄) = μ = $100
# MAGIC * SE(X̄) = σ/√n = $70.71/√n
# MAGIC * **SÍ es normal** (incluso si X no lo es)

# COMMAND ----------

# DBTITLE 1,📚 Conceptos Clave C
# MAGIC %md
# MAGIC ### 📚 Conceptos Clave para el Ejercicio C
# MAGIC
# MAGIC **1. Teorema Central del Límite (TCL):**
# MAGIC * La distribución de medias muestrales tiende a ser normal
# MAGIC * **Funciona incluso si la población NO es normal** (¡esto es lo poderoso!)
# MAGIC * Requiere n ≥ 30 para convergencia razonable
# MAGIC
# MAGIC **2. Propiedades de la distribución de medias:**
# MAGIC * E(X̄) = μ (la media de medias = media poblacional)
# MAGIC * Var(X̄) = σ²/n (la varianza SE REDUCE con n)
# MAGIC * SE(X̄) = σ/√n (error estándar de la media)
# MAGIC
# MAGIC **3. Reducción de variabilidad:**
# MAGIC * Duplicar la precisión requiere 4x más muestras
# MAGIC * Para SE = k, necesitas n = (σ/k)²
# MAGIC * Ejemplo: σ=$70, para SE=$10 necesitas n = (70/10)² = 49
# MAGIC
# MAGIC **4. Intervalo de confianza 95%:**
# MAGIC * IC = X̄ ± 1.96 × SE
# MAGIC * Para n=30: IC = X̄ ± 1.96 × ($70.71/√30) = X̄ ± $25.29
# MAGIC
# MAGIC **5. Aplicación en finanzas:**
# MAGIC * Proyección de ingresos promedio
# MAGIC * Estimación de costos promedio
# MAGIC * Diseño de tamaño de muestra para auditorías
# MAGIC * Validación de métodos estadísticos (justifica uso de pruebas paramétricas)
# MAGIC
# MAGIC **6. Por qué es importante:**
# MAGIC * Permite usar métodos normales (intervalos, pruebas t) incluso con poblaciones no normales
# MAGIC * Fundamental para inferencia estadística
# MAGIC * Base de muestreo y estimación en economía

# COMMAND ----------

# DBTITLE 1,💻 Solución Ejercicio C
# ==========================================
# EJERCICIO C: TEOREMA CENTRAL DEL LÍMITE
# ==========================================

# Parámetros de la población (Gamma - NO NORMAL)
shape_param = 2
scale_param = 50
mu_poblacion = shape_param * scale_param  # E(X) = 100
sigma_poblacion = np.sqrt(shape_param * scale_param**2)  # σ = 70.71

print("="*80)
print("EJERCICIO C: TEOREMA CENTRAL DEL LÍMITE EN PROYECCIÓN DE INGRESOS")
print("="*80)
print(f"Población: X ~ Gamma(shape={shape_param}, scale={scale_param})")
print(f"\n⚠️  IMPORTANTE: La población NO es normal (sesgo positivo)")
print(f"   E(X) = μ = ${mu_poblacion:.2f} por transacción")
print(f"   σ(X) = ${sigma_poblacion:.2f}")
print("="*80)

# ========================================
# 1. SIMULACIÓN DE POBLACIÓN
# ========================================
print("\n1. SIMULACIÓN DE POBLACIÓN (10,000 transacciones):")
print("-"*80)

np.random.seed(42)
poblacion = stats.gamma.rvs(a=shape_param, scale=scale_param, size=10000)

media_pob = poblacion.mean()
std_pob = poblacion.std()
mediana_pob = np.median(poblacion)

print(f"   Media observada: ${media_pob:.2f} (teórica: ${mu_poblacion:.2f})")
print(f"   Mediana observada: ${mediana_pob:.2f} (< media por sesgo)")
print(f"   Desv. Est. observada: ${std_pob:.2f} (teórica: ${sigma_poblacion:.2f})")
print(f"   \n   📊 Sesgo positivo confirmado: mediana < media")
print(f"   (Pocas transacciones muy grandes jalan la media hacia arriba)")

# ========================================
# 2. MUESTREO Y CÁLCULO DE MEDIAS
# ========================================
print("\n2. MUESTREO Y CÁLCULO DE MEDIAS MUESTRALES:")
print("-"*80)

tamanios = [5, 10, 30, 50]
num_muestras = 1000
resultados = {}

for n in tamanios:
    # Tomar 1000 muestras de tamaño n
    medias_muestrales = []
    for _ in range(num_muestras):
        muestra = np.random.choice(poblacion, size=n, replace=False)
        medias_muestrales.append(muestra.mean())
    
    medias_muestrales = np.array(medias_muestrales)
    
    # Estadísticas de las medias
    media_de_medias = medias_muestrales.mean()
    std_de_medias = medias_muestrales.std()
    se_teorico = sigma_poblacion / np.sqrt(n)
    
    # Almacenar resultados
    resultados[n] = {
        'medias': medias_muestrales,
        'media_de_medias': media_de_medias,
        'std_observado': std_de_medias,
        'se_teorico': se_teorico
    }
    
    print(f"\n   MUESTRAS DE TAMAÑO n = {n}:")
    print(f"   " + "="*70)
    print(f"   Media de las {num_muestras} medias: ${media_de_medias:.2f}")
    print(f"   Desv. Est. de las medias (observado): ${std_de_medias:.2f}")
    print(f"   Error estándar teórico (SE = σ/√n): ${se_teorico:.2f}")
    print(f"   Diferencia (observado - teórico): ${abs(std_de_medias - se_teorico):.2f}")
    print(f"   Reducción de variabilidad vs población: {sigma_poblacion/std_de_medias:.1f}x")

# ========================================
# 3. APLICACIÓN EN PROYECCIÓN (n=30)
# ========================================
print("\n3. APLICACIÓN EN PROYECCIÓN DE INGRESOS (n=30):")
print("-"*80)

n_proyeccion = 30
se_proyeccion = sigma_poblacion / np.sqrt(n_proyeccion)
margen_error_95 = 1.96 * se_proyeccion

print(f"\n   Error estándar: SE = σ/√n = ${sigma_poblacion:.2f}/√{n_proyeccion} = ${se_proyeccion:.2f}")
print(f"\n   Intervalo de confianza 95% para la media:")
print(f"   IC = μ ± 1.96 × SE")
print(f"   IC = ${mu_poblacion:.2f} ± ${margen_error_95:.2f}")
print(f"   IC = [${mu_poblacion - margen_error_95:.2f}, ${mu_poblacion + margen_error_95:.2f}]")

print(f"\n   💼 Interpretación para negocios:")
print(f"   Si proyectas ingresos con {n_proyeccion} transacciones:")
print(f"   - La media estará dentro de ±${margen_error_95:.2f} del promedio poblacional")
print(f"   - Con 95% de confianza")
print(f"   - Margen de error: {(margen_error_95/mu_poblacion)*100:.1f}% del promedio")

# ========================================
# 4. DISEÑO DE MUESTRA (SE ≤ $10)
# ========================================
print("\n4. DISEÑO DE TAMAÑO DE MUESTRA:")
print("-"*80)

se_objetivo = 10
n_requerido = (sigma_poblacion / se_objetivo)**2

print(f"\n   Objetivo: SE ≤ ${se_objetivo:.2f}")
print(f"   \n   Fórmula: n = (σ / SE_objetivo)²")
print(f"   n = (${sigma_poblacion:.2f} / ${se_objetivo:.2f})²")
print(f"   n = {n_requerido:.0f} transacciones")

se_real = sigma_poblacion / np.sqrt(n_requerido)
print(f"\n   Verificación: SE con n={n_requerido:.0f} es ${se_real:.2f} ✓")

print(f"\n   👉 Para reducir margen de error a la mitad, necesitas 4x más muestras")
print(f"   Ejemplo: SE=$20 con n=12, para SE=$10 necesitas n=50 (4.2x más)")

# ========================================
# 5. VERIFICACIÓN DEL TCL
# ========================================
print("\n5. VERIFICACIÓN EXPERIMENTAL DEL TEOREMA CENTRAL DEL LÍMITE:")
print("-"*80)

print(f"\n   ✅ CONDICIONES DEL TCL:")
print(f"   1. Muestras aleatorias independientes: ✓")
print(f"   2. Tamaño de muestra suficiente (n ≥ 30): ✓")
print(f"   3. Población con varianza finita: ✓")

print(f"\n   🎯 PREDICCIONES DEL TCL:")
print(f"   1. E(X̄) → μ = ${mu_poblacion:.2f}")
for n in tamanios:
    diff = abs(resultados[n]['media_de_medias'] - mu_poblacion)
    print(f"      n={n:2d}: Media observada = ${resultados[n]['media_de_medias']:.2f} (diff: ${diff:.2f})")

print(f"\n   2. SE(X̄) = σ/√n")
for n in tamanios:
    obs = resultados[n]['std_observado']
    teo = resultados[n]['se_teorico']
    diff = abs(obs - teo)
    print(f"      n={n:2d}: SE observado = ${obs:.2f}, teórico = ${teo:.2f} (diff: ${diff:.2f})")

print(f"\n   3. X̄ ~ N(μ, σ/√n) para n ≥ 30 (incluso con población NO normal)")
print(f"      Verificación visual en gráficos a continuación...")

print("\n" + "="*80)

# COMMAND ----------

# DBTITLE 1,📊 Visualizaciones Ejercicio C
# ==========================================
# VISUALIZACIONES - EJERCICIO C
# ==========================================

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

# ------------------------------------------
# GRÁFICO 1: POBLACIÓN ORIGINAL (NO NORMAL)
# ------------------------------------------
ax1 = fig.add_subplot(gs[0, :])

ax1.hist(poblacion, bins=60, density=True, alpha=0.6, color='orange', edgecolor='black', label='Población (10,000 obs.)')

# Superponer distribución teórica Gamma
x_gamma = np.linspace(0, 400, 500)
y_gamma = stats.gamma.pdf(x_gamma, a=shape_param, scale=scale_param)
ax1.plot(x_gamma, y_gamma, 'r-', linewidth=3, label=f'Gamma teórica (shape={shape_param}, scale={scale_param})')

# Marcar estadísticos
ax1.axvline(media_pob, color='blue', linestyle='--', linewidth=2.5, label=f'Media = ${media_pob:.2f}')
ax1.axvline(mediana_pob, color='green', linestyle='--', linewidth=2.5, label=f'Mediana = ${mediana_pob:.2f}')

# Anotar sesgo
ax1.text(0.70, 0.85, '⚠️  POBLACIÓN NO NORMAL\nSesgo positivo (cola derecha)\nMediana < Media', 
         transform=ax1.transAxes, fontsize=11, 
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
         verticalalignment='top')

ax1.set_xlabel('Monto de transacción ($)', fontsize=12)
ax1.set_ylabel('Densidad de probabilidad', fontsize=12)
ax1.set_title('Distribución de la Población\nTransacciones Individuales (NO Normal)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(alpha=0.3)
ax1.set_xlim(0, 400)

# ------------------------------------------
# GRÁFICOS 2-5: DISTRIBUCIONES DE MEDIAS
# ------------------------------------------
for i, n in enumerate(tamanios):
    row = 1 + i // 2
    col = i % 2
    ax = fig.add_subplot(gs[row, col])
    
    medias = resultados[n]['medias']
    media_obs = resultados[n]['media_de_medias']
    std_obs = resultados[n]['std_observado']
    se_teo = resultados[n]['se_teorico']
    
    # Histograma de medias muestrales
    ax.hist(medias, bins=40, density=True, alpha=0.6, color='steelblue', edgecolor='black', label=f'Medias (1000 muestras)')
    
    # Superponer distribución normal teórica
    x_norm = np.linspace(medias.min(), medias.max(), 200)
    y_norm = stats.norm.pdf(x_norm, loc=mu_poblacion, scale=se_teo)
    ax.plot(x_norm, y_norm, 'r-', linewidth=3, label=f'N(μ={mu_poblacion:.0f}, SE={se_teo:.1f})')
    
    # Marcar estadísticos
    ax.axvline(media_obs, color='green', linestyle='--', linewidth=2, label=f'Media obs. = ${media_obs:.2f}')
    ax.axvline(mu_poblacion, color='red', linestyle=':', linewidth=2, label=f'μ = ${mu_poblacion:.2f}')
    
    # Zona ±1 SE
    ax.axvspan(mu_poblacion - se_teo, mu_poblacion + se_teo, alpha=0.15, color='red')
    
    # Indicador de normalidad
    if n >= 30:
        normalidad = "✅ NORMAL"
        color_nota = 'lightgreen'
    else:
        normalidad = "⚠️  Convergiendo"
        color_nota = 'lightyellow'
    
    ax.text(0.97, 0.97, f'n = {n}\n{normalidad}', 
            transform=ax.transAxes, fontsize=10, 
            bbox=dict(boxstyle='round', facecolor=color_nota, alpha=0.8),
            verticalalignment='top', horizontalalignment='right')
    
    ax.set_xlabel('Media muestral ($)', fontsize=10)
    ax.set_ylabel('Densidad', fontsize=10)
    ax.set_title(f'Distribución de Medias Muestrales (n={n})\nSE observado = ${std_obs:.2f}, teórico = ${se_teo:.2f}', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

plt.suptitle('🎯 TEOREMA CENTRAL DEL LÍMITE: De Población NO Normal a Medias Normales', 
             fontsize=16, fontweight='bold', y=0.995)

plt.show()

# ------------------------------------------
# GRÁFICO ADICIONAL: COMPARACIÓN Q-Q PLOT
# ------------------------------------------
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))

for i, n in enumerate(tamanios):
    ax = axes2[i // 2, i % 2]
    medias = resultados[n]['medias']
    
    # Q-Q plot (cuantiles observados vs teóricos)
    stats.probplot(medias, dist="norm", plot=ax)
    
    # Personalizar
    ax.get_lines()[0].set_marker('o')
    ax.get_lines()[0].set_markersize(4)
    ax.get_lines()[0].set_color('steelblue')
    ax.get_lines()[0].set_alpha(0.6)
    ax.get_lines()[1].set_color('red')
    ax.get_lines()[1].set_linewidth(2)
    
    if n >= 30:
        estado = "✅ Ajuste excelente a Normal"
        color_fondo = 'lightgreen'
    else:
        estado = "⚠️  Convergiendo a Normal"
        color_fondo = 'lightyellow'
    
    ax.set_title(f'Q-Q Plot: n={n}\n{estado}', fontsize=11, fontweight='bold')
    ax.text(0.05, 0.95, f'{1000} muestras', transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color_fondo, alpha=0.7))
    ax.grid(alpha=0.3)

plt.suptitle('Verificación de Normalidad: Q-Q Plots\n(Puntos cerca de la línea roja = distribución normal)', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# ------------------------------------------
# GRÁFICO: REDUCCIÓN DE ERROR ESTÁNDAR
# ------------------------------------------
fig3, ax3 = plt.subplots(1, 1, figsize=(12, 7))

n_range = np.arange(1, 101)
se_range = sigma_poblacion / np.sqrt(n_range)

ax3.plot(n_range, se_range, 'b-', linewidth=3, label='SE = σ/√n')
ax3.fill_between(n_range, 0, se_range, alpha=0.2, color='blue')

# Marcar los n simulados
for n in tamanios:
    se = sigma_poblacion / np.sqrt(n)
    ax3.plot(n, se, 'ro', markersize=12, zorder=5)
    ax3.text(n, se + 3, f'n={n}\nSE=${se:.1f}', ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Marcar n para SE objetivo
ax3.axhline(se_objetivo, color='green', linestyle='--', linewidth=2, label=f'SE objetivo = ${se_objetivo}')
ax3.axvline(n_requerido, color='green', linestyle='--', linewidth=2)
ax3.plot(n_requerido, se_objetivo, 'gs', markersize=14, zorder=5, label=f'n={n_requerido:.0f} para SE≤$10')

# Zona recomendada (n ≥ 30)
ax3.axvspan(30, 100, alpha=0.1, color='green', label='n ≥ 30 (TCL confiable)')

ax3.set_xlabel('Tamaño de muestra (n)', fontsize=12)
ax3.set_ylabel('Error Estándar (SE = σ/√n) en $', fontsize=12)
ax3.set_title('Reducción del Error Estándar con el Tamaño de Muestra\nσ = $70.71', 
              fontsize=13, fontweight='bold')
ax3.legend(fontsize=10, loc='upper right')
ax3.grid(alpha=0.3)
ax3.set_xlim(0, 100)
ax3.set_ylim(0, 75)

plt.tight_layout()
plt.show()

# ------------------------------------------
# CONCLUSIONES
# ------------------------------------------
print("\n✅ CONCLUSIONES Y APLICACIONES:")
print("="*80)

print(f"\n1. VERIFICACIÓN DEL TCL:")
print(f"   ✓ La población es Gamma (NO normal, con sesgo positivo)")
print(f"   ✓ Las medias muestrales SÍ son aproximadamente normales para n ≥ 30")
print(f"   ✓ E(X̄) → μ y SE(X̄) → σ/√n (verificado experimentalmente)")

print(f"\n2. APLICACIÓN EN PROYECCIÓN DE INGRESOS:")
print(f"   Con {n_proyeccion} transacciones:")
print(f"   - Ingreso promedio proyectado: ${mu_poblacion:.2f} ± ${margen_error_95:.2f} (95% CI)")
print(f"   - Error estándar: ${se_proyeccion:.2f}")
print(f"   - Margen de error: {(margen_error_95/mu_poblacion)*100:.1f}% del promedio")

print(f"\n3. DISEÑO DE MUESTRA:")
print(f"   - Para SE ≤ ${se_objetivo}: necesitas n ≥ {n_requerido:.0f} transacciones")
print(f"   - Duplicar precisión (SE/2) requiere 4x más muestras")
print(f"   - Triplicar precisión (SE/3) requiere 9x más muestras")

print(f"\n4. IMPLICACIONES PRÁCTICAS:")
print(f"   ✓ Podemos usar intervalos de confianza normales incluso con datos NO normales")
print(f"   ✓ Pruebas t y otros métodos paramétricos son válidos para n ≥ 30")
print(f"   ✓ Fundamental para auditoría, encuestas, y muestreo en economía")

print(f"\n5. REGLA PRÁCTICA:")
print(f"   - n < 30: Usar solo si población es normal o casi normal")
print(f"   - n ≥ 30: TCL garantiza normalidad de X̄ (usar métodos paramétricos)")
print(f"   - n ≥ 50: Muy robusto, incluso con poblaciones muy sesgadas")

print(f"\n6. MENSAJE CLAVE:")
print(f"   🎯 El TCL es la base de la inferencia estadística")
print(f"   Permite generalizar de muestras a poblaciones con confianza cuantificada")

print("\n" + "="*80)

# COMMAND ----------

# DBTITLE 1,🎯 Conclusiones Finales
# MAGIC %md
# MAGIC # 🎯 Conclusiones Finales del Notebook
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Resumen de Distribuciones y Aplicaciones Económicas
# MAGIC
# MAGIC | Distribución | Aplicación Empresarial | Pregunta Clave | Método |
# MAGIC |---|---|---|---|
# MAGIC | **Binomial** | Control de calidad (lotes con defectos) | ¿Cuántos defectos esperar en n ensayos? | P(X=k), P(X≤k), percentiles |
# MAGIC | **Normal** | Costos operativos, salarios, ventas | ¿Qué % está en cierto rango? ¿Outliers? | P(a≤X≤b), intervalos, Z-scores |
# MAGIC | **TCL** | Proyección de ingresos, encuestas | ¿Cuál es el error de la media muestral? | SE = σ/√n, IC, n = (σ/SE)² |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Conceptos Clave Aprendidos
# MAGIC
# MAGIC ### 1. **Distribución Binomial** (Ejercicio A)
# MAGIC * Modela **eventos discretos con probabilidad constante** (defectos, defaults, conversiones)
# MAGIC * Parámetros: n (ensayos), p (probabilidad de éxito)
# MAGIC * E(X) = np, σ(X) = √[np(1-p)]
# MAGIC * **Aplicación:** Control de calidad, dimensionamiento de provisiones, gestión de riesgos
# MAGIC
# MAGIC ### 2. **Distribución Normal y Estandarización** (Ejercicio B)
# MAGIC * La distribución continua más importante en economía
# MAGIC * Parámetros: μ (media), σ (desviación estándar)
# MAGIC * **Regla Empírica:** 68% (±1σ), 95% (±2σ), 99.7% (±3σ)
# MAGIC * **Z-scores:** Permiten comparar observaciones en diferentes escalas
# MAGIC   - Z = (X - μ) / σ
# MAGIC   - |Z| > 2 → Outlier (requiere investigación)
# MAGIC * **Aplicación:** Control presupuestario, benchmarking, identificación de anomalías
# MAGIC
# MAGIC ### 3. **Teorema Central del Límite** (Ejercicio C)
# MAGIC * **Enunciado:** La distribución de medias muestrales tiende a ser normal, 
# MAGIC   **incluso si la población NO es normal**
# MAGIC * Requiere: n ≥ 30 (tamaño de muestra suficiente)
# MAGIC * Propiedades:
# MAGIC   - E(X̄) = μ (media de medias = media poblacional)
# MAGIC   - SE(X̄) = σ/√n (error estándar de la media)
# MAGIC   - X̄ ~ N(μ, σ/√n) para n grande
# MAGIC * **Implicaciones:**
# MAGIC   - Justifica el uso de intervalos de confianza normales
# MAGIC   - Base de pruebas t y otros métodos paramétricos
# MAGIC   - Permite inferencia estadística robusta
# MAGIC * **Aplicación:** Proyección de ingresos, encuestas, auditorías, diseño de muestras
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💼 Aplicaciones en Ciencias Económicas
# MAGIC
# MAGIC ### **Contabilidad y Auditoría**
# MAGIC * ✅ Muestreo de facturas/transacciones (TCL)
# MAGIC * ✅ Control presupuestario y detección de anomalías (Normal + Z-scores)
# MAGIC * ✅ Estimación de errores y provisiones (Binomial, TCL)
# MAGIC
# MAGIC ### **Finanzas**
# MAGIC * ✅ Análisis de riesgo crediticio (Binomial: defaults en carteras)
# MAGIC * ✅ Proyección de ingresos y flujos de caja (TCL)
# MAGIC * ✅ Intervalos de confianza para retornos (Normal, TCL)
# MAGIC
# MAGIC ### **Gestión de Operaciones**
# MAGIC * ✅ Control de calidad y procesos (Binomial)
# MAGIC * ✅ Benchmarking de sucursales/unidades (Normal + Z-scores)
# MAGIC * ✅ Diseño de capacidad y dimensionamiento (todas)
# MAGIC
# MAGIC ### **Economía Aplicada**
# MAGIC * ✅ Diseño de encuestas y tamaño de muestra (TCL)
# MAGIC * ✅ Inferencia estadística y pruebas de hipótesis (todas)
# MAGIC * ✅ Evaluación de políticas públicas (TCL)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔑 Fórmulas Clave para Recordar
# MAGIC
# MAGIC ### **Binomial:**
# MAGIC * P(X = k) = C(n,k) × pᵏ × (1-p)ⁿ⁻ᵏ
# MAGIC * E(X) = np
# MAGIC * σ(X) = √[np(1-p)]
# MAGIC
# MAGIC ### **Normal:**
# MAGIC * X ~ N(μ, σ²)
# MAGIC * Z = (X - μ) / σ → Z ~ N(0, 1)
# MAGIC * P(μ - 2σ ≤ X ≤ μ + 2σ) ≈ 0.95
# MAGIC
# MAGIC ### **Teorema Central del Límite:**
# MAGIC * X̄ ~ N(μ, σ²/n) para n ≥ 30
# MAGIC * SE = σ / √n (error estándar)
# MAGIC * IC 95% = X̄ ± 1.96 × SE
# MAGIC * Diseño de muestra: n = (σ / SE_objetivo)²
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 Próximos Pasos
# MAGIC
# MAGIC 1. **Practicar con datos reales** de su organización
# MAGIC 2. **Aplicar estos métodos** en proyectos de economía y administración
# MAGIC 3. **Combinar distribuciones** para análisis más complejos
# MAGIC 4. **Profundizar en inferencia estadística** (pruebas de hipótesis, regresión)
# MAGIC 5. **Explorar otras distribuciones** (Exponencial, Weibull, Log-Normal)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 Mensaje Final
# MAGIC
# MAGIC > **Las distribuciones de probabilidad son el lenguaje de la incertidumbre.**
# MAGIC >
# MAGIC > En economía, contabilidad y administración, casi todas las decisiones se toman bajo incertidumbre:
# MAGIC > * ¿Cuánto venderé el próximo mes?
# MAGIC > * ¿Cuántos clientes incumplirán sus pagos?
# MAGIC > * ¿Está mi sucursal operando dentro de rangos normales?
# MAGIC >
# MAGIC > **Dominar estas herramientas te permite:**
# MAGIC > * ✅ Cuantificar riesgos y oportunidades
# MAGIC > * ✅ Tomar decisiones basadas en datos, no en intuición
# MAGIC > * ✅ Comunicar incertidumbre de manera precisa
# MAGIC > * ✅ Evaluar y comparar alternativas objetivamente
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎓 ¡Felicitaciones por completar estos ejercicios!
# MAGIC
# MAGIC Has adquirido habilidades fundamentales para análisis estadístico aplicado a ciencias económicas.
# MAGIC
# MAGIC **¡Sigue practicando y aplicando estos conceptos en tu carrera profesional!** 🚀