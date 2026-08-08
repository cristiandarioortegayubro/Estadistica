# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Introducción
# MAGIC %md
# MAGIC # Ejercicios Complementarios - Probabilidad Básica
# MAGIC
# MAGIC Este notebook contiene ejercicios adicionales para reforzar los conceptos de probabilidad aplicados a problemas de negocios.
# MAGIC
# MAGIC ## Contenido
# MAGIC
# MAGIC **Ejercicio A: Selección de Personal**
# MAGIC * Probabilidad Total
# MAGIC * Teorema de Bayes
# MAGIC * Inversión de condicionales
# MAGIC
# MAGIC **Ejercicio B: Control de Inventario**
# MAGIC * Probabilidad Total con múltiples eventos
# MAGIC * Análisis de riesgo por proveedor
# MAGIC * Teorema de Bayes con 3 opciones
# MAGIC
# MAGIC **Ejercicio C: Análisis de Mercado**
# MAGIC * Test de Independencia Estadística
# MAGIC * Regla de la Suma (unión de eventos)
# MAGIC * Probabilidad Condicional
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC 📝 **Nota:** Estos ejercicios complementan la práctica principal. Se recomienda completar primero los ejercicios básicos.

# COMMAND ----------

# DBTITLE 1,Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Configuración para reproducibilidad
np.random.seed(42)

# Configuración de visualización con fondo blanco y sin recuadro
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

print("✅ Librerías importadas correctamente")

# COMMAND ----------

# DBTITLE 1,Ejercicio A: Selección de Personal
# MAGIC %md
# MAGIC ## Ejercicio A: Selección de Personal
# MAGIC
# MAGIC ### Contexto de Negocios
# MAGIC Una empresa recibe 100 solicitudes de empleo: 60 de mujeres y 40 de hombres. Se sabe que:
# MAGIC * **30%** de las mujeres tienen experiencia relevante
# MAGIC * **50%** de los hombres tienen experiencia relevante
# MAGIC
# MAGIC ### Preguntas
# MAGIC
# MAGIC Si se selecciona una solicitud al azar:
# MAGIC
# MAGIC 1. **¿Cuál es la probabilidad de que tenga experiencia relevante?**
# MAGIC    - Aplicaremos la Ley de Probabilidad Total
# MAGIC
# MAGIC 2. **Si una solicitud tiene experiencia, ¿cuál es la probabilidad de que sea de una mujer?**
# MAGIC    - Usaremos el Teorema de Bayes para "invertir" la condicional

# COMMAND ----------

# DBTITLE 1,📚 Conceptos: Probabilidad Total y Teorema de Bayes
# MAGIC %md
# MAGIC ### 📚 Conceptos Fundamentales
# MAGIC
# MAGIC **Ley de Probabilidad Total**
# MAGIC
# MAGIC Cuando un evento B puede ocurrir a través de varios escenarios mutuamente excluyentes (A₁, A₂, ..., Aₙ):
# MAGIC
# MAGIC $$P(B) = P(B|A_1) \cdot P(A_1) + P(B|A_2) \cdot P(A_2) + ... + P(B|A_n) \cdot P(A_n)$$
# MAGIC
# MAGIC **En nuestro problema:**
# MAGIC * B = "Tiene experiencia"
# MAGIC * A₁ = "Es mujer", A₂ = "Es hombre"
# MAGIC * P(Experiencia) = P(Exp|Mujer) × P(Mujer) + P(Exp|Hombre) × P(Hombre)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Teorema de Bayes**
# MAGIC
# MAGIC Permite "invertir" probabilidades condicionales. Si conocemos P(B|A), podemos calcular P(A|B):
# MAGIC
# MAGIC $$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$
# MAGIC
# MAGIC **En nuestro problema:**
# MAGIC * Conocemos: P(Experiencia | Mujer) = 30%
# MAGIC * Queremos: P(Mujer | Experiencia) = ?
# MAGIC * Usamos Bayes para invertir la condicional
# MAGIC
# MAGIC **Componentes:**
# MAGIC * **P(A)**: probabilidad a priori (antes de observar B)
# MAGIC * **P(B|A)**: verosimilitud (probabilidad de observar B dado A)
# MAGIC * **P(B)**: probabilidad total de B (calculada con la ley anterior)
# MAGIC * **P(A|B)**: probabilidad a posteriori (después de observar B)

# COMMAND ----------

# DBTITLE 1,Solución Ejercicio A
# ============================================================================
# EJERCICIO A: Selección de Personal
# ============================================================================

# DATOS DEL PROBLEMA
total_solicitudes = 100
solicitudes_mujeres = 60
solicitudes_hombres = 40

# Tasas de experiencia relevante
tasa_exp_mujeres = 0.30  # 30% de las mujeres tienen experiencia
tasa_exp_hombres = 0.50  # 50% de los hombres tienen experiencia

# Calcular probabilidades
P_mujer = solicitudes_mujeres / total_solicitudes
P_hombre = solicitudes_hombres / total_solicitudes
P_exp_dado_mujer = tasa_exp_mujeres
P_exp_dado_hombre = tasa_exp_hombres

print("=== Ejercicio A: Selección de Personal ===")
print(f"\nDatos:")
print(f"  Total de solicitudes: {total_solicitudes}")
print(f"  Mujeres: {solicitudes_mujeres} ({P_mujer*100:.0f}%)")
print(f"  Hombres: {solicitudes_hombres} ({P_hombre*100:.0f}%)")
print(f"  P(Experiencia | Mujer) = {P_exp_dado_mujer}")
print(f"  P(Experiencia | Hombre) = {P_exp_dado_hombre}")

# ============================================================================
# PREGUNTA 1: P(Experiencia) - PROBABILIDAD TOTAL
# ============================================================================
# Aplicamos la ley de probabilidad total:
# P(Exp) = P(Exp|Mujer) × P(Mujer) + P(Exp|Hombre) × P(Hombre)

P_experiencia = (P_exp_dado_mujer * P_mujer) + (P_exp_dado_hombre * P_hombre)

print(f"\n1. ¿Cuál es la probabilidad de que tenga experiencia relevante?")
print(f"   P(Exp) = P(Exp|M) × P(M) + P(Exp|H) × P(H)")
print(f"   P(Exp) = {P_exp_dado_mujer} × {P_mujer} + {P_exp_dado_hombre} × {P_hombre}")
print(f"   P(Exp) = {P_exp_dado_mujer * P_mujer} + {P_exp_dado_hombre * P_hombre}")
print(f"   P(Exp) = {P_experiencia:.2f} ({P_experiencia*100:.0f}%)")

# Cálculo alternativo por conteo directo
mujeres_con_exp = int(solicitudes_mujeres * tasa_exp_mujeres)
hombres_con_exp = int(solicitudes_hombres * tasa_exp_hombres)
total_con_exp = mujeres_con_exp + hombres_con_exp

print(f"\n   Verificación por conteo:")
print(f"   - Mujeres con experiencia: {mujeres_con_exp}")
print(f"   - Hombres con experiencia: {hombres_con_exp}")
print(f"   - Total con experiencia: {total_con_exp}/{total_solicitudes} = {total_con_exp/total_solicitudes:.2f}")

# ============================================================================
# PREGUNTA 2: P(Mujer | Experiencia) - TEOREMA DE BAYES
# ============================================================================
# Queremos "invertir" la condicional: de P(Exp|Mujer) a P(Mujer|Exp)
# Fórmula de Bayes: P(Mujer|Exp) = P(Exp|Mujer) × P(Mujer) / P(Exp)

P_mujer_dado_exp = (P_exp_dado_mujer * P_mujer) / P_experiencia

print(f"\n2. Si una solicitud tiene experiencia, ¿cuál es la probabilidad de que sea de una mujer?")
print(f"   P(Mujer|Exp) = P(Exp|Mujer) × P(Mujer) / P(Exp)")
print(f"   P(Mujer|Exp) = {P_exp_dado_mujer} × {P_mujer} / {P_experiencia:.2f}")
print(f"   P(Mujer|Exp) = {P_mujer_dado_exp:.4f} ({P_mujer_dado_exp*100:.2f}%)")

print(f"\n   Verificación por conteo:")
print(f"   De {total_con_exp} personas con experiencia, {mujeres_con_exp} son mujeres")
print(f"   P(Mujer|Exp) = {mujeres_con_exp}/{total_con_exp} = {mujeres_con_exp/total_con_exp:.4f}")

print(f"\nℹ️ Interpretación:")
print(f"   Aunque las mujeres son {P_mujer*100:.0f}% de las solicitudes, solo representan")
print(f"   {P_mujer_dado_exp*100:.1f}% de quienes tienen experiencia, porque su tasa de")
print(f"   experiencia ({P_exp_dado_mujer*100:.0f}%) es menor que la de los hombres ({P_exp_dado_hombre*100:.0f}%).")

# COMMAND ----------

# DBTITLE 1,Visualización Ejercicio A
# Visualización del Ejercicio A
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Distribución de experiencia por género
categorias = ['Mujeres', 'Hombres']
con_experiencia = [mujeres_con_exp, hombres_con_exp]
sin_experiencia = [solicitudes_mujeres - mujeres_con_exp, solicitudes_hombres - hombres_con_exp]

x = np.arange(len(categorias))
width = 0.35

axes[0].bar(x - width/2, con_experiencia, width, label='Con Experiencia', color='#2ecc71', edgecolor='black')
axes[0].bar(x + width/2, sin_experiencia, width, label='Sin Experiencia', color='#e74c3c', edgecolor='black')
axes[0].set_xlabel('Género')
axes[0].set_ylabel('Número de Solicitudes')
axes[0].set_title('Distribución de Experiencia por Género')
axes[0].set_xticks(x)
axes[0].set_xticklabels(categorias)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Agregar valores sobre las barras
for i, (con, sin) in enumerate(zip(con_experiencia, sin_experiencia)):
    axes[0].text(i - width/2, con + 1, str(con), ha='center', fontweight='bold')
    axes[0].text(i + width/2, sin + 1, str(sin), ha='center', fontweight='bold')

# Gráfico 2: Composición de personas con experiencia
etiquetas = ['Mujeres\ncon exp', 'Hombres\ncon exp']
valores_exp = [mujeres_con_exp, hombres_con_exp]
colores = ['#ff9999', '#66b3ff']
explode = (0.05, 0.05)

axes[1].pie(valores_exp, labels=etiquetas, autopct='%1.1f%%', startangle=90, 
            colors=colores, explode=explode, shadow=True)
axes[1].set_title('Composición de Solicitudes con Experiencia')

plt.tight_layout()
plt.show()

print(f"\n📊 Observaciones clave:")
print(f"  - {P_exp_dado_mujer*100:.0f}% de las mujeres tienen experiencia vs {P_exp_dado_hombre*100:.0f}% de los hombres")
print(f"  - De todas las solicitudes con experiencia, {P_mujer_dado_exp*100:.1f}% son de mujeres")
print(f"  - La menor tasa de experiencia en mujeres reduce su representación en el pool con experiencia")

# COMMAND ----------

# DBTITLE 1,Ejercicio B: Control de Inventario
# MAGIC %md
# MAGIC ## Ejercicio B: Control de Inventario
# MAGIC
# MAGIC ### Contexto de Negocios
# MAGIC Una tienda recibe productos de tres proveedores con diferentes características:
# MAGIC
# MAGIC | Proveedor | % del Stock | Tasa de Defectos |
# MAGIC |-----------|-------------|------------------|
# MAGIC | A | 50% | 1% |
# MAGIC | B | 30% | 3% |
# MAGIC | C | 20% | 5% |
# MAGIC
# MAGIC ### Preguntas
# MAGIC
# MAGIC 1. **¿Cuál es la probabilidad de que un producto seleccionado al azar sea defectuoso?**
# MAGIC    - Aplicaremos la Ley de Probabilidad Total con 3 eventos
# MAGIC
# MAGIC 2. **Si un producto es defectuoso, ¿cuál es la probabilidad de que provenga del proveedor C?**
# MAGIC    - Usaremos el Teorema de Bayes para identificar el proveedor más problemático

# COMMAND ----------

# DBTITLE 1,Solución Ejercicio B
# ============================================================================
# EJERCICIO B: Control de Inventario
# ============================================================================

# DATOS DEL PROBLEMA
# Tres proveedores con diferentes proporciones de stock y tasas de defectos
P_A = 0.50  # Proveedor A: 50% del stock
P_B = 0.30  # Proveedor B: 30% del stock
P_C = 0.20  # Proveedor C: 20% del stock

P_def_dado_A = 0.01  # Proveedor A: 1% de defectos
P_def_dado_B = 0.03  # Proveedor B: 3% de defectos
P_def_dado_C = 0.05  # Proveedor C: 5% de defectos

print("=== Ejercicio B: Control de Inventario ===")
print(f"\nDatos de Proveedores:")
print(f"  Proveedor A: {P_A*100:.0f}% del stock, {P_def_dado_A*100:.0f}% defectos")
print(f"  Proveedor B: {P_B*100:.0f}% del stock, {P_def_dado_B*100:.0f}% defectos")
print(f"  Proveedor C: {P_C*100:.0f}% del stock, {P_def_dado_C*100:.0f}% defectos")

# ============================================================================
# PREGUNTA 1: P(Defectuoso) - PROBABILIDAD TOTAL
# ============================================================================
# Aplicamos la ley de probabilidad total considerando los 3 proveedores:
# P(Def) = P(Def|A) × P(A) + P(Def|B) × P(B) + P(Def|C) × P(C)

P_defectuoso = (P_def_dado_A * P_A) + (P_def_dado_B * P_B) + (P_def_dado_C * P_C)

print(f"\n1. ¿Cuál es la probabilidad de que un producto sea defectuoso?")
print(f"   P(Def) = P(Def|A) × P(A) + P(Def|B) × P(B) + P(Def|C) × P(C)")
print(f"   P(Def) = {P_def_dado_A} × {P_A} + {P_def_dado_B} × {P_B} + {P_def_dado_C} × {P_C}")
print(f"   P(Def) = {P_def_dado_A * P_A:.4f} + {P_def_dado_B * P_B:.4f} + {P_def_dado_C * P_C:.4f}")
print(f"   P(Def) = {P_defectuoso:.4f} ({P_defectuoso*100:.2f}%)")

# Desglose por proveedor
contrib_A = (P_def_dado_A * P_A) / P_defectuoso
contrib_B = (P_def_dado_B * P_B) / P_defectuoso
contrib_C = (P_def_dado_C * P_C) / P_defectuoso

print(f"\n   Contribución de cada proveedor a los defectos totales:")
print(f"   - Proveedor A: {contrib_A*100:.1f}%")
print(f"   - Proveedor B: {contrib_B*100:.1f}%")
print(f"   - Proveedor C: {contrib_C*100:.1f}%")

# ============================================================================
# PREGUNTA 2: P(Proveedor C | Defectuoso) - TEOREMA DE BAYES
# ============================================================================
# Queremos saber: si encontramos un defecto, ¿qué tan probable es que venga de C?
# Fórmula de Bayes: P(C|Def) = P(Def|C) × P(C) / P(Def)

P_C_dado_defectuoso = (P_def_dado_C * P_C) / P_defectuoso

print(f"\n2. Si un producto es defectuoso, ¿cuál es la probabilidad de que provenga del proveedor C?")
print(f"   P(C|Def) = P(Def|C) × P(C) / P(Def)")
print(f"   P(C|Def) = {P_def_dado_C} × {P_C} / {P_defectuoso:.4f}")
print(f"   P(C|Def) = {P_C_dado_defectuoso:.4f} ({P_C_dado_defectuoso*100:.2f}%)")

# Calcular para todos los proveedores
P_A_dado_defectuoso = (P_def_dado_A * P_A) / P_defectuoso
P_B_dado_defectuoso = (P_def_dado_B * P_B) / P_defectuoso

print(f"\n   Probabilidad de origen dado que es defectuoso:")
print(f"   - P(A|Def) = {P_A_dado_defectuoso:.4f} ({P_A_dado_defectuoso*100:.1f}%)")
print(f"   - P(B|Def) = {P_B_dado_defectuoso:.4f} ({P_B_dado_defectuoso*100:.1f}%)")
print(f"   - P(C|Def) = {P_C_dado_defectuoso:.4f} ({P_C_dado_defectuoso*100:.1f}%)")
print(f"   Total: {P_A_dado_defectuoso + P_B_dado_defectuoso + P_C_dado_defectuoso:.4f}")

print(f"\nℹ️ Interpretación:")
print(f"   Aunque el proveedor C solo suministra {P_C*100:.0f}% del stock, es responsable")
print(f"   de {P_C_dado_defectuoso*100:.1f}% de los productos defectuosos debido a su alta")
print(f"   tasa de defectos ({P_def_dado_C*100:.0f}%). Es el proveedor más problemático.")

# COMMAND ----------

# DBTITLE 1,Visualización Ejercicio B
# Visualización del Ejercicio B
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Gráfico 1: Proporción de stock por proveedor
proveedores = ['Proveedor A', 'Proveedor B', 'Proveedor C']
stock_prop = [P_A * 100, P_B * 100, P_C * 100]
colores_stock = ['#3498db', '#f39c12', '#e74c3c']

axes[0, 0].bar(proveedores, stock_prop, color=colores_stock, edgecolor='black')
axes[0, 0].set_ylabel('Porcentaje del Stock (%)')
axes[0, 0].set_title('Distribución del Stock por Proveedor')
axes[0, 0].grid(axis='y', alpha=0.3)

for i, v in enumerate(stock_prop):
    axes[0, 0].text(i, v + 1, f'{v:.0f}%', ha='center', fontweight='bold')

# Gráfico 2: Tasa de defectos por proveedor
tasas_defectos = [P_def_dado_A * 100, P_def_dado_B * 100, P_def_dado_C * 100]

axes[0, 1].bar(proveedores, tasas_defectos, color=colores_stock, edgecolor='black')
axes[0, 1].set_ylabel('Tasa de Defectos (%)')
axes[0, 1].set_title('Tasa de Defectos por Proveedor')
axes[0, 1].grid(axis='y', alpha=0.3)

for i, v in enumerate(tasas_defectos):
    axes[0, 1].text(i, v + 0.2, f'{v:.1f}%', ha='center', fontweight='bold')

# Gráfico 3: Contribución a defectos totales
contribuciones = [contrib_A * 100, contrib_B * 100, contrib_C * 100]

axes[1, 0].bar(proveedores, contribuciones, color=colores_stock, edgecolor='black')
axes[1, 0].set_ylabel('Contribución a Defectos Totales (%)')
axes[1, 0].set_title('Origen de Productos Defectuosos')
axes[1, 0].grid(axis='y', alpha=0.3)
axes[1, 0].axhline(y=P_C*100, color='red', linestyle='--', linewidth=1, label=f'% Stock C ({P_C*100:.0f}%)')
axes[1, 0].legend()

for i, v in enumerate(contribuciones):
    axes[1, 0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

# Gráfico 4: P(Proveedor | Defectuoso) - Teorema de Bayes
prob_origen = [P_A_dado_defectuoso * 100, P_B_dado_defectuoso * 100, P_C_dado_defectuoso * 100]

axes[1, 1].pie(prob_origen, labels=proveedores, autopct='%1.1f%%', startangle=90, 
               colors=colores_stock, explode=(0.05, 0.05, 0.1), shadow=True)
axes[1, 1].set_title('P(Proveedor | Defectuoso)\n(Teorema de Bayes)')

plt.tight_layout()
plt.show()

print(f"\n📊 Análisis de Riesgo por Proveedor:")
print(f"\nRiesgo = (% Stock) × (Tasa Defectos):")
for i, prov in enumerate(['A', 'B', 'C']):
    stock_pct = [P_A, P_B, P_C][i] * 100
    tasa = [P_def_dado_A, P_def_dado_B, P_def_dado_C][i] * 100
    riesgo = [contrib_A, contrib_B, contrib_C][i] * 100
    print(f"  Proveedor {prov}: {stock_pct:.0f}% × {tasa:.1f}% = {riesgo:.1f}% de defectos totales")

print(f"\n⚠️ Recomendación: Priorizar mejoras con Proveedor C, que causa {contrib_C*100:.1f}%")
print(f"   de los defectos pese a solo suministrar {P_C*100:.0f}% del stock.")

# COMMAND ----------

# DBTITLE 1,Ejercicio C: Análisis de Mercado
# MAGIC %md
# MAGIC ## Ejercicio C: Análisis de Mercado
# MAGIC
# MAGIC ### Contexto de Negocios
# MAGIC Una empresa analiza dos características de sus clientes:
# MAGIC * **40%** son clientes premium
# MAGIC * **25%** realizan compras online
# MAGIC * **12%** son premium Y compran online
# MAGIC
# MAGIC ### Preguntas
# MAGIC
# MAGIC 1. **¿Son independientes estas características?**
# MAGIC    - Verificaremos si P(A ∩ B) = P(A) × P(B)
# MAGIC
# MAGIC 2. **¿Cuál es la probabilidad de que un cliente sea premium O compre online?**
# MAGIC    - Aplicaremos la Regla de la Suma
# MAGIC
# MAGIC 3. **Si un cliente es premium, ¿cuál es la probabilidad de que compre online?**
# MAGIC    - Calcularemos probabilidad condicional

# COMMAND ----------

# DBTITLE 1,📚 Conceptos: Independencia y Regla de la Suma
# MAGIC %md
# MAGIC ### 📚 Conceptos Fundamentales
# MAGIC
# MAGIC **Independencia Estadística**
# MAGIC
# MAGIC Dos eventos A y B son **independientes** si el hecho de que ocurra uno NO afecta la probabilidad del otro.
# MAGIC
# MAGIC **Test de independencia:**
# MAGIC $$P(A \cap B) = P(A) \times P(B)$$
# MAGIC
# MAGIC Si se cumple esta igualdad → Son independientes  
# MAGIC Si NO se cumple → Son dependientes (hay relación entre ellos)
# MAGIC
# MAGIC **Verificación alternativa con condicionales:**
# MAGIC * Si independientes: P(B|A) = P(B)
# MAGIC * Si dependientes: P(B|A) ≠ P(B)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Regla de la Suma (Unión)**
# MAGIC
# MAGIC Para calcular la probabilidad de que ocurra A **O** B:
# MAGIC
# MAGIC $$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$
# MAGIC
# MAGIC 👉 Restamos la intersección para no contar dos veces los casos que cumplen ambas condiciones.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Probabilidad Condicional**
# MAGIC
# MAGIC La probabilidad de B dado que ocurrió A:
# MAGIC
# MAGIC $$P(B|A) = \frac{P(A \cap B)}{P(A)}$$

# COMMAND ----------

# DBTITLE 1,Solución Ejercicio C
# ============================================================================
# EJERCICIO C: Análisis de Mercado
# ============================================================================

# DATOS DEL PROBLEMA
P_premium = 0.40       # 40% son clientes premium
P_online = 0.25        # 25% realizan compras online
P_premium_y_online = 0.12  # 12% son premium Y compran online

print("=== Ejercicio C: Análisis de Mercado ===")
print(f"\nDatos:")
print(f"  P(Premium) = {P_premium} ({P_premium*100:.0f}%)")
print(f"  P(Online) = {P_online} ({P_online*100:.0f}%)")
print(f"  P(Premium ∩ Online) = {P_premium_y_online} ({P_premium_y_online*100:.0f}%)")

# ============================================================================
# PREGUNTA 1: ¿SON INDEPENDIENTES?
# ============================================================================
# Dos eventos son independientes si: P(A ∩ B) = P(A) × P(B)
# Si NO se cumple esta igualdad, los eventos son dependientes

P_premium_y_online_si_indep = P_premium * P_online

print(f"\n1. ¿Son independientes estas características?")
print(f"   Si fueran independientes: P(Premium ∩ Online) = P(Premium) × P(Online)")
print(f"   Esperado si indep: {P_premium} × {P_online} = {P_premium_y_online_si_indep:.2f}")
print(f"   Observado: {P_premium_y_online}")
print(f"   Diferencia: {abs(P_premium_y_online - P_premium_y_online_si_indep):.2f}")

if abs(P_premium_y_online - P_premium_y_online_si_indep) < 0.01:
    print(f"\n   ✅ Son INDEPENDIENTES")
else:
    print(f"\n   ❌ NO son independientes")
    
# Análisis adicional: probabilidades condicionales
P_online_dado_premium = P_premium_y_online / P_premium
P_online_dado_no_premium = (P_online - P_premium_y_online) / (1 - P_premium)

print(f"\n   Probabilidades condicionales:")
print(f"   P(Online | Premium) = {P_online_dado_premium:.4f} ({P_online_dado_premium*100:.1f}%)")
print(f"   P(Online | No Premium) = {P_online_dado_no_premium:.4f} ({P_online_dado_no_premium*100:.1f}%)")
print(f"   P(Online) = {P_online} ({P_online*100:.0f}%)")

if P_online_dado_premium > P_online:
    diferencia_rel = ((P_online_dado_premium - P_online) / P_online) * 100
    print(f"\n   📈 Los clientes Premium tienen {diferencia_rel:.1f}% más probabilidad de comprar online")
elif P_online_dado_premium < P_online:
    diferencia_rel = ((P_online - P_online_dado_premium) / P_online) * 100
    print(f"\n   📉 Los clientes Premium tienen {diferencia_rel:.1f}% menos probabilidad de comprar online")
else:
    print(f"\n   ➡️ No hay relación entre ser Premium y comprar online")

# ============================================================================
# PREGUNTA 2: P(Premium O Online) - REGLA DE LA SUMA
# ============================================================================
# Fórmula: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

P_premium_o_online = P_premium + P_online - P_premium_y_online

print(f"\n2. ¿Cuál es la probabilidad de que un cliente sea premium O compre online?")
print(f"   P(Premium ∪ Online) = P(Premium) + P(Online) - P(Premium ∩ Online)")
print(f"   P(Premium ∪ Online) = {P_premium} + {P_online} - {P_premium_y_online}")
print(f"   P(Premium ∪ Online) = {P_premium_o_online:.2f} ({P_premium_o_online*100:.0f}%)")

print(f"\n   Interpretación: {P_premium_o_online*100:.0f}% de los clientes tiene al menos")
print(f"   una de estas dos características (Premium o compras Online)")

# ============================================================================
# PREGUNTA 3: P(Online | Premium) - PROBABILIDAD CONDICIONAL
# ============================================================================
# Fórmula: P(B|A) = P(A ∩ B) / P(A)

print(f"\n3. Si un cliente es premium, ¿cuál es la probabilidad de que compre online?")
print(f"   P(Online | Premium) = P(Premium ∩ Online) / P(Premium)")
print(f"   P(Online | Premium) = {P_premium_y_online} / {P_premium}")
print(f"   P(Online | Premium) = {P_online_dado_premium:.2f} ({P_online_dado_premium*100:.0f}%)")

print(f"\n   Comparación:")
print(f"   - Probabilidad general de comprar online: {P_online*100:.0f}%")
print(f"   - Entre clientes Premium: {P_online_dado_premium*100:.0f}%")
print(f"   - Entre clientes No Premium: {P_online_dado_no_premium*100:.0f}%")

print(f"\nℹ️ Conclusión:")
print(f"   Los clientes Premium son más propensos a comprar online ({P_online_dado_premium*100:.0f}%)")
print(f"   que los clientes regulares ({P_online_dado_no_premium*100:.0f}%). Hay una relación positiva")
print(f"   entre ambas características, por lo que NO son independientes.")

# COMMAND ----------

# DBTITLE 1,Visualización Ejercicio C
# Visualización del Ejercicio C
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Simular 1000 clientes para crear un diagrama de Venn aproximado
n_clientes = 1000
n_premium = int(P_premium * n_clientes)
n_online = int(P_online * n_clientes)
n_ambos = int(P_premium_y_online * n_clientes)

# Calcular las regiones del diagrama de Venn
solo_premium = n_premium - n_ambos
solo_online = n_online - n_ambos
ambos = n_ambos
ninguno = n_clientes - solo_premium - solo_online - ambos

# Gráfico 1: Diagrama de Venn (aproximado con pie chart)
categorias_venn = ['Solo Premium', 'Premium y Online', 'Solo Online', 'Ninguno']
valores_venn = [solo_premium, ambos, solo_online, ninguno]
colores_venn = ['#ff9999', '#9999ff', '#99ff99', '#ffcc99']

axes[0, 0].pie(valores_venn, labels=categorias_venn, autopct='%1.1f%%', startangle=90, 
               colors=colores_venn, explode=(0.05, 0.1, 0.05, 0))
axes[0, 0].set_title('Distribución de Clientes\n(Diagrama de Venn)')

# Gráfico 2: Comparación de independencia
categorias_indep = ['Observado', 'Si fueran\nIndependientes']
valores_indep = [P_premium_y_online * 100, P_premium_y_online_si_indep * 100]
colores_indep = ['#e74c3c', '#95a5a6']

axes[0, 1].bar(categorias_indep, valores_indep, color=colores_indep, edgecolor='black')
axes[0, 1].set_ylabel('P(Premium ∩ Online) %')
axes[0, 1].set_title('Test de Independencia')
axes[0, 1].grid(axis='y', alpha=0.3)

for i, v in enumerate(valores_indep):
    axes[0, 1].text(i, v + 0.5, f'{v:.0f}%', ha='center', fontweight='bold')

# Agregar línea de diferencia
if abs(valores_indep[0] - valores_indep[1]) > 0.5:
    axes[0, 1].annotate('', xy=(1, valores_indep[0]), xytext=(1, valores_indep[1]),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    mid_y = (valores_indep[0] + valores_indep[1]) / 2
    axes[0, 1].text(1.15, mid_y, f'{abs(valores_indep[0] - valores_indep[1]):.0f}%\ndif', 
                   fontsize=9, color='red', fontweight='bold')

# Gráfico 3: Probabilidades condicionales
categorias_cond = ['P(Online|\nPremium)', 'P(Online|\nNo Premium)', 'P(Online)']
valores_cond = [P_online_dado_premium * 100, P_online_dado_no_premium * 100, P_online * 100]
colores_cond = ['#2ecc71', '#e67e22', '#3498db']

axes[1, 0].bar(categorias_cond, valores_cond, color=colores_cond, edgecolor='black')
axes[1, 0].set_ylabel('Probabilidad (%)')
axes[1, 0].set_title('Probabilidades Condicionales de Compra Online')
axes[1, 0].grid(axis='y', alpha=0.3)
axes[1, 0].axhline(y=P_online * 100, color='blue', linestyle='--', linewidth=1, alpha=0.5)

for i, v in enumerate(valores_cond):
    axes[1, 0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

# Gráfico 4: Tabla de contingencia como heatmap
data_contingencia = np.array([
    [ambos, solo_premium],
    [solo_online, ninguno]
])

im = axes[1, 1].imshow(data_contingencia, cmap='Blues', aspect='auto')
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_yticks([0, 1])
axes[1, 1].set_xticklabels(['Premium', 'No Premium'])
axes[1, 1].set_yticklabels(['Online', 'No Online'])
axes[1, 1].set_title('Tabla de Contingencia\n(n=1000 clientes simulados)')

# Agregar valores en cada celda
for i in range(2):
    for j in range(2):
        text = axes[1, 1].text(j, i, f'{data_contingencia[i, j]}\n({data_contingencia[i, j]/n_clientes*100:.1f}%)',
                              ha="center", va="center", color="black", fontsize=12, fontweight='bold')

plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.show()

print(f"\n📊 Resumen del Análisis:")
print(f"\n1. Independencia: NO - P(Premium ∩ Online) ≠ P(Premium) × P(Online)")
print(f"   Diferencia: {abs(P_premium_y_online - P_premium_y_online_si_indep)*100:.0f} puntos porcentuales")
print(f"\n2. Unión: {P_premium_o_online*100:.0f}% de clientes son Premium O compran Online")
print(f"\n3. Condicional: {P_online_dado_premium*100:.0f}% de Premium compran online vs {P_online*100:.0f}% general")
print(f"\n🎯 Insight: Los clientes Premium adoptan más el canal online, sugiriendo")
print(f"   que una estrategia digital podría atraer o retener clientes de alto valor.")

# COMMAND ----------

