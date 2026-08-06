# Estadística para el Análisis de Negocios

## 📚 Descripción del Curso

Material didáctico completo de Estadística aplicada al análisis de negocios, con enfoque práctico en Python y aplicaciones reales a problemas de administración de empresas, economía, contabilidad y finanzas.

## ✅ Estado del Material: 100% Completo

* **8 notebooks** con contenido completo (4 teóricos + 4 prácticos)
* **42 secciones teóricas** cubriendo todos los conceptos
* **30+ ejercicios prácticos** con Python aplicados a negocios
* **50+ visualizaciones** con matplotlib y seaborn
* **~3,500 líneas de código** ejecutable y comentado

---

## 📖 Contenido por Unidad

### 📗 Unidad I: Estadística Descriptiva (Probabilidad Básica)

**Teoría** - [01_Teoria_Probabilidad_Basica](Unidad_I_Estadistica_Descriptiva/01_Teoria_Probabilidad_Basica)
* Experimentos y sucesos aleatorios
* Probabilidad clásica, frecuencial y subjetiva
* Operaciones con sucesos (unión, intersección, complemento)
* Reglas de suma y producto
* Probabilidad condicional
* Independencia de sucesos
* Teorema de Bayes
* Árboles de decisión
* Aplicaciones en finanzas y seguros

**Práctica** - [02_Practica_Probabilidad_Basica](Unidad_I_Estadistica_Descriptiva/02_Practica_Probabilidad_Basica)
* 17 ejercicios interactivos con Python
* Cálculos de probabilidad en contextos empresariales
* Verificación empírica con simulaciones
* Análisis de riesgo financiero
* Probabilidad condicional en ventas
* Teorema de Bayes en diagnóstico empresarial

---

### 📘 Unidad II: Distribuciones de Probabilidad

**Teoría** - [01_Teoria_Distribuciones](Unidad_II_Distribuciones_Probabilidad/01_Teoria_Distribuciones)
* Variables aleatorias discretas y continuas
* Funciones de probabilidad y densidad
* Esperanza, varianza y momentos
* Distribución Bernoulli
* Distribución Binomial
* Distribución de Poisson
* Distribución Normal
* Estandarización (Z-scores)
* Teorema Central del Límite
* Uso de `scipy.stats` en Python

**Práctica** - [02_Practica_Distribuciones](Unidad_II_Distribuciones_Probabilidad/02_Practica_Distribuciones)

**Ejercicios aplicados a negocios:**
1. **Bernoulli**: Análisis de conversión de clientes en e-commerce
2. **Binomial**: Tasa de conversión de campañas de email marketing
3. **Poisson**: Gestión de capacidad en call center
4. **Normal**: Análisis de ventas diarias con regla empírica
5. **Estandarización**: Comparación de tiempos de entrega entre servicios
6. **TCL**: Distribución de salarios y aplicación en recursos humanos

---

### 📙 Unidad III: Muestreo, Estimación e Intervalos de Confianza

**Teoría** - [01_Teoria_Muestreo_Intervalos](Unidad_III_Muestreo_Estimacion/01_Teoria_Muestreo_Intervalos)
* Población vs Muestra
* Tipos de muestreo (aleatorio, estratificado, sistemático, conglomerados)
* Distribución muestral de la media
* Teorema Central del Límite (detallado)
* Estimadores puntuales
* Intervalos de confianza para la media (σ conocida y desconocida)
* Intervalos de confianza para proporciones
* Intervalos para diferencia de medias
* Detección de outliers (IQR, Z-score)
* Determinación del tamaño de muestra

**Práctica** - [02_Practica_Muestreo_Intervalos](Unidad_III_Muestreo_Estimacion/02_Practica_Muestreo_Intervalos)

**Ejercicios aplicados a negocios:**
1. **TCL**: Simulación con distribución de salarios (asimétrica → normal)
2. **IC para media**: Análisis de gasto promedio de clientes con diferentes niveles de confianza
3. **IC para proporción**: Satisfacción de clientes y determinación de tamaño de muestra
4. **Detección de outliers**: Identificación de transacciones atípicas con métodos IQR y Z-score

---

### 📕 Unidad IV: Pruebas de Hipótesis, Regresión y ANOVA

**Teoría** - [01_Teoria_Pruebas_Hipotesis](Unidad_IV_Prueba_Hipotesis/01_Teoria_Pruebas_Hipotesis)
* Hipótesis nula y alternativa
* Errores Tipo I y Tipo II
* Nivel de significancia y potencia
* Pasos de una prueba de hipótesis
* Método del valor crítico
* Método del p-valor
* Pruebas Z y t para la media
* Pruebas para proporciones
* Prueba t para diferencia de medias
* Regresión lineal simple
* Coeficiente de determinación R²
* Correlación de Pearson
* ANOVA de un factor
* Supuestos y verificación

**Práctica** - [02_Practica_Pruebas_Regresion_ANOVA](Unidad_IV_Prueba_Hipotesis/02_Practica_Pruebas_Regresion_ANOVA)

**Ejercicios aplicados a negocios:**
1. **Prueba t**: Validación de promesa de tiempos de entrega de una empresa de logística
2. **Regresión lineal**: Análisis de ROI en publicidad (gasto en marketing vs ventas)
3. **ANOVA**: Comparación de efectividad de tres estrategias de marketing diferentes

---

## 🛠️ Tecnologías y Bibliotecas

### Python Core
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

### Estadística
```python
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
```

### Funciones Principales de scipy.stats

**Distribuciones:**
* `stats.bernoulli(p)` - Distribución Bernoulli
* `stats.binom(n, p)` - Distribución Binomial
* `stats.poisson(mu)` - Distribución de Poisson
* `stats.norm(loc, scale)` - Distribución Normal
* `stats.t(df, loc, scale)` - Distribución t de Student
* `stats.f(dfn, dfd)` - Distribución F

**Métodos comunes:**
* `.pmf(k)` - Función de probabilidad (discretas)
* `.pdf(x)` - Función de densidad (continuas)
* `.cdf(x)` - Función de distribución acumulada
* `.ppf(q)` - Percentil (inversa de CDF)
* `.rvs(size)` - Generar muestras aleatorias
* `.mean()`, `.var()`, `.std()` - Parámetros

**Pruebas de hipótesis:**
* `stats.ttest_1samp()` - Prueba t de una muestra
* `stats.ttest_ind()` - Prueba t de dos muestras independientes
* `stats.pearsonr()` - Correlación de Pearson
* `stats.linregress()` - Regresión lineal simple
* `stats.f_oneway()` - ANOVA de un factor

---

## 💼 Aplicaciones en Análisis de Negocios

### Marketing y Ventas
* **Tasas de conversión** (Distribución Binomial)
* **A/B testing** (Pruebas t de dos muestras)
* **ROI de publicidad** (Regresión lineal)
* **Segmentación de clientes** (ANOVA)
* **Análisis de campañas** (Intervalos de confianza)
* **Predicción de ventas** (Regresión)

### Finanzas y Contabilidad
* **Análisis de riesgo** (Intervalos de confianza, desviación estándar)
* **Proyecciones financieras** (Regresión)
* **Detección de fraude** (Outliers, Z-scores)
* **Comparación de inversiones** (ANOVA, pruebas t)
* **Valuación de activos** (Distribución Normal)
* **Control presupuestario** (Pruebas de hipótesis)

### Operaciones y Logística
* **Tiempos de entrega** (Distribución Normal)
* **Control de calidad** (Pruebas de hipótesis)
* **Capacidad de servicio** (Distribución de Poisson)
* **Optimización de recursos** (ANOVA)
* **Gestión de inventarios** (Distribuciones de probabilidad)
* **Eficiencia operacional** (Intervalos de confianza)

### Recursos Humanos
* **Análisis de salarios** (Distribución Normal, TCL)
* **Evaluación de desempeño** (Pruebas estadísticas)
* **Rotación de personal** (Proporciones, Chi-cuadrado)
* **Capacitación** (ANOVA para comparar métodos)

---

## 📊 Estructura de Archivos

```
Estadistica/
│
├── README.md                                           ✅ Este archivo
│
├── Unidad_I_Estadistica_Descriptiva/
│   ├── 01_Teoria_Probabilidad_Basica                   ✅ [9 secciones]
│   └── 02_Practica_Probabilidad_Basica                 ✅ [17 ejercicios]
│
├── Unidad_II_Distribuciones_Probabilidad/
│   ├── 01_Teoria_Distribuciones                        ✅ [11 secciones]
│   └── 02_Practica_Distribuciones                      ✅ [6 ejercicios]
│
├── Unidad_III_Muestreo_Estimacion/
│   ├── 01_Teoria_Muestreo_Intervalos                   ✅ [11 secciones]
│   └── 02_Practica_Muestreo_Intervalos                 ✅ [4 ejercicios]
│
├── Unidad_IV_Prueba_Hipotesis/
│   ├── 01_Teoria_Pruebas_Hipotesis                     ✅ [11 secciones]
│   └── 02_Practica_Pruebas_Regresion_ANOVA             ✅ [3 ejercicios]
│
├── GUIA_RAPIDA_UNIDADES_III_IV.md                      ✅ Referencia rápida
└── RESUMEN_FINAL.md                                    ✅ Resumen ejecutivo
```

---

## 🎯 Características del Material

### ✨ Enfoque Práctico
* **Problemas reales de negocios** en cada ejercicio
* **Código Python ejecutable** y comentado paso a paso
* **50+ visualizaciones** con matplotlib para facilitar comprensión
* **Verificación empírica** de propiedades teóricas con simulaciones
* **Interpretación** de resultados en contexto empresarial

### 📈 Progresión Pedagógica
* **De lo simple a lo complejo**: ejercicios graduales
* **Teoría + Práctica**: cada unidad tiene notebook teórico y práctico
* **Verificación experimental**: simulaciones Monte Carlo
* **Casos de estudio**: aplicaciones a industrias reales
* **Código reproducible**: todo el material es ejecutable

### 🔍 Cobertura Completa
* **Probabilidad básica**: fundamentos matemáticos
* **Distribuciones**: discretas y continuas
* **Inferencia estadística**: estimación e intervalos
* **Pruebas de hipótesis**: validación de afirmaciones
* **Modelado**: regresión y ANOVA

---

## 🚀 Cómo Usar Este Material

### Para Estudiantes
1. **Estudiar la teoría** en los notebooks teóricos de cada unidad
2. **Ejecutar los ejemplos** paso a paso en los notebooks prácticos
3. **Modificar los parámetros** para experimentar
4. **Aplicar a datos propios** de su organización o proyecto
5. **Consultar las guías rápidas** para referencia

### Para Instructores
1. **Seguir la secuencia** de las 4 unidades
2. **Adaptar los ejercicios** a su industria específica
3. **Ampliar con casos de estudio** adicionales
4. **Usar las visualizaciones** como material de clase
5. **Asignar ejercicios** de los notebooks prácticos como tarea

### Para Profesionales
1. **Revisar conceptos específicos** según necesidad
2. **Usar como referencia** de scipy.stats
3. **Adaptar código** a problemas reales
4. **Consultar la guía rápida** para recordar fórmulas
5. **Explorar aplicaciones** en su área de negocio

---

## 📐 Fórmulas Principales

### Probabilidad
```
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
P(A|B) = P(A ∩ B) / P(B)
Independencia: P(A ∩ B) = P(A) × P(B)
Bayes: P(A|B) = P(B|A) × P(A) / P(B)
```

### Distribuciones
```
Binomial:  P(X=k) = C(n,k) × p^k × (1-p)^(n-k)
Poisson:   P(X=k) = (e^(-λ) × λ^k) / k!
Normal:    Z = (X - μ) / σ
```

### Intervalos de Confianza
```
IC para μ (σ conocida):     x̄ ± z_(α/2) × (σ/√n)
IC para μ (σ desconocida):  x̄ ± t_(α/2,n-1) × (s/√n)
IC para p:                  p̂ ± z_(α/2) × √[p̂(1-p̂)/n]
```

### Pruebas de Hipótesis
```
Estadístico Z:  (x̄ - μ₀) / (σ/√n)
Estadístico t:  (x̄ - μ₀) / (s/√n)
Regresión:      β̂₁ = Σ[(xᵢ-x̄)(yᵢ-ȳ)] / Σ(xᵢ-x̄)²
R²:             1 - SSE/SST
ANOVA:          F = MSB/MSW
```

---

## 📚 Recursos Adicionales

### Guías de Referencia
* **[GUIA_RAPIDA_UNIDADES_III_IV.md](GUIA_RAPIDA_UNIDADES_III_IV.md)**: Fórmulas, conceptos y procedimientos
* **[RESUMEN_FINAL.md](RESUMEN_FINAL.md)**: Estado completo del material y estadísticas

### Temas Cubiertos
* ✅ Probabilidad clásica, frecuencial y subjetiva
* ✅ Teorema de Bayes y aplicaciones
* ✅ Distribuciones discretas (Bernoulli, Binomial, Poisson)
* ✅ Distribuciones continuas (Normal, t, F)
* ✅ Teorema Central del Límite
* ✅ Intervalos de confianza
* ✅ Pruebas de hipótesis (Z, t, proporciones)
* ✅ Regresión lineal simple
* ✅ Correlación
* ✅ ANOVA
* ✅ Detección de outliers

---

## 🎓 Objetivos de Aprendizaje

Al completar este curso, los estudiantes serán capaces de:

### Conocimientos
* ✅ Comprender los fundamentos de probabilidad y estadística
* ✅ Conocer las distribuciones de probabilidad más importantes
* ✅ Entender el concepto de inferencia estadística
* ✅ Conocer las pruebas de hipótesis principales
* ✅ Comprender regresión lineal y ANOVA

### Habilidades
* ✅ Calcular probabilidades en contextos empresariales
* ✅ Construir intervalos de confianza
* ✅ Realizar pruebas de hipótesis
* ✅ Ajustar modelos de regresión
* ✅ Interpretar resultados estadísticos
* ✅ Usar Python (numpy, scipy, matplotlib) para análisis
* ✅ Detectar datos atípicos
* ✅ Visualizar resultados estadísticos

### Competencias
* ✅ Aplicar estadística a problemas reales de negocios
* ✅ Tomar decisiones basadas en evidencia estadística
* ✅ Comunicar resultados estadísticos a audiencias no técnicas
* ✅ Evaluar afirmaciones usando datos
* ✅ Diseñar experimentos y análisis

---

## 📊 Estadísticas del Material

| Métrica | Valor |
|---------|-------|
| Notebooks completos | 8 de 8 (100%) |
| Secciones teóricas | 42 secciones |
| Ejercicios prácticos | 30+ ejercicios |
| Celdas de código Python | 55+ celdas |
| Visualizaciones | 50+ gráficos |
| Líneas de código | ~3,500 líneas |
| Guías de referencia | 2 archivos .md |
| Horas de contenido | 40-60 horas |

---

## 💡 Casos de Uso por Industria

### Retail y E-commerce
* Análisis de tasas de conversión
* Predicción de ventas
* Segmentación de clientes
* A/B testing de páginas web
* Detección de fraude en transacciones

### Banca y Finanzas
* Análisis de riesgo crediticio
* Modelado de carteras
* Detección de transacciones sospechosas
* Proyecciones de flujos de caja
* Valuación de instrumentos financieros

### Manufactura
* Control de calidad estadístico
* Optimización de procesos
* Predicción de demanda
* Análisis de tiempos de producción
* Gestión de inventarios

### Servicios
* Satisfacción de clientes
* Gestión de tiempos de espera
* Optimización de recursos humanos
* Análisis de SLAs
* Predicción de rotación de clientes

### Salud
* Análisis de tiempos de atención
* Eficiencia de tratamientos
* Gestión de capacidad hospitalaria
* Análisis de costos
* Estudios epidemiológicos

---

## ⚙️ Requisitos

### Software
* **Python 3.8+**
* **Databricks** (o Jupyter Notebook local)
* **Bibliotecas**: numpy, pandas, matplotlib, seaborn, scipy, statsmodels

### Conocimientos Previos
* **Matemáticas básicas**: álgebra, funciones
* **Python básico**: variables, listas, loops, funciones (deseable pero no esencial)
* **Ningún conocimiento previo de estadística** es requerido

### Instalación Local (opcional)
```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels
```

---

## 🤝 Contribuciones y Mejoras

Este material está diseñado para ser extensible y adaptable:

### Posibles Extensiones
* Agregar más ejercicios prácticos
* Incluir casos de estudio de industrias específicas
* Ampliar con distribuciones adicionales (Chi-cuadrado, Beta, Gamma)
* Agregar pruebas no paramétricas
* Incluir análisis de series temporales
* Agregar machine learning básico
* Crear notebooks de evaluación/examen

### Áreas de Especialización
* Estadística para finanzas cuantitativas
* Análisis de datos de marketing
* Control estadístico de calidad
* Econometría básica
* Análisis de datos de encuestas

---

## 📝 Notas Importantes

### Sobre los Datos
* Los **datos son simulados** con fines pedagógicos
* Los **parámetros son realistas** basados en industrias reales
* Los **resultados son reproducibles** (semillas fijadas con `np.random.seed()`)

### Sobre el Código
* Todo el código es **ejecutable directamente**
* Las **visualizaciones** se generan automáticamente
* Los **comentarios** explican cada paso
* Se usan **convenciones estándar** de Python y estadística

### Sobre la Pedagogía
* **Aprender haciendo**: los notebooks son interactivos
* **Verificación empírica**: simular para comprender
* **Contexto empresarial**: todos los ejemplos son de negocios
* **Progresión natural**: de fundamentos a aplicaciones

---

## 📞 Información del Curso

**Curso**: Estadística para el Análisis de Negocios  
**Nivel**: Universitario / Postgrado  
**Modalidad**: Presencial / Virtual  
**Duración**: 40-60 horas  
**Institución**: Universidad del Atlántico  
**Año**: 2026

---

## ✅ Estado del Proyecto

**Versión**: 1.0  
**Última actualización**: Agosto 2026  
**Estado**: ✅ **COMPLETO Y LISTO PARA ENSEÑAR**

### Changelog
* **v1.0** (Agosto 2026): Material completo con las 4 unidades, teoría y práctica
  * 8 notebooks con contenido detallado
  * 42 secciones teóricas
  * 30+ ejercicios prácticos enfocados en negocios
  * 50+ visualizaciones
  * Guías de referencia rápida
  * README completo

---

## 📄 Licencia

Este material es de uso académico para el curso de Estadística para el Análisis de Negocios.

---

**¡Bienvenido al curso de Estadística para el Análisis de Negocios!** 📊📈

Para comenzar, abre el notebook [Unidad I - Teoría](Unidad_I_Estadistica_Descriptiva/01_Teoria_Probabilidad_Basica) y empieza tu viaje en el mundo de la estadística aplicada a negocios.
