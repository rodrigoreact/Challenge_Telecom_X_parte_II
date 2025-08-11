# Paso 1: Importar librerías necesarias
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score


# URL del dataset en formato JSON
url = "https://raw.githubusercontent.com/ingridcristh/challenge2-data-science-LATAM/refs/heads/main/TelecomX_Data.json"

# Descarga del JSON
response = requests.get(url)

if response.status_code == 200:
    data_json = response.json()
    df = pd.json_normalize(data_json)
    print("Datos cargados correctamente.")
else:
    print(f"Error al acceder a la URL: {response.status_code}")
    df = pd.DataFrame()

print(df.head())
# Eliminar columnas no útiles antes del preprocesamiento:
# 1. Columnas no deseadas (por ejemplo 'customerID')
cols_a_eliminar = ['customerID']
df = df.drop(columns=cols_a_eliminar)
# Mostrar número de valores únicos por columna
print(df.nunique().sort_values())

# Mostrar porcentaje de valores nulos por columna
print(df.isnull().mean().sort_values(ascending=False))

# 2. Guardar el DataFrame limpio en un nuevo archivo CSV
df.to_csv("TelecomX_Data_limpio.csv", index=False)

# 3. Confirmar que la variable df es la que usamos en adelante
print("DataFrame limpio cargado, primeras filas:")
print(df.head())
print(f"Número de columnas en DataFrame limpio: {df.shape[1]}")
print("Columnas actuales:")
print(df.columns.tolist())

# Paso 2 – Limpieza de la variable objetivo
target = "Churn"

# Asegurar que no haya espacios y convertir valores inesperados a NaN
df[target] = df[target].astype(str).str.strip()
df.loc[~df[target].isin(["Yes", "No"]), target] = np.nan

# Eliminar filas con NaN en Churn
df = df.dropna(subset=[target])

# Mapear Yes/No a 1/0
df[target] = df[target].map({"Yes": 1, "No": 0})

# Paso 3 – Separar variables predictoras y objetivo
X = df.drop(columns=[target])
y = df[target]

# Paso 4 – Codificación de variables categóricas
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Paso 5 – Escalado de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Paso 6 – División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Paso 7 – Entrenar varios modelos
modelos = {
    "Regresión Lineal": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

resultados = []
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    resultados.append((nombre, rmse, r2))

# Paso 8 – Mostrar resultados
df_resultados = pd.DataFrame(resultados, columns=["Modelo", "RMSE", "R²"])
print(df_resultados)

# Mejor modelo
mejor_modelo = df_resultados.loc[df_resultados["R²"].idxmax(), "Modelo"]
print(f"El mejor modelo es: {mejor_modelo}")

# Mejor RMSE
mejor_rmse = df_resultados.loc[df_resultados["RMSE"].idxmin(), "RMSE"]
print(f"El mejor RMSE es: {mejor_rmse}")

#               Modelo      RMSE        R²
# 0   Regresión Lineal  0.378699  0.263211
# 1      Random Forest  0.380043  0.257976
# 2  Gradient Boosting  0.370593  0.294418
# El mejor modelo es: Gradient Boosting
# El mejor RMSE es: 0.3705928378819955
# Veo que eso valores son realmente pésimos.
# ChatGPT: Claro, esos resultados no son sorprendentes porque estás usando modelos de regresión
#  (que predicen valores continuos) para un problema 
# claramente de clasificación binaria (Churn: 0 o 1).

# ¿Por qué?
# El RMSE y R² son métricas para regresión, no para clasificación.
# Tu variable objetivo es categórica binaria (clientes que se van o no).

# Pero si quieres mejorar estos resultados, puedes intentar usar modelos
#  de clasificación (que predicen 0 o 1) para un problema de regresión.
# Qué hacer para mejorar: usar modelos de clasificación y métricas adecuadas
# Te recomiendo usar:

# Modelos de clasificación:
                    # LogisticRegression
                    # RandomForestClassifier
                    # GradientBoostingClassifier
# Métricas para clasificación:
                    # Accuracy
                    # Precision
                    # Recall
                    # F1-score
                    # ROC AUC

modelos_clasif = {
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

resultados_clasif = []

for nombre, modelo in modelos_clasif.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]  # Probabilidad clase positiva
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    resultados_clasif.append((nombre, acc, prec, rec, f1, roc_auc))

df_resultados_clasif = pd.DataFrame(
    resultados_clasif, 
    columns=["Modelo", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
)

print(df_resultados_clasif)
# Salida:
#                 Modelo  Accuracy  Precision    Recall        F1   ROC AUC
# 0  Regresión Logística  0.809084   0.676871  0.533512  0.596702  0.840400
# 1        Random Forest  0.798439   0.665428  0.479893  0.557632  0.831901
# 2    Gradient Boosting  0.802697   0.677903  0.485255  0.565625  0.840450

# ¡Genial! Esos resultados son muchísimo más acordes a un problema de clasificación binaria y bastante razonables para un dataset real de churn.

# Te resumo qué indican esas métricas para que las interpretes mejor:
# Accuracy (~80%): 8 de cada 10 clientes se clasifican correctamente.
# Precision (~67%): De todos los que el modelo predice que van a abandonar, el 67% realmente lo hacen.
# Recall (~48-53%): De todos los clientes que efectivamente abandonan, el modelo detecta aproximadamente la mitad.
# F1 (~0.56-0.60): Balance entre precisión y recall, está OK pero puede mejorar.
# ROC AUC (~0.83-0.84): Buen poder discriminativo para separar clases.

# Próximos pasos para mejorar resultados
# Balanceo de clases: si hay mucha diferencia entre cantidad de churners y no churners, usa técnicas tipo SMOTE o submuestreo.
# Feature engineering: crear nuevas variables que capturen mejor el comportamiento.
# Optimización de hiperparámetros con GridSearchCV o RandomizedSearchCV.
# Modelos más complejos: como XGBoost, LightGBM o redes neuronales.
# Evaluar con curvas ROC y Precision-Recall para análisis visual.

# ¿Qué es SMOTE?
# SMOTE (Synthetic Minority Over-sampling Technique) genera muestras sintéticas de la clase minoritaria para
# balancear el dataset de entrenamiento.
# Cómo integrarlo en tu código:
# Instalamos la librería imblearn:
# pip install imblearn
# Luego, importamos SMOTE:
# from imblearn.over_sampling import SMOTE
# Aplicamos SMOTE a tus datos:
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# Después del train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Aplicar SMOTE solo al conjunto de entrenamiento
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Antes de SMOTE: {np.bincount(y_train)}")
print(f"Después de SMOTE: {np.bincount(y_train_smote)}")

# Entrenar modelos con los datos balanceados
modelos_clasif = {
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

resultados_clasif = []

for nombre, modelo in modelos_clasif.items():
    modelo.fit(X_train_smote, y_train_smote)
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]  # Probabilidad clase positiva
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    resultados_clasif.append((nombre, acc, prec, rec, f1, roc_auc))

df_resultados_clasif = pd.DataFrame(
    resultados_clasif, 
    columns=["Modelo", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
)

print(df_resultados_clasif)

#                 Modelo  Accuracy  Precision    Recall        F1   ROC AUC
# 0  Regresión Logística  0.809084   0.676871  0.533512  0.596702  0.840400
# 1        Random Forest  0.798439   0.665428  0.479893  0.557632  0.831901
# 2    Gradient Boosting  0.802697   0.677903  0.485255  0.565625  0.840450
# Antes de SMOTE: [4139 1495]
#--------------------------------------------------------------------------
# Después de SMOTE: [4139 4139]
#                 Modelo  Accuracy  Precision    Recall        F1   ROC AUC
# 0  Regresión Logística  0.749468   0.518584  0.783422  0.624068  0.839213
# 1        Random Forest  0.784954   0.600000  0.569519  0.584362  0.827569
# 2    Gradient Boosting  0.781405   0.572052  0.700535  0.629808  0.844549
# ¿Qué se puede concluir?
# SMOTE mejora el desequilibrio de clases en el conjunto de entrenamiento,
# lo que permite que el modelo se desempeñe mejor en la clasificación de clientes que abandonan.

# Pero bajó un poco el desempeño en la clasificación de clientes que no abandonan, bajó Accuracy, Precision.



# | Modelo              | Accuracy (antes) | Precision (antes) | Recall (antes) | F1 (antes) | ROC AUC (antes) | Accuracy (desp.) | Precision (desp.) | Recall (desp.) | F1 (desp.) | ROC AUC (desp.) |
# | ------------------- | ---------------- | ----------------- | -------------- | ---------- | --------------- | ---------------- | ----------------- | -------------- | ---------- | --------------- |
# | Regresión Logística | 0.809            | 0.677             | 0.534          | 0.597      | 0.840           | 0.749            | 0.519             | **0.783**      | 0.624      | 0.839           |
# | Random Forest       | 0.798            | 0.665             | 0.480          | 0.558      | 0.832           | **0.785**        | 0.600             | 0.570          | 0.584      | 0.828           |
# | Gradient Boosting   | 0.803            | 0.678             | 0.485          | 0.566      | 0.840           | 0.781            | 0.572             | 0.701          | **0.630**  | **0.845**       |

# y qué sucedió con F1?
# El F1 con SMOTE mejoró o al menos se mantuvo similar, dependiendo del modelo:

# Regresión Logística:
# Antes: 0.597
# Después SMOTE: 0.624 → Mejoró
# Random Forest:
# Antes: 0.558
# Después SMOTE: 0.584 → Mejoró
# Gradient Boosting:
# Antes: 0.566
# Después SMOTE: 0.630 → Mejoró

# El F1 es la media armónica entre precisión y recall, así que al subir mucho el recall y bajar algo la precisión, el balance general (F1) suele mejorar cuando SMOTE ayuda a detectar mejor la clase minoritaria.

# En resumen: el F1 mejoró en los tres modelos al aplicar SMOTE.Ok.Bien.


# Próximos pasos para mejorar resultados
# Balanceo de clases: si hay mucha diferencia entre cantidad de churners y no churners, usa técnicas tipo SMOTE o submuestreo.
# Feature engineering: crear nuevas variables que capturen mejor el comportamiento.
# Optimización de hiperparámetros con GridSearchCV o RandomizedSearchCV.
# Modelos más complejos: como XGBoost, LightGBM o redes neuronales.
# Evaluar con curvas ROC y Precision-Recall para análisis visual.

# ¿Qué es SMOTE?
# SMOTE (Synthetic Minority Over-sampling Technique) genera muestras sintéticas de la clase minoritaria para
# balancear el dataset de entrenamiento.
# Cómo funciona:
# SMOTE copia aleatoriamente las muestras de la clase minoritaria y las combina con sus vecinas para crear muestras sintéticas.
# Esto ayuda a mejorar el desequilibrio de clases en el conjunto de entrenamiento, lo que permite que el modelo se
# desempeñe mejor en la clasificación de clientes que abandonan.

# Definimos las técnicas de balanceo a probar

# Apliquemos SMOTE variantes: SMOTEENN, SMOTETomek para limpiar ruido.

balanceadores = {
    "SMOTE": SMOTE(random_state=42),
    "SMOTEENN": SMOTEENN(random_state=42),
    "SMOTETomek": SMOTETomek(random_state=42)
}

modelos_clasif = {
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

resultados = []

for bal_name, balancer in balanceadores.items():
    print(f"\nAplicando {bal_name}...")
    X_train_bal, y_train_bal = balancer.fit_resample(X_train, y_train)
    print(f"Distribución clases antes: {np.bincount(y_train)}")
    print(f"Distribución clases después: {np.bincount(y_train_bal)}")
    
    for model_name, modelo in modelos_clasif.items():
        modelo.fit(X_train_bal, y_train_bal)
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        resultados.append((
            bal_name, model_name, acc, prec, rec, f1, roc_auc
        ))

# Mostrar resultados en tabla
df_resultados = pd.DataFrame(resultados, columns=[
    "Balanceo", "Modelo", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"
])

print(df_resultados)

# Vamos a analizar tus resultados y qué conclusiones se pueden sacar de cada técnica de balanceo:

# | Balanceo   | Modelo              | Accuracy | Precision | Recall | F1    | ROC AUC |
# | ---------- | ------------------- | -------- | --------- | ------ | ----- | ------- |
# | SMOTE      | Regresión Logística | 0.749    | 0.519     | 0.783  | 0.624 | 0.839   |
# | SMOTE      | Random Forest       | 0.785    | 0.600     | 0.570  | 0.584 | 0.828   |
# | SMOTE      | Gradient Boosting   | 0.781    | 0.572     | 0.701  | 0.630 | 0.845   |
# | SMOTEENN   | Regresión Logística | 0.685    | 0.451     | 0.866  | 0.593 | 0.839   |
# | SMOTEENN   | Random Forest       | 0.732    | 0.497     | 0.805  | 0.614 | 0.827   |
# | SMOTEENN   | Gradient Boosting   | 0.720    | 0.484     | 0.837  | 0.613 | 0.841   |
# | SMOTETomek | Regresión Logística | 0.749    | 0.519     | 0.783  | 0.624 | 0.839   |
# | SMOTETomek | Random Forest       | 0.781    | 0.588     | 0.580  | 0.584 | 0.834   |
# | SMOTETomek | Gradient Boosting   | 0.788    | 0.584     | 0.701  | 0.637 | 0.845   |

# Conclusiones clave:
# SMOTE:

# Buen balance entre precisión y recall.
# Mejor accuracy en Random Forest.
# Gradient Boosting tiene el mejor F1 y ROC AUC entre los tres modelos con SMOTE.
# SMOTEENN:
# Reduce accuracy, precision y F1 en comparación con SMOTE.
# Pero mejora bastante el recall (detecta más churners).
# Esto es esperado porque SMOTEENN elimina puntos "ruidosos" después del sobremuestreo, 
# haciendo que el modelo sea más sensible para detectar la clase minoritaria, a costa de más falsos positivos.
# Útil si el costo de no detectar un churner es muy alto.

# SMOTETomek:

# Resultados muy similares a SMOTE puro.
# Precision y recall están balanceados y con un F1 ligeramente mejor que SMOTE en Gradient Boosting.
# Mantiene buen accuracy.

# ¿Qué elegir?
# Si tu prioridad es maximizar la detección (recall) para no perder clientes aunque aumenten falsos positivos,
# SMOTEENN es una buena opción.
# Si quieres un balance más estable entre recall, precisión y accuracy, SMOTE o SMOTETomek funcionan muy bien, 
# con ligera ventaja para SMOTETomek en Gradient Boosting.
# El modelo Gradient Boosting sigue siendo el mejor en casi todos los casos por su balance en métricas.

#Ahora quiero un ajuste con  fino de hiperparámetros y con modelos Gradient Boosting y Random Forest.
# Pipeline con balanceo + búsqueda de hiperparámetros

# Balanceo con SMOTETomek (lo que vimos que funciona bien)
smotetomek = SMOTETomek(random_state=42)

# Aplicamos balanceo solo sobre entrenamiento
X_train_bal, y_train_bal = smotetomek.fit_resample(X_train, y_train)

# Parámetros para Random Forest
param_dist_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Parámetros para Gradient Boosting
# Parámetros para Gradient Boosting
param_dist_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.6, 0.8, 1.0]
}

# Creación de modelos base
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

# Definimos el scorer (puedes cambiar por 'roc_auc' o 'accuracy' si quieres)
scorer = make_scorer(f1_score)

# RandomizedSearchCV para RF
rs_rf = RandomizedSearchCV(
    rf, param_distributions=param_dist_rf, n_iter=30, scoring=scorer,
    cv=5, verbose=2, random_state=42, n_jobs=-1
)

# RandomizedSearchCV para GB
rs_gb = RandomizedSearchCV(
    gb, param_distributions=param_dist_gb, n_iter=30, scoring=scorer,
    cv=5, verbose=2, random_state=42, n_jobs=-1
)

# Ajustar RF
print("Ajustando Random Forest...")
rs_rf.fit(X_train_bal, y_train_bal)

# Ajustar GB
print("Ajustando Gradient Boosting...")
rs_gb.fit(X_train_bal, y_train_bal)

# Mejoros parámetros y evaluación en test
print("Mejores parámetros RF:", rs_rf.best_params_)
print("Mejor F1 RF (CV):", rs_rf.best_score_)

print("Mejores parámetros GB:", rs_gb.best_params_)
print("Mejor F1 GB (CV):", rs_gb.best_score_)

# Evaluar en test set original (sin balancear)
from sklearn.metrics import classification_report

# RF mejorado
y_pred_rf = rs_rf.best_estimator_.predict(X_test)
print("Reporte Random Forest optimizado:")
print(classification_report(y_test, y_pred_rf))

# GB mejorado
y_pred_gb = rs_gb.best_estimator_.predict(X_test)
print("Reporte Gradient Boosting optimizado:")
print(classification_report(y_test, y_pred_gb))

# | **Modelo**            | **Parámetros principales**                                                                                        | **Mejor F1 (CV)** | **Accuracy (test)** | **Clase**    | **Precision** | **Recall** | **F1-score** | **Support** |
# | --------------------- | ----------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------- | ------------ | ------------- | ---------- | ------------ | ----------- |
# | **Random Forest**     | n\_estimators=200, min\_samples\_split=2, min\_samples\_leaf=1, max\_depth=30, bootstrap=False                    | 0.8770            | 0.78                | 0            | 0.84          | 0.87       | 0.86         | 1035        |
# |                       |                                                                                                                   |                   |                     | 1            | 0.60          | 0.54       | 0.57         | 374         |
# |                       |                                                                                                                   |                   |                     | Macro Avg    | 0.72          | 0.71       | 0.71         | 1409        |
# |                       |                                                                                                                   |                   |                     | Weighted Avg | 0.78          | 0.78       | 0.78         | 1409        |
# | **Gradient Boosting** | subsample=0.8, n\_estimators=100, min\_samples\_split=10, min\_samples\_leaf=4, max\_depth=7, learning\_rate=0.05 | 0.8513            | 0.78                | 0            | 0.86          | 0.83       | 0.85         | 1035        |
# |                       |                                                                                                                   |                   |                     | 1            | 0.57          | 0.62       | 0.59         | 374         |
# |                       |                                                                                                                   |                   |                     | Macro Avg    | 0.71          | 0.73       | 0.72         | 1409        |
# |                       |                                                                                                                   |                   |                     | Weighted Avg | 0.78          | 0.78       | 0.78         | 1409        |
# 1. Mejor F1 en validación cruzada (CV) — en entrenamiento balanceado:
# Random Forest: 0.877

# Gradient Boosting: 0.851

# Ambos modelos mejoraron mucho el F1 en CV respecto a antes (antes estábamos en 0.6 aprox), lo cual indica que el ajuste fino fue muy efectivo para optimizar la métrica.

# 2. Evaluación en el conjunto de test original (sin balanceo):
# Modelo	Precisión (churn=1)	Recall (churn=1)	F1 (churn=1)	Accuracy general
# Random Forest	0.60	0.54	0.57	0.78
# Gradient Boosting	0.57	0.62	0.59	0.78

# Accuracy se mantiene alrededor de 78%, bastante consistente con experimentos anteriores.

# Recall para la clase positiva (churn=1) sigue siendo moderado (54-62%), lo que significa que detecta aproximadamente la mitad a un poco más de la mitad de los churners reales.

# Precisión es algo baja (57-60%), indicando que hay falsos positivos, es decir, clientes clasificados como churners que no lo son.

# F1 para churners está entre 0.57 y 0.59, ligeramente mejor que antes del ajuste (que estaba cerca de 0.56-0.60).






