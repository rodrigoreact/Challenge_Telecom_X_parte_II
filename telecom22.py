import pandas as pd
import requests
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, make_scorer, f1_score
from imblearn.combine import SMOTETomek

# 1. Descargar y cargar dataset JSON
url = "https://raw.githubusercontent.com/ingridcristh/challenge2-data-science-LATAM/refs/heads/main/TelecomX_Data.json"
response = requests.get(url)

if response.status_code == 200:
    data_json = response.json()
    df = pd.json_normalize(data_json)
    print("Datos cargados correctamente.")
else:
    print(f"Error al acceder a la URL: {response.status_code}")
    df = pd.DataFrame()

# 2. Eliminar columnas no útiles
cols_a_eliminar = ['customerID']
df = df.drop(columns=cols_a_eliminar)

# 3. Limpieza variable objetivo
target = "Churn"
df[target] = df[target].astype(str).str.strip()
df.loc[~df[target].isin(["Yes", "No"]), target] = np.nan
df = df.dropna(subset=[target])
df[target] = df[target].map({"Yes": 1, "No": 0})

# 4. Separar variables predictoras y objetivo
X = df.drop(columns=[target])
y = df[target]

# 5. Codificación One-Hot para variables categóricas nominales
X = pd.get_dummies(X, drop_first=True)

# 6. Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. División train/test con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Balanceo con SMOTETomek
smotetomek = SMOTETomek(random_state=42)
X_train_bal, y_train_bal = smotetomek.fit_resample(X_train, y_train)

# 9. Definir parámetros para ajuste hiperparámetros
param_dist_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

param_dist_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.6, 0.8, 1.0]
}

# 10. Crear modelos base y scorer
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)
scorer = make_scorer(f1_score)

# 11. Configurar búsqueda RandomizedSearchCV
rs_rf = RandomizedSearchCV(
    rf, param_dist_rf, n_iter=30, scoring=scorer, cv=5, verbose=2, random_state=42, n_jobs=-1
)

rs_gb = RandomizedSearchCV(
    gb, param_dist_gb, n_iter=30, scoring=scorer, cv=5, verbose=2, random_state=42, n_jobs=-1
)

# 12. Ajustar modelos
print("Ajustando Random Forest...")
rs_rf.fit(X_train_bal, y_train_bal)

print("Ajustando Gradient Boosting...")
rs_gb.fit(X_train_bal, y_train_bal)

# 13. Resultados de mejor modelo
print("Mejores parámetros RF:", rs_rf.best_params_)
print("Mejor F1 RF (CV):", rs_rf.best_score_)
print("Mejores parámetros GB:", rs_gb.best_params_)
print("Mejor F1 GB (CV):", rs_gb.best_score_)

# 14. Evaluar en test set original (sin balancear)
y_pred_rf = rs_rf.best_estimator_.predict(X_test)
print("Reporte Random Forest optimizado:")
print(classification_report(y_test, y_pred_rf))

y_pred_gb = rs_gb.best_estimator_.predict(X_test)
print("Reporte Gradient Boosting optimizado:")
print(classification_report(y_test, y_pred_gb))

# 15. Evaluar en test set balanceado
y_pred_rf_bal = rs_rf.best_estimator_.predict(X_test)
print("Reporte Random Forest optimizado (balanceado):")
print(classification_report(y_test, y_pred_rf_bal))

y_pred_gb_bal = rs_gb.best_estimator_.predict(X_test)
print("Reporte Gradient Boosting optimizado (balanceado):")
print(classification_report(y_test, y_pred_gb_bal))