import pandas as pd
import requests
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

# Descargar y cargar dataset JSON
url = "https://raw.githubusercontent.com/ingridcristh/challenge2-data-science-LATAM/refs/heads/main/TelecomX_Data.json"
response = requests.get(url)

if response.status_code == 200:
    data_json = response.json()
    df = pd.json_normalize(data_json)
    print("Datos cargados correctamente.")
else:
    print(f"Error al acceder a la URL: {response.status_code}")
    df = pd.DataFrame()

# Eliminar columnas no útiles
cols_a_eliminar = ['customerID']
df = df.drop(columns=cols_a_eliminar)

# Limpieza variable objetivo
target = "Churn"
df[target] = df[target].astype(str).str.strip()
df.loc[~df[target].isin(["Yes", "No"]), target] = np.nan
df = df.dropna(subset=[target])
df[target] = df[target].map({"Yes": 1, "No": 0})

# Separar variables predictoras y objetivo
X = df.drop(columns=[target])
y = df[target]

# Codificación variables categóricas
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Paso 3 – Separar variables predictoras y objetivo
X = df.drop(columns=[target])
y = df[target]

# Paso 4 – Codificación One-Hot para variables categóricas (nominales)
X = pd.get_dummies(X, drop_first=True)

# Paso 5 – Escalado de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Paso 6 – División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División train/test con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Balanceo con SMOTETomek (mejor rendimiento en análisis previo)
smotetomek = SMOTETomek(random_state=42)
X_train_bal, y_train_bal = smotetomek.fit_resample(X_train, y_train)

# Modelos y parámetros para ajuste
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

rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)
scorer = make_scorer(f1_score)

# Búsqueda RandomizedSearchCV
rs_rf = RandomizedSearchCV(rf, param_dist_rf, n_iter=30, scoring=scorer, cv=5, verbose=2, random_state=42, n_jobs=-1)
rs_gb = RandomizedSearchCV(gb, param_dist_gb, n_iter=30, scoring=scorer, cv=5, verbose=2, random_state=42, n_jobs=-1)

print("Ajustando Random Forest...")
rs_rf.fit(X_train_bal, y_train_bal)

print("Ajustando Gradient Boosting...")
rs_gb.fit(X_train_bal, y_train_bal)

print("Mejores parámetros RF:", rs_rf.best_params_)
print("Mejor F1 RF (CV):", rs_rf.best_score_)
print("Mejores parámetros GB:", rs_gb.best_params_)
print("Mejor F1 GB (CV):", rs_gb.best_score_)

# Evaluación en test original sin balancear
y_pred_rf = rs_rf.best_estimator_.predict(X_test)
print("Reporte Random Forest optimizado:")
print(classification_report(y_test, y_pred_rf))

y_pred_gb = rs_gb.best_estimator_.predict(X_test)
print("Reporte Gradient Boosting optimizado:")
print(classification_report(y_test, y_pred_gb))
