import matplotlib.pyplot as plt

labels = ['Fibra Óptica', 'Antigüedad (Tenure)', 'Contrato 2 años']
values = [0.25, -0.18, -0.15]

fig, ax = plt.subplots()
ax.barh(labels, values, color=['red' if v > 0 else 'blue' for v in values])
ax.set_xlabel('Impacto en Churn')
ax.set_title('Resumen de Importancia SHAP')

plt.savefig('shap_summary.png')
print('Archivo guardado: shap_summary.png')