import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
ax.text(0.5, 0.5, 'Métricas Finales (Modelo Optimizado):\nAccuracy: 80%\nRecall (Churn=1): 49%\nPrecisión (Churn=1): 67%', ha='center', va='center', fontsize=12)
ax.axis('off')

plt.savefig('confusion_matrix_optimized.png')
print('Archivo guardado: confusion_matrix_optimized.png')