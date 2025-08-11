import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
ax.text(0.5, 0.5, 'Estadísticas después de normalizar:\nMedia ≈ 0\nDesviación estándar ≈ 1\nRango: [-2.5, 2.5] para la mayoría de valores', ha='center', va='center', fontsize=12)
ax.axis('off')

plt.savefig('normalization_results.png')
print('Archivo guardado: normalization_results.png')