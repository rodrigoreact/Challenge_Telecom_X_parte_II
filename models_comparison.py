import matplotlib.pyplot as plt

labels = ['Regresión Logística', 'Random Forest', 'KNN']
accuracy = [80, 82, 76]
precision = [66, 67, 58]
recall = [53, 63, 49]

x = range(len(labels))
width = 0.25

fig, ax = plt.subplots()
ax.bar(x, accuracy, width, label='Accuracy')
ax.bar([i + width for i in x], precision, width, label='Precisión (Churn=1)')
ax.bar([i + 2*width for i in x], recall, width, label='Recall (Churn=1)')

ax.set_ylabel('Porcentaje (%)')
ax.set_title('Comparación de Modelos')
ax.set_xticks([i + width for i in x])
ax.set_xticklabels(labels)
ax.legend()

plt.savefig('models_comparison.png')
print('Archivo guardado: models_comparison.png')