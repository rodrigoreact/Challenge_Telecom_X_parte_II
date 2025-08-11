import matplotlib.pyplot as plt

labels = ['Clase 0', 'Clase 1']
before = [4133, 1501]
after = [4133, 4133]

x = range(len(labels))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x, before, width, label='Antes')
ax.bar([i + width for i in x], after, width, label='Después')

ax.set_ylabel('Número de registros')
ax.set_title('Distribución antes y después de SMOTE')
ax.set_xticks([i + width/2 for i in x])
ax.set_xticklabels(labels)
ax.legend()

plt.savefig('smote_comparison.png')
print('Archivo guardado: smote_comparison.png')