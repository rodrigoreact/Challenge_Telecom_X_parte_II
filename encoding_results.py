import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
ax.text(0.5, 0.8, 'Variables transformadas:', ha='center', fontsize=14)
ax.text(0.5, 0.6, 'Churn: Yes → 1, No → 0', ha='center', fontsize=12)
ax.text(0.5, 0.4, 'Contract: 3 nuevas columnas (Month-to-month, One year, Two year)', ha='center', fontsize=12)
ax.text(0.5, 0.2, 'Gender: Male → 1, Female → 0', ha='center', fontsize=12)
ax.axis('off')

plt.savefig('encoding_results.png')
print('Archivo guardado: encoding_results.png')