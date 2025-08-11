import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
ax.text(0.5, 0.9, 'Recomendaciones Estratégicas:', ha='center', fontsize=14)
ax.text(0.5, 0.7, 'Paquetes promocionales para clientes con fibra óptica', ha='center', fontsize=12)
ax.text(0.5, 0.5, 'Incentivos para migrar a contratos anuales', ha='center', fontsize=12)
ax.text(0.5, 0.3, 'Programa de fidelización para clientes nuevos (primeros 6 meses)', ha='center', fontsize=12)
ax.text(0.5, 0.1, 'Monitoreo proactivo de clientes con pago electrónico', ha='center', fontsize=12)
ax.axis('off')

plt.savefig('action_plan.png')
print('Archivo guardado: action_plan.png')