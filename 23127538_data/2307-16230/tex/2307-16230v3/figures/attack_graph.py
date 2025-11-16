import matplotlib.pyplot as plt
import numpy as np

window_size = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

private_accuracy = np.array([0.990, 0.989, 0.985, 0.948, 0.941, 0.969, 0.984, 0.959, 0.963])
cracking_accuracy = np.array([0.95, 0.93 ,0.78 ,0.54 ,0.52 ,0.51 ,0.501, 0.500, 0.498])

reverse_train_accuracy = np.array([0.58, 0.53 ,0.52 ,0.51 ,0.50 ,0.50 ,0.501, 0.500, 0.50])
# accuracy_values1 = np.array([0.515, 0.937 ,0.986 ,0.993 ,0.997 ,0.999 ,0.999  ])
# accuracy_values2 = np.array([5.57, 5.54 ,5.6 ,5.9 ,6.0 , 6.2 , 7.5 ])

fig, ax1 = plt.subplots(figsize=(8, 6))

# Change color for each plot to make them more distinguishable
ax1.set_xlabel('Window Size', fontsize=18, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=18, fontweight='bold')
ax1.plot(window_size, private_accuracy, 'o-', color='tab:red', label='Detection Accuracy', linewidth=3)
ax1.plot(window_size, cracking_accuracy, 's-', color='tab:blue', label='Cracking Accuracy', linewidth=3)
ax1.plot(window_size, reverse_train_accuracy, '^-', color='tab:green', label='Reverse Training Accuracy', linewidth=2)

# Customize tick parameters
ax1.tick_params(axis='y', labelcolor='black', width=3)
plt.tick_params(axis='x', which='major', labelsize=20, width=2)
plt.tick_params(axis='y', which='major', labelsize=20, width=2)

plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
# Add title and legend
# plt.title('Accuracy & PPL vs. $\delta$ value', fontsize=18, fontweight='bold')
ax1.legend(loc='center right')
ax1.legend(fontsize=20, prop={'weight':'bold'})
# plt.show()
plt.savefig('attack_graph.pdf', format='pdf')
