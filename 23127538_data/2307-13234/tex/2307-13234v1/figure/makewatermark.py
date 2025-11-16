from matplotlib import pyplot as plt

plt.text(x=0.5, y=0.5, s="preliminary", fontsize=50, color="gray", alpha=0.5, ha='center', va='center', rotation=32.)
plt.axis('off')
plt.tight_layout()

#plt.show()
plt.savefig('fig_watermark.pdf', bbox_inches='tight', transparent=True)