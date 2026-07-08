import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Example data (replace these with your actual simulation results)
# ---------------------------
junctions = ['gneJ11', 'gneJ2', 'gneJ21', 'gneJ3', 'gneJ7']
awt_junction = [0.30, 0.27, 0.30, 0.41, 0.28]       # Avg waiting time per junction
aql_junction = [0.17, 0.13, 0.16, 0.29, 0.15]       # Avg queue length per junction

# ---------------------------
# 1. Per-junction AWT
# ---------------------------
plt.figure(figsize=(8,5))
plt.bar(junctions, awt_junction, color='skyblue')
plt.xlabel('Junction')
plt.ylabel('Average Waiting Time (s)')
plt.title('Average Waiting Time per Junction')
plt.tight_layout()
plt.savefig('plots/awt_per_junction.png')
plt.show()

# ---------------------------
# 2. Per-junction AQL
# ---------------------------
plt.figure(figsize=(8,5))
plt.bar(junctions, aql_junction, color='orange')
plt.xlabel('Junction')
plt.ylabel('Average Queue Length (vehicles)')
plt.title('Average Queue Length per Junction')
plt.tight_layout()
plt.savefig('plots/aql_per_junction.png')
plt.show()
