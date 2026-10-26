import matplotlib.pyplot as plt

vals = [52, 65, 36, 38, 24, 27, 21, 25, 0, 0]
labs = [f"{i * 10}-{i * 10 + 9}" for i in range(10)]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labs, vals, color="tab:blue", width=0.5, label="response time")

ax.bar_label(bars, fmt="%.3f", padding=3)
ax.set_xlabel("Cluster Box Groups")
ax.set_ylabel("Worst Response Time")
ax.set_title("Worst Response Time for Cluster Boxes")
ax.margins(y=0.15)

plt.tight_layout()
plt.savefig("mp2_p1_response_graph.png", dpi=300)
plt.show()