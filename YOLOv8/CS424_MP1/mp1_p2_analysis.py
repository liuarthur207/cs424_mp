import json
import matplotlib.pyplot as plt
from IoTObjectDetection.IoTObjectDetectionModuleHelperFunctions import ( get_group_avg_response_time )

path = "../../dataset/object_processing_order_history/camera_frame_processing_history.json"
with open(path) as file:
    history = json.load(file)
    
avgs = get_group_avg_response_time(history)

labels = [f"{i * 10}-{i * 10 + 9}" for i in range(10)]
counts = [0] * 10

for entry in history.values():
    group = int(entry["depth"] / 10) #simple loop for the counts above the bars
    if 0 <= group < 10:
        counts[group] += 1

colors = plt.get_cmap("tab10").colors
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(
    labels,
    avgs,
    color=colors,
    edgecolor="black",
    linewidth=0.6
)
ax.bar_label(bars, labels=counts)
ax.set_xlabel("BBox Groups")
ax.set_ylabel("Average response time")
ax.set_title("Average Response Time of BBox")

plt.tight_layout()
plt.savefig("problem2_response_times.png")
plt.show()