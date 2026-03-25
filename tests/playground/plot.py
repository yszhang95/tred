import pandas as pd
import matplotlib.pyplot as plt

# ? Replace this with your actual path
# csv_path = "path/to/your/itpc_timing_summary.csv"
csv_path = "./itpc_timing_summary.csv"

# Load CSV
df = pd.read_csv(csv_path)

# Sum up timing for each step
columns_to_plot = ["recomb", "drift", "raster", "chunksum_charge", "convo"]
summed_values = df[columns_to_plot].sum()

plt.figure(figsize=(8, 8))
plt.pie(
    summed_values,
    labels=columns_to_plot,
    autopct='%1.1f%%',
    startangle=140
)
plt.title("Total Time Distribution by Step")
plt.axis("equal")  # Ensure it's a circle
plt.tight_layout()
plt.savefig('pie_chart_all.png')
plt.show()

columns_to_plot = ["recomb", "drift", "raster", "chunksum_charge", ]
summed_values = df[columns_to_plot].sum()

# Plot pie chart
plt.figure(figsize=(8, 8))
plt.pie(
    summed_values,
    labels=columns_to_plot,
    autopct='%1.1f%%',
    startangle=140
)
plt.title("Total Time Distribution by Step (no convo)")
plt.axis("equal")  # Ensure it's a circle
plt.tight_layout()
plt.savefig('pie_chart_noconvo.png')
plt.show()
