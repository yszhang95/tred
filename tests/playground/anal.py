import re
import json
import pandas as pd

# File path to the log file
#log_file_path = "log3.log"  # Change this to your actual file path
log_file_path = "run_chunkify2.log"  # Change this to your actual file path

# Read the log from file
with open(log_file_path, "r") as f:
    log_data = f.read()

# Define regex patterns
step_pattern = re.compile(r"INFO ([\d.e+-]+) (recomb|drift|raster|chunksum_charge|convo)")
itpc_pattern = re.compile(r"INFO itpc(\d+), tpc label (\d+), batch label \d+, N segments (\d+), elapsed ([\d.e+-]+) sec")

results = []
current = {}

# Parse log lines
for line in log_data.strip().split("\n"):
    step_match = step_pattern.search(line)
    if step_match:
        time, step = step_match.groups()
        current[step] = float(time)
    else:
        itpc_match = itpc_pattern.search(line)
        if itpc_match:
            itpc, tpc, n_segments, elapsed = itpc_match.groups()
            current.update({
                "itpc": int(itpc),
                "tpc": int(tpc),
                "N_segments": int(n_segments),
                "elapsed": float(elapsed)
            })
            results.append(current)
            current = {}

# Convert to DataFrame
df = pd.DataFrame(results)

# Save to JSON
json_path = "itpc_timing_summary.json"
df.to_json(json_path, orient="records", indent=2)

# Save to CSV (optional)
csv_path = "itpc_timing_summary.csv"
df.to_csv(csv_path, index=False)

print(f"Saved summary to:\n- {json_path}\n- {csv_path}")

