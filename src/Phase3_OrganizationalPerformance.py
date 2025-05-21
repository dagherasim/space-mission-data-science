import pandas as pd
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------------------------------------------------
# PHASE 3: Organizational Performance Analysis
# --------------------------------------------------------------------------------------------------
# This script analyzes the performance of various launch organizations based on mission outcomes.
# The analysis includes:
#   - Success rate per organization
#   - Best and worst performers (minimum mission threshold applied)
#   - Success rate bar chart
# Results are saved to 'charts/' directory.
# --------------------------------------------------------------------------------------------------

# Load cleaned dataset
df = pd.read_csv("processed_data/mission_launches_clean.csv")

# Define output path for visualizations
charts_dir = "charts"
os.makedirs(charts_dir, exist_ok=True)

# Normalize Mission_Status column
df['Mission_Status'] = df['Mission_Status'].str.strip().str.capitalize()

# Compute success metrics per organization
success_df = df.groupby('Organisation').agg(
    total_missions=('Mission_Status', 'count'),
    successful_missions=('Mission_Status', lambda x: (x == 'Success').sum())
)

success_df['success_rate'] = (success_df['successful_missions'] / success_df['total_missions']) * 100
success_df = success_df.sort_values(by='success_rate', ascending=False)

# Filter out organizations with fewer than 10 launches (to ensure statistical relevance)
filtered_success_df = success_df[success_df['total_missions'] >= 10]

# === Save top 10 success rates ===
top_success = filtered_success_df.head(10)
bottom_success = filtered_success_df.tail(10)

# === Plot Success Rates ===
plt.figure(figsize=(12, 6))
top_success['success_rate'].sort_values().plot(kind='barh', color='green')
plt.title("Top 10 Organizations by Mission Success Rate")
plt.xlabel("Success Rate (%)")
plt.tight_layout()
plt.savefig(f"{charts_dir}/top_10_success_rates.png")
plt.close()

plt.figure(figsize=(12, 6))
bottom_success['success_rate'].sort_values().plot(kind='barh', color='red')
plt.title("Bottom 10 Organizations by Mission Success Rate (≥10 Missions)")
plt.xlabel("Success Rate (%)")
plt.tight_layout()
plt.savefig(f"{charts_dir}/bottom_10_success_rates.png")
plt.close()

# Save raw data
filtered_success_df.to_csv("processed_data/organization_success_rates.csv")

# Output results
print("✅ Phase 3 completed successfully.")
print("Success rates calculated for organizations with ≥10 missions.")
print("Charts saved to:", charts_dir)
