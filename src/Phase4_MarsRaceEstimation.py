import pandas as pd
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------------------------------------------------
# PHASE 4: Predictive Estimation & Mars Race Leadership
# --------------------------------------------------------------------------------------------------
# This script uses historical mission performance data to estimate:
#   - The most successful and reliable organizations
#   - A statistical probability-based guess at which organization is most likely to reach Mars
# This version uses a conventional scoring approach. Optional ML extensions can be added later.
# --------------------------------------------------------------------------------------------------

# Load precomputed performance data
success_df = pd.read_csv("processed_data/organization_success_rates.csv")

# Only consider organizations with at least 10 missions for meaningful analysis
qualified = success_df[success_df['total_missions'] >= 10].copy()

# Normalize columns for scoring
qualified['normalized_success_rate'] = qualified['success_rate'] / 100
qualified['normalized_volume'] = qualified['total_missions'] / qualified['total_missions'].max()

# Composite Mars-readiness score (adjust weights as needed)
qualified['mars_score'] = (
    0.7 * qualified['normalized_success_rate'] +
    0.3 * qualified['normalized_volume']
)

# Sort by Mars score
qualified = qualified.sort_values(by='mars_score', ascending=False)

# Save results
os.makedirs("processed_data", exist_ok=True)
qualified.to_csv("processed_data/mars_readiness_scores.csv", index=False)

# === Visualization: Mars Readiness Top 10 ===
plt.figure(figsize=(12, 6))
qualified.head(10).set_index('Organisation')['mars_score'].sort_values().plot(kind='barh', color='orange')
plt.title("Top 10 Organizations Most Likely to Reach Mars First")
plt.xlabel("Composite Mars Readiness Score")
plt.tight_layout()

# Save chart
os.makedirs("charts", exist_ok=True)
plt.savefig("charts/mars_race_top10.png")
plt.close()

# Display Top Candidate
top_candidate = qualified.iloc[0]

print("✅ Phase 4 completed successfully.")
print("Most probable Mars first-arriver:")
print(f"🚀 {top_candidate['Organisation']} — Estimated readiness score: {top_candidate['mars_score']:.3f}")
