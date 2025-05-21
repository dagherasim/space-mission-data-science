import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------------------------------------------
# PHASE 5: ML-Based Organizational Success Estimation & Mars Projection
# --------------------------------------------------------------------------------------------------
# This script trains a model to predict mission success based on historical data,
# then estimates each organization's future success probability and simulated Mars ETA.
# --------------------------------------------------------------------------------------------------

# === Load and filter dataset ===
df = pd.read_csv("processed_data/mission_launches_clean.csv")
df = df[df['Mission_Status'].isin(['Success', 'Failure'])].copy()

# === Feature selection ===
features = df[['Organisation', 'Rocket_Status', 'Year']]
target = df['Mission_Status']

# === Encode categorical variables ===
le_org = LabelEncoder()
le_rocket = LabelEncoder()
le_target = LabelEncoder()

features['Organisation'] = le_org.fit_transform(features['Organisation'])
features['Rocket_Status'] = le_rocket.fit_transform(features['Rocket_Status'])
target_encoded = le_target.fit_transform(target)

# === Train/test split and model training ===
X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Estimate success/failure rate per organization ===
unique_orgs = df['Organisation'].unique()
org_results = []

for org in unique_orgs:
    org_data = df[df['Organisation'] == org]
    if len(org_data) < 5:
        continue  # Filter out low-activity organizations

    X_org = org_data[['Organisation', 'Rocket_Status', 'Year']].copy()
    X_org['Organisation'] = le_org.transform([org] * len(X_org))
    X_org['Rocket_Status'] = le_rocket.transform(X_org['Rocket_Status'])

    probs = model.predict_proba(X_org)
    success_prob = np.mean(probs[:, le_target.transform(['Success'])[0]])
    failure_prob = np.mean(probs[:, le_target.transform(['Failure'])[0]])

    org_results.append({
        'Organisation': org,
        'Estimated_Success_Probability': round(success_prob, 3),
        'Estimated_Failure_Probability': round(failure_prob, 3),
        'Historical_Missions': len(org_data)
    })

# === Create and save DataFrame ===
org_estimates_df = pd.DataFrame(org_results).sort_values(by='Estimated_Success_Probability', ascending=False)
os.makedirs("processed_data", exist_ok=True)
org_estimates_df.to_csv("processed_data/org_success_estimates_ml.csv", index=False)

print("✅ ML estimation complete. Top organizations by predicted success probability:")
print(org_estimates_df.head(10))

# === Chart: ML-Predicted Success Rates (Top 15) ===
top15 = org_estimates_df.head(15).sort_values(by='Estimated_Success_Probability')
plt.figure(figsize=(12, 6))
plt.barh(top15['Organisation'], top15['Estimated_Success_Probability'], color='skyblue')
plt.xlabel("Estimated Success Probability")
plt.title("Top 15 Organizations by ML-Predicted Success Rate")
plt.tight_layout()
os.makedirs("charts", exist_ok=True)
plt.savefig("charts/ml_predicted_success_rates.png")
plt.close()

print("📊 Success probability chart saved as: charts/ml_predicted_success_rates.png")

# --------------------------
# Mars Year Reach Simulation 
# --------------------------

required_successes_to_mars = 20   # More realistic bar for Mars readiness
historical_timespan = 30         # Spread over 3 decades to smooth spikes
base_year = 2025

# Prevent division spikes from low mission volume
org_estimates_df['Adjusted_Missions'] = org_estimates_df['Historical_Missions'].clip(lower=10)

# Estimate successes per year
org_estimates_df['Estimated_Successes_Per_Year'] = (
    org_estimates_df['Estimated_Success_Probability'] *
    (org_estimates_df['Adjusted_Missions'] / historical_timespan)
)

# Filter out orgs with extremely low activity
org_estimates_df = org_estimates_df[org_estimates_df['Estimated_Successes_Per_Year'] > 0.1]

# Estimate Mars ETA year
org_estimates_df['Estimated_Mars_Year'] = base_year + (
    required_successes_to_mars / org_estimates_df['Estimated_Successes_Per_Year']
)

# Filter for sanity (no Mars in 2100+ or 2026)
org_estimates_df = org_estimates_df[
    (org_estimates_df['Estimated_Mars_Year'] >= 2026) &
    (org_estimates_df['Estimated_Mars_Year'] <= 2100)
]

# Save updated results
org_estimates_df.to_csv("processed_data/org_success_estimates_ml.csv", index=False)

# --------------------------
# Plot Mars Reach Projection
# --------------------------
top_mars_candidates = org_estimates_df.sort_values(by='Estimated_Mars_Year').head(10)

plt.figure(figsize=(12, 6))
bars = plt.barh(top_mars_candidates['Organisation'], top_mars_candidates['Estimated_Mars_Year'], color='tomato')

# Add labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
             f"{int(width)}", va='center', fontsize=9)

plt.xlabel("Estimated Year to Reach Mars")
plt.title("Simulated Mars Arrival Projection (Top 10, Adjusted Model)")
plt.xlim(2026, top_mars_candidates['Estimated_Mars_Year'].max() + 5)
plt.tight_layout()
plt.savefig("charts/mars_reach_projection.png")
plt.close()

print("📡 Mars simulation chart updated with realistic assumptions.")

