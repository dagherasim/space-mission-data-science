import streamlit as st
import pandas as pd
from PIL import Image

# === Page Config ===
st.set_page_config(page_title="Space Mission Data Analysis", layout="wide")

# === Sidebar Navigation ===
st.sidebar.title("📁 Navigation")
section = st.sidebar.radio("Select a section:", [
    "Project Summary",
    "Phase 2 – Exploratory Analysis",
    "Phase 3 – Organizational Performance",
    "Phase 4 – Mars Readiness Estimation",
    "Phase 5 – ML Success Prediction & Mars Simulation",
    "Phase 5 – ML Classification Report"
])

# === Header ===
if section == "Project Summary":
    st.title("🚀 Space Mission Data Analysis (1957–Present)")
    st.markdown("A comprehensive data science project that analyzes global space missions from 1957 to today.")
    st.markdown("Explore how mission trends evolved, which agencies perform best, and use ML to simulate who might reach Mars first.")


# === Load Data ===
org_success = pd.read_csv("../processed_data/org_success_estimates_ml.csv")
mars_scores = pd.read_csv("../processed_data/mars_readiness_scores.csv")
classification_report = open("../reports/ml_classification_report.txt", "r").read()


# === Section 1: Project Summary ===
if section == "Project Summary":
    st.subheader("📌 Project Overview")
    st.markdown("""
    This project was divided into multiple structured phases:

    - **Phase 1:** Data cleaning and preparation  
    - **Phase 2:** Exploratory visual analysis  
    - **Phase 3:** Success performance ranking per organization  
    - **Phase 4:** Statistical readiness scoring for reaching Mars  
    - **Phase 5:** Machine Learning predictions and realistic Mars ETA simulation  

    The dataset contains **4,300+ missions** and spans over 60 years of spaceflight history.
    """)

# === Section 2: Phase 2 – EDA ===
elif section == "Phase 2 – Exploratory Analysis":
    st.title("📊 Phase 2: Exploratory Data Analysis")
    
    st.subheader("Launches Per Year")
    st.markdown("This line chart shows the evolution of space missions over time. Peaks may correspond with major global events, such as the Space Race, satellite booms, or commercial space expansions.")
    st.image("../charts/launches_per_year.png")

    st.subheader("Top 10 Organizations by Launches")
    st.markdown("This bar chart ranks the organizations with the highest total number of launches. It reflects mission volume, not necessarily reliability.")
    st.image("../charts/top_organizations.png")

    st.subheader("Mission Outcome Distribution")
    st.markdown("This plot displays how many missions resulted in success, failure, or other outcomes. It provides a quick view of historical mission reliability.")
    st.image("../charts/mission_outcomes.png")

    st.subheader("📊 Phase 2: Exploratory Data Analysis")
    st.image("../charts/launches_per_year.png", caption="Launches Per Year")
    st.image("../charts/top_organizations.png", caption="Top 10 Organizations by Launches")
    st.image("../charts/mission_outcomes.png", caption="Mission Outcome Distribution")

# === Section 3: Phase 3 – Organizational Performance ===
elif section == "Phase 3 – Organizational Performance":
    st.title("📈 Phase 3: Organizational Performance Analysis")

    st.subheader("Top 10 by Success Rate")
    st.markdown("This chart displays the ten most reliable space agencies based on the percentage of successful missions. Only organizations with ≥10 missions were included.")
    st.image("../charts/top_10_success_rates.png")

    st.subheader("Bottom 10 by Success Rate")
    st.markdown("This chart highlights the least reliable agencies — useful for identifying riskier or experimental operations. Again, filtered to orgs with at least 10 missions.")
    st.image("../charts/bottom_10_success_rates.png")

    st.subheader("Detailed Success Rate Table")
    st.markdown("This table provides the full list of qualifying organizations, with total missions, successes, and calculated success rates.")
    st.dataframe(pd.read_csv("../processed_data/organization_success_rates.csv"))


# === Section 4: Phase 4 – Statistical Mars Estimation ===
elif section == "Phase 4 – Mars Readiness Estimation":
    st.title("🪐 Phase 4: Mars Readiness Estimation (Statistical)")

    st.subheader("Mars Readiness Score (Top 10)")
    st.markdown("This chart ranks agencies based on a custom readiness score. The score combines launch volume and historical success rate to estimate likelihood of Mars capability.")
    st.image("../charts/mars_race_top10.png")

    st.subheader("Readiness Scores Table")
    st.markdown("This table lists all organizations included in the statistical Mars-readiness evaluation, showing success rate, total launches, and score.")
    st.dataframe(mars_scores)


# === Section 5: Phase 5 – ML Predictions & Mars Simulation ===
elif section == "Phase 5 – ML Success Prediction & Mars Simulation":
    st.title("🤖 Phase 5: Machine Learning Predictions and Simulated Mars Arrival")

    st.subheader("Predicted Success Rates (ML Model)")
    st.markdown("This bar chart shows the predicted mission success probability for each organization based on a trained ML model using Random Forest classification.")
    st.image("../charts/ml_predicted_success_rates.png")

    st.subheader("Simulated Mars Arrival Year (Top 10)")
    st.markdown("This projection estimates when each organization could realistically reach Mars based on predicted mission reliability and historical mission rate.")
    st.image("../charts/mars_reach_projection.png")

    st.subheader("ML-Based Predictions Table")
    st.markdown("This table provides the full output of the ML model, including predicted success/failure probabilities and simulated Mars arrival year.")
    st.dataframe(org_success)

# === Section 6: ML Classification Report ===
elif section == "Phase 5 – ML Classification Report":
    st.title("📄 ML Evaluation Report")
    st.subheader("Model Evaluation (Random Forest Classifier)")
    st.markdown("This classification report shows how well the model performed in predicting mission outcomes during testing. Key metrics include precision, recall, and F1-score.")
    st.code(classification_report)

