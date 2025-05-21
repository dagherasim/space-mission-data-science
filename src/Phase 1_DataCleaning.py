import pandas as pd
from dateutil import parser
import os

# --------------------------------------------------------------------------------------------------
# PHASE 1: Data Acquisition and Cleaning
# --------------------------------------------------------------------------------------------------
# This script is responsible for loading the raw space mission dataset, cleaning inconsistencies,
# and preparing the data for further analysis. Key operations include:
#   - Parsing dates with varying formats
#   - Normalizing timezone-aware datetime entries
#   - Cleaning price data
#   - Extracting the launch year
# --------------------------------------------------------------------------------------------------

# Load the dataset
df = pd.read_csv("mission_launches.csv")

# Drop redundant index columns
df.drop(columns=["Unnamed: 0.1", "Unnamed: 0"], inplace=True)

# Define a safe parser function using the dateutil library
def parse_date_safe(date_str):
    try:
        return parser.parse(date_str)
    except:
        return pd.NaT

# Parse the 'Date' column and remove invalid entries
df['Date'] = df['Date'].apply(parse_date_safe)
df = df[df['Date'].notna()]

# Normalize all datetime values to be timezone-naive
df['Date'] = df['Date'].apply(lambda x: x.replace(tzinfo=None) if x.tzinfo else x)

# Convert column to pandas datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract launch year from 'Date' column
df['Year'] = df['Date'].dt.year

# Clean the 'Price' column by removing symbols and converting to float
df['Price'] = df['Price'].replace('[\$,]', '', regex=True).astype(float)

# Optional: Export the cleaned dataset to a new CSV (for traceability)
os.makedirs("processed_data", exist_ok=True)
df.to_csv("processed_data/mission_launches_clean.csv", index=False)

# Sanity check output
print("✅ Phase 1 completed successfully.")
print("Rows after cleaning:", df.shape[0])
print("Date column dtype:", df['Date'].dtype)
