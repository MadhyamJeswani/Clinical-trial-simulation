# ============================================================
# CLINICAL TRIAL SIMULATION - CAPGEMINI INTERNSHIP 2026
# File: data_exploration.py
# Purpose: Load and explore the clinical trials dataset
# Author: Madhyam
# ============================================================

# STEP 1: Import libraries
# pandas helps us work with data like a spreadsheet in Python
import pandas as pd
import numpy as np

# ============================================================
# STEP 2: Load the dataset
# ============================================================

# This tells Python where your CSV file is
# Since we are inside 'notebooks' folder, we go one level up (..)
# then into 'data' folder to find the CSV
df = pd.read_csv('../data/clin_trials.csv', low_memory=False)

# ============================================================
# STEP 3: Basic exploration
# ============================================================

print("=" * 60)
print("CLINICAL TRIALS DATASET - BASIC EXPLORATION")
print("=" * 60)

# How many rows and columns?
print(f"\n Total Rows (trials): {df.shape[0]}")
print(f" Total Columns: {df.shape[1]}")

# What are the column names?
print("\n Column Names:")
for col in df.columns:
    print(f"   - {col}")

# Show first 5 rows
print("\n First 5 rows of data:")
print(df.head())

# Basic statistics
print("\n Phase Distribution (how many trials per phase):")
print(df['Phases'].value_counts())

print("\n Overall Status Distribution:")
print(df['Overall Status'].value_counts())

print("\n Study Type Distribution:")
print(df['Study Type'].value_counts())

print("\n Missing Values per column:")
print(df.isnull().sum())

print("\n" + "=" * 60)
print("EXPLORATION COMPLETE!")
print("=" * 60)
print(df.describe())