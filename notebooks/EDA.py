# ============================================================
# CLINICAL TRIAL SIMULATION - CAPGEMINI INTERNSHIP 2026
# File: eda.py
# Purpose: Exploratory Data Analysis - understand our data
#          through charts and statistics
# Author: Madhyam
# ============================================================

# ============================================================
# STEP 1: Import libraries
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the visual style of all charts
# Think of this like choosing a theme for your charts
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ============================================================
# STEP 2: Load the CLEANED dataset
# We always work with cleaned data from now on
# ============================================================
print("Loading cleaned dataset...")
df = pd.read_csv('../data/clin_trials_cleaned.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# STEP 3: CHART 1 - Phase Distribution
# How many trials are in each phase?
# ============================================================
print("\nCreating Chart 1: Phase Distribution...")

plt.figure()  # creates a new empty chart canvas

# Count how many trials per phase
phase_counts = df['Phase'].value_counts()

# Create a bar chart
colors = ['#2196F3', '#4CAF50', '#FF9800']  # blue, green, orange
bars = plt.bar(phase_counts.index, phase_counts.values, color=colors)

# Add labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # x position
        height + 500,                         # y position
        f'{int(height):,}',                   # text to show
        ha='center',                          # horizontal alignment
        fontsize=12,
        fontweight='bold'
    )

plt.title('Clinical Trials by Phase', fontsize=16, fontweight='bold')
plt.xlabel('Trial Phase', fontsize=13)
plt.ylabel('Number of Trials', fontsize=13)
plt.tight_layout()
plt.savefig('../data/chart1_phase_distribution.png')
plt.show()
print("Chart 1 saved!")

# ============================================================
# STEP 4: CHART 2 - Success vs Failure by Phase
# This is very important for our simulation!
# ============================================================
print("\nCreating Chart 2: Success vs Failure by Phase...")

# Filter out unknown outcomes (-1) - keep only 0 and 1
df_known = df[df['Outcome'] != -1]

# Group by Phase AND Outcome and count
outcome_by_phase = df_known.groupby(
    ['Phase', 'Outcome']
).size().unstack()

# Rename columns from 0,1 to Failure, Success
outcome_by_phase.columns = ['Failure', 'Success']

# Calculate success RATE (percentage)
outcome_by_phase['Success Rate %'] = (
    outcome_by_phase['Success'] /
    (outcome_by_phase['Success'] + outcome_by_phase['Failure'])
    * 100
).round(2)

print("\nSuccess Rate by Phase:")
print(outcome_by_phase)

# Create grouped bar chart
plt.figure()
x = range(len(outcome_by_phase.index))
width = 0.35

bars1 = plt.bar(
    [i - width/2 for i in x],
    outcome_by_phase['Success'],
    width,
    label='Success',
    color='#4CAF50'
)
bars2 = plt.bar(
    [i + width/2 for i in x],
    outcome_by_phase['Failure'],
    width,
    label='Failure',
    color='#F44336'
)

plt.title('Success vs Failure by Phase', fontsize=16, fontweight='bold')
plt.xlabel('Trial Phase', fontsize=13)
plt.ylabel('Number of Trials', fontsize=13)
plt.xticks(x, outcome_by_phase.index)
plt.legend()
plt.tight_layout()
plt.savefig('../data/chart2_success_failure.png')
plt.show()
print("Chart 2 saved!")

# ============================================================
# STEP 5: CHART 3 - Top 10 Diseases Being Studied
# ============================================================
print("\nCreating Chart 3: Top 10 Conditions...")

# Split conditions (some rows have multiple conditions separated by comma)
# explode() splits them into separate rows
conditions = df['Conditions'].str.split(',').explode()

# Strip extra spaces and count
conditions = conditions.str.strip()
top_conditions = conditions.value_counts().head(10)

plt.figure(figsize=(12, 7))
bars = plt.barh(
    top_conditions.index[::-1],   # reverse so highest is on top
    top_conditions.values[::-1],
    color='#9C27B0'
)

plt.title('Top 10 Conditions Being Studied', fontsize=16, fontweight='bold')
plt.xlabel('Number of Trials', fontsize=13)
plt.ylabel('Condition', fontsize=13)
plt.tight_layout()
plt.savefig('../data/chart3_top_conditions.png')
plt.show()
print("Chart 3 saved!")

# ============================================================
# STEP 6: CHART 4 - Top 10 Organizations Running Trials
# ============================================================
print("\nCreating Chart 4: Top 10 Organizations...")

top_orgs = df['Organization Full Name'].value_counts().head(10)

plt.figure(figsize=(12, 7))
plt.barh(
    top_orgs.index[::-1],
    top_orgs.values[::-1],
    color='#FF5722'
)

plt.title('Top 10 Organizations Running Clinical Trials',
          fontsize=16, fontweight='bold')
plt.xlabel('Number of Trials', fontsize=13)
plt.ylabel('Organization', fontsize=13)
plt.tight_layout()
plt.savefig('../data/chart4_top_organizations.png')
plt.show()
print("Chart 4 saved!")

# ============================================================
# STEP 7: CHART 5 - Organization Class Distribution
# INDUSTRY vs OTHER vs NIH etc
# ============================================================
print("\nCreating Chart 5: Organization Class...")

org_class = df['Organization Class'].value_counts()

plt.figure()
plt.pie(
    org_class.values,
    labels=org_class.index,
    autopct='%1.1f%%',    # show percentage on each slice
    startangle=90,
    colors=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
)

plt.title('Trial Sponsors by Organization Type',
          fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../data/chart5_org_class.png')
plt.show()
print("Chart 5 saved!")

# ============================================================
# STEP 8: KEY STATISTICS SUMMARY
# These numbers will directly feed into our simulation engine!
# ============================================================
print("\n" + "=" * 60)
print("KEY STATISTICS FOR SIMULATION ENGINE")
print("=" * 60)

for phase in ['Phase 1', 'Phase 2', 'Phase 3']:
    phase_df = df_known[df_known['Phase'] == phase]
    total = len(phase_df)
    success = len(phase_df[phase_df['Outcome'] == 1])
    failure = len(phase_df[phase_df['Outcome'] == 0])
    rate = round(success / total * 100, 1)
    print(f"\n{phase}:")
    print(f"  Total trials    : {total:,}")
    print(f"  Successful      : {success:,}")
    print(f"  Failed          : {failure:,}")
    print(f"  Success Rate    : {rate}%")

print("\n" + "=" * 60)
print("EDA COMPLETE! All charts saved to data folder!")
print("=" * 60)