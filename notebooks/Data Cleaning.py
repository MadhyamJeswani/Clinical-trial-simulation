# ============================================================
# CLINICAL TRIAL SIMULATION - CAPGEMINI INTERNSHIP 2026
# File: data_cleaning.py
# Purpose: Clean and filter the clinical trials dataset
# Author: Madhyam
# ============================================================

# ============================================================
# STEP 1: Import libraries
# ============================================================
import pandas as pd
import numpy as np

# ============================================================
# STEP 2: Load the raw dataset
# ============================================================
print("Loading dataset...")
df = pd.read_csv('../data/clin_trials.csv', low_memory=False)
print(f"Original dataset size: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# STEP 3: Keep only INTERVENTIONAL studies
# These are actual drug trials where patients receive treatment
# OBSERVATIONAL studies just watch patients - not useful for us
# ============================================================
df = df[df['Study Type'] == 'INTERVENTIONAL']
print(f"\nAfter keeping only INTERVENTIONAL trials: {df.shape[0]} rows")

# ============================================================
# STEP 4: Keep only Phase 1, 2, and 3 trials
# We don't need Phase 4 or unknown phases for our simulation
# ============================================================
phases_we_want = ['PHASE1', 'PHASE2', 'PHASE3',
                  'PHASE1, PHASE2', 'PHASE2, PHASE3',
                  'EARLY_PHASE1']

df = df[df['Phases'].isin(phases_we_want)]
print(f"After keeping only Phase 1/2/3: {df.shape[0]} rows")

# ============================================================
# STEP 5: Keep only useful columns
# We don't need ALL 17 columns - just the ones relevant
# to our simulation
# ============================================================
columns_to_keep = [
    'Organization Full Name',  # who is running the trial
    'Organization Class',      # INDUSTRY, OTHER, NIH etc
    'Brief Title',             # short name of trial
    'Overall Status',          # COMPLETED, TERMINATED etc
    'Start Date',              # when trial started
    'Standard Age',            # ADULT, CHILD etc
    'Conditions',              # disease being studied
    'Primary Purpose',         # TREATMENT, PREVENTION etc
    'Interventions',           # drug name
    'Phases',                  # PHASE1, PHASE2, PHASE3
    'Outcome Measure'          # what was measured
]

df = df[columns_to_keep]
print(f"\nAfter keeping only useful columns: {df.shape[1]} columns")

# ============================================================
# STEP 6: Handle missing values
# Some cells are empty - we need to fill or remove them
# ============================================================

# Fill missing Standard Age with 'ADULT' (most common value)
df['Standard Age'] = df['Standard Age'].fillna('ADULT')

# Fill missing Outcome Measure with 'Not Specified'
df['Outcome Measure'] = df['Outcome Measure'].fillna('Not Specified')

# Drop any remaining rows with missing values
df = df.dropna()
print(f"After handling missing values: {df.shape[0]} rows")

# ============================================================
# STEP 7: Create a simplified Phase column
# Right now phases look like 'PHASE1, PHASE2'
# We want simple labels: 'Phase 1', 'Phase 2', 'Phase 3'
# ============================================================

def simplify_phase(phase):
    # This is a function - it takes a phase value and simplifies it
    if 'PHASE3' in str(phase):
        return 'Phase 3'
    elif 'PHASE2' in str(phase):
        return 'Phase 2'
    elif 'PHASE1' in str(phase) or 'EARLY_PHASE1' in str(phase):
        return 'Phase 1'
    else:
        return 'Unknown'

# Apply the function to every row in the Phases column
df['Phase'] = df['Phases'].apply(simplify_phase)

# Drop the old Phases column - we now have the clean Phase column
df = df.drop(columns=['Phases'])

print(f"\nSimplified Phase distribution:")
print(df['Phase'].value_counts())

# ============================================================
# STEP 8: Create a Success/Failure outcome column
# COMPLETED = Success (1)
# TERMINATED, WITHDRAWN, SUSPENDED = Failure (0)
# ============================================================

def get_outcome(status):
    # Function to convert status to success/failure
    if status == 'COMPLETED':
        return 1   # 1 means SUCCESS
    elif status in ['TERMINATED', 'WITHDRAWN', 'SUSPENDED']:
        return 0   # 0 means FAILURE
    else:
        return -1  # -1 means UNKNOWN (still ongoing etc)

df['Outcome'] = df['Overall Status'].apply(get_outcome)

print(f"\nOutcome Distribution:")
print(df['Outcome'].value_counts())
print("(1=Success, 0=Failure, -1=Unknown/Ongoing)")

# ============================================================
# STEP 9: Save the cleaned dataset
# ============================================================
df.to_csv('../data/clin_trials_cleaned.csv', index=False)
print(f"\nClean dataset saved!")
print(f"Final size: {df.shape[0]} rows, {df.shape[1]} columns")

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETE!")
print("=" * 60)
# ============================================================
# STEP 10: Verify the cleaned dataset
# ============================================================
print("\nLoading cleaned dataset to verify...")
df_clean = pd.read_csv('../data/clin_trials_cleaned.csv')

print(f"Clean dataset shape: {df_clean.shape}")
print(f"\nFirst 3 rows:")
print(df_clean.head(3))
print(f"\nPhase distribution:")
print(df_clean['Phase'].value_counts())
print(f"\nOutcome distribution:")
print(df_clean['Outcome'].value_counts())