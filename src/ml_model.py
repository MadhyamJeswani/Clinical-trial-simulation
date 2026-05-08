# ============================================================
# CLINICAL TRIAL SIMULATION - CAPGEMINI INTERNSHIP 2026
# File: ml_model.py
# Purpose: Improved ML pipeline with comprehensive
#          feature engineering
# Author: Madhyam
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix,
                             roc_auc_score)
import pickle

print("✅ Libraries imported!")

# ============================================================
# STEP 1: Load raw dataset
# We load RAW this time to access all original columns
# ============================================================
print("\nLoading raw dataset...")
df = pd.read_csv('../data/clin_trials.csv', low_memory=False)
print(f"Raw dataset: {df.shape[0]:,} rows")

# Apply same cleaning as before
df = df[df['Study Type'] == 'INTERVENTIONAL']
phases_we_want = ['PHASE1', 'PHASE2', 'PHASE3',
                  'PHASE1, PHASE2', 'PHASE2, PHASE3',
                  'EARLY_PHASE1']
df = df[df['Phases'].isin(phases_we_want)]

# Create Phase column
def simplify_phase(phase):
    if 'PHASE3' in str(phase):
        return 'Phase 3'
    elif 'PHASE2' in str(phase):
        return 'Phase 2'
    elif 'PHASE1' in str(phase) or 'EARLY_PHASE1' in str(phase):
        return 'Phase 1'
    else:
        return 'Unknown'

df['Phase'] = df['Phases'].apply(simplify_phase)

# Create Outcome column
def get_outcome(status):
    if status == 'COMPLETED':
        return 1
    elif status in ['TERMINATED', 'WITHDRAWN', 'SUSPENDED']:
        return 0
    else:
        return -1

df['Outcome'] = df['Overall Status'].apply(get_outcome)
df['Organization Class'] = df['Organization Class'].fillna('UNKNOWN')
df['Primary Purpose'] = df['Primary Purpose'].fillna('Unknown')
df['Standard Age'] = df['Standard Age'].fillna('ADULT')
df['Conditions'] = df['Conditions'].fillna('')
df['Interventions'] = df['Interventions'].fillna('')
df['Organization Full Name'] = df['Organization Full Name'].fillna('')
df['Start Date'] = df['Start Date'].fillna('')

print(f"After cleaning: {df.shape[0]:,} rows")

# ============================================================
# STEP 2: Comprehensive Feature Engineering
# ============================================================
print("\n" + "="*60)
print("COMPREHENSIVE FEATURE ENGINEERING")
print("="*60)

# ── Feature 1: Phase as ordered number ──────────────────────
phase_map = {'Phase 1': 1, 'Phase 2': 2, 'Phase 3': 3}
df['phase_number'] = df['Phase'].map(phase_map).fillna(1)
print("✅ F1:  phase_number")

# ── Feature 2: Expanded Big Pharma list ─────────────────────
# Now includes ALL major pharma from our dataset analysis!
big_pharma_list = [
    # Original 13
    'Pfizer', 'Novartis', 'Roche', 'AstraZeneca',
    'GlaxoSmithKline', 'Merck', 'Eli Lilly',
    'Bristol-Myers', 'Sanofi', 'Boehringer',
    'AbbVie', 'Amgen', 'Johnson',
    # NEW — discovered from dataset analysis
    'Hoffmann-La Roche',   # different name from Roche!
    'Takeda',
    'Bayer',
    'Novo Nordisk',
    'Janssen',             # J&J subsidiary
    'Astellas',
    'Gilead',
    'Eisai',
    'UCB Pharma',
    'Genentech',
    'Celgene',
    'Biogen',
    'Teva',
    'Wyeth',               # Pfizer subsidiary
    'Organon',
    'Regeneron',
    'Moderna',
    'BioNTech',
    'MSD',                 # Merck's international name
    'Daiichi Sankyo',
    'Shire',
    'Alexion',
    'Vertex',
]

df['is_big_pharma'] = df['Organization Full Name'].apply(
    lambda org: int(any(
        p.lower() in str(org).lower()
        for p in big_pharma_list
    ))
)
print(f"✅ F2:  is_big_pharma "
      f"({df['is_big_pharma'].sum():,} trials)")

# ── Feature 3: Top Cancer Centers ───────────────────────────
# These specialized centers have significantly better outcomes
top_cancer_centers = [
    'MD Anderson', 'Anderson Cancer',
    'Memorial Sloan', 'Sloan Kettering',
    'Dana-Farber', 'Dana Farber',
    'Sidney Kimmel',
    'City of Hope',
    'Fred Hutchinson',
    'Roswell Park',
    'Mayo Clinic',
    'Johns Hopkins',
    'Massachusetts General',
]
df['is_top_cancer_center'] = df['Organization Full Name'].apply(
    lambda org: int(any(
        c.lower() in str(org).lower()
        for c in top_cancer_centers
    ))
)
print(f"✅ F3:  is_top_cancer_center "
      f"({df['is_top_cancer_center'].sum():,} trials)")

# ── Feature 4: Disease Category Flags ───────────────────────
# Cancer
df['is_cancer'] = df['Conditions'].str.contains(
    'cancer|tumor|carcinoma|leukemia|lymphoma|melanoma|'
    'neoplasm|oncology|sarcoma|myeloma|glioma',
    case=False, na=False
).astype(int)
print(f"✅ F4:  is_cancer "
      f"({df['is_cancer'].sum():,} trials)")

# Healthy volunteers (Phase 1 safety trials)
df['is_healthy'] = df['Conditions'].str.contains(
    'healthy|volunteer',
    case=False, na=False
).astype(int)
print(f"✅ F5:  is_healthy "
      f"({df['is_healthy'].sum():,} trials)")

# Mental health (high dropout, hard to succeed)
df['is_mental_health'] = df['Conditions'].str.contains(
    'schizophrenia|depression|anxiety|bipolar|'
    'psychiatric|mental|alzheimer|dementia|autism',
    case=False, na=False
).astype(int)
print(f"✅ F6:  is_mental_health "
      f"({df['is_mental_health'].sum():,} trials)")

# Diabetes (very common, well understood)
df['is_diabetes'] = df['Conditions'].str.contains(
    'diabetes|diabetic|insulin|glycemic',
    case=False, na=False
).astype(int)
print(f"✅ F7:  is_diabetes "
      f"({df['is_diabetes'].sum():,} trials)")

# Infectious disease
df['is_infectious'] = df['Conditions'].str.contains(
    'infection|virus|bacterial|HIV|covid|hepatitis|'
    'tuberculosis|malaria|influenza',
    case=False, na=False
).astype(int)
print(f"✅ F8:  is_infectious "
      f"({df['is_infectious'].sum():,} trials)")

# Cardiovascular
df['is_cardiovascular'] = df['Conditions'].str.contains(
    'heart|cardiac|cardiovascular|hypertension|'
    'stroke|coronary|atrial|ventricular',
    case=False, na=False
).astype(int)
print(f"✅ F9:  is_cardiovascular "
      f"({df['is_cardiovascular'].sum():,} trials)")

# Pain (high placebo response = harder to prove drug works)
df['is_pain'] = df['Conditions'].str.contains(
    'pain|analgesic|nociceptive|neuropathic',
    case=False, na=False
).astype(int)
print(f"✅ F10: is_pain "
      f"({df['is_pain'].sum():,} trials)")

# ── Feature 5: Intervention Type ────────────────────────────
df['is_drug'] = df['Interventions'].str.contains(
    'drug|tablet|capsule|oral|injection|dose|mg',
    case=False, na=False
).astype(int)

df['is_biological'] = df['Interventions'].str.contains(
    'vaccine|antibody|biologic|protein|gene|'
    'cell therapy|immunotherapy|monoclonal',
    case=False, na=False
).astype(int)
print("✅ F11: is_drug + is_biological")

# ── Feature 6: Trial Complexity ─────────────────────────────
df['num_conditions'] = df['Conditions'].str.count(',') + 1
print("✅ F12: num_conditions")

# ── Feature 7: Primary Purpose Flags ────────────────────────
df['is_treatment'] = (
    df['Primary Purpose'] == 'TREATMENT'
).astype(int)

df['is_prevention'] = (
    df['Primary Purpose'] == 'PREVENTION'
).astype(int)

df['is_basic_science'] = (
    df['Primary Purpose'] == 'BASIC_SCIENCE'
).astype(int)
print("✅ F13: is_treatment + is_prevention + is_basic_science")

# ── Feature 8: Organization Class Success Rates ─────────────
# From our analysis: FED=86.4%, INDUSTRY=84.8%, OTHER=76.7%
# Encode with ACTUAL success rates instead of arbitrary numbers
org_success_map = {
    'FED':       0.864,
    'INDUSTRY':  0.848,
    'NIH':       0.855,
    'NETWORK':   0.806,
    'OTHER':     0.767,
    'OTHER_GOV': 0.853,
    'UNKNOWN':   0.820,
    'INDIV':     0.823,
}
df['org_success_rate'] = df['Organization Class'].map(
    org_success_map
).fillna(0.820)
print("✅ F14: org_success_rate (actual rates from data!)")

# ── Feature 9: Start Year (without misleading recent flag) ──
df['start_year'] = pd.to_datetime(
    df['Start Date'], errors='coerce'
).dt.year.fillna(2005).astype(int)

# Bin into eras instead of raw year
# This avoids the recent trial bias problem
def get_era(year):
    if year < 2000:
        return 0  # very old
    elif year < 2005:
        return 1  # old
    elif year < 2010:
        return 2  # moderate
    elif year < 2015:
        return 3  # recent
    else:
        return 4  # very recent

df['trial_era'] = df['start_year'].apply(get_era)
print("✅ F15: trial_era (binned — avoids recent bias)")

# ── Encode remaining categorical columns ────────────────────
le = LabelEncoder()
df['age_encoded'] = le.fit_transform(
    df['Standard Age'].astype(str)
)
print("✅ F16: age_encoded")

# ============================================================
# STEP 3: Final feature set
# ============================================================
feature_columns = [
    # Phase
    'phase_number',
    # Organization
    'is_big_pharma',
    'is_top_cancer_center',
    'org_success_rate',
    # Disease category
    'is_cancer',
    'is_healthy',
    'is_mental_health',
    'is_diabetes',
    'is_infectious',
    'is_cardiovascular',
    'is_pain',
    # Intervention
    'is_drug',
    'is_biological',
    # Trial characteristics
    'num_conditions',
    'is_treatment',
    'is_prevention',
    'is_basic_science',
    'trial_era',
    'age_encoded',
]

print(f"\n✅ Total features: {len(feature_columns)}")

# ============================================================
# STEP 4: Keep known outcomes only
# ============================================================
df_ml = df[df['Outcome'] != -1].copy()
print(f"\nRows with known outcomes: {df_ml.shape[0]:,}")
print(f"Success rate: {df_ml['Outcome'].mean()*100:.1f}%")

# Show success rates by key features
print("\nSuccess rate by Phase:")
print(df_ml.groupby('Phase')['Outcome'].mean().round(3))

print("\nSuccess rate by Big Pharma:")
print(df_ml.groupby('is_big_pharma')['Outcome'].mean().round(3))

print("\nSuccess rate by Cancer:")
print(df_ml.groupby('is_cancer')['Outcome'].mean().round(3))

# ============================================================
# STEP 5: Define X and y
# ============================================================
X = df_ml[feature_columns]
y = df_ml['Outcome']

print(f"\nFeature matrix: {X.shape}")

# ============================================================
# STEP 6: Train/Test Split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Training: {X_train.shape[0]:,} | "
      f"Testing: {X_test.shape[0]:,}")

# ============================================================
# STEP 7: Logistic Regression
# ============================================================
print("\n" + "="*60)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*60)

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_prob = lr_model.predict_proba(X_test)[:, 1]
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_prob)

print(f"Accuracy : {lr_accuracy*100:.2f}%")
print(f"ROC-AUC  : {lr_auc:.3f}")
print(classification_report(
    y_test, lr_pred,
    target_names=['Failure', 'Success'],
    zero_division=0
))

# ============================================================
# STEP 8: Random Forest
# ============================================================
print("\n" + "="*60)
print("MODEL 2: RANDOM FOREST")
print("="*60)
print("Training 100 trees...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_prob)

print(f"Accuracy : {rf_accuracy*100:.2f}%")
print(f"ROC-AUC  : {rf_auc:.3f}")
print(classification_report(
    y_test, rf_pred,
    target_names=['Failure', 'Success'],
    zero_division=0
))

# ============================================================
# STEP 9: Charts
# ============================================================
print("\nGenerating charts...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Chart 1: Accuracy
models = ['Logistic\nRegression', 'Random\nForest']
accuracies = [lr_accuracy*100, rf_accuracy*100]
aucs = [lr_auc, rf_auc]
colors = ['#2196F3', '#4CAF50']

bars = axes[0].bar(models, accuracies, color=colors, width=0.5)
for bar, acc in zip(bars, accuracies):
    axes[0].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.5,
        f'{acc:.1f}%',
        ha='center', fontweight='bold', fontsize=13
    )
axes[0].axhline(y=70, color='red',
                linestyle='--', label='Target 70%')
axes[0].set_title('Accuracy Comparison',
                  fontweight='bold', fontsize=13)
axes[0].set_ylabel('Accuracy %')
axes[0].set_ylim(0, 100)
axes[0].legend()

# Chart 2: AUC
bars2 = axes[1].bar(models, aucs, color=colors, width=0.5)
for bar, auc in zip(bars2, aucs):
    axes[1].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.005,
        f'{auc:.3f}',
        ha='center', fontweight='bold', fontsize=13
    )
axes[1].axhline(y=0.7, color='red',
                linestyle='--', label='Good: 0.7')
axes[1].set_title('ROC-AUC Comparison',
                  fontweight='bold', fontsize=13)
axes[1].set_ylabel('ROC-AUC Score')
axes[1].set_ylim(0, 1)
axes[1].legend()

# Chart 3: Feature Importance
rf_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

axes[2].barh(
    rf_importance['Feature'],
    rf_importance['Importance'],
    color='#4CAF50'
)
axes[2].set_title('Feature Importance\n(Random Forest)',
                  fontweight='bold', fontsize=13)
axes[2].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('../data/ml_comparison_v2.png')
plt.show()
print("✅ Charts saved!")

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_lr = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm_lr, annot=True, fmt='d',
            cmap='Blues', ax=axes[0],
            xticklabels=['Pred Fail', 'Pred Success'],
            yticklabels=['Act Fail', 'Act Success'])
axes[0].set_title(
    f'Logistic Regression\nAccuracy: {lr_accuracy*100:.1f}%',
    fontweight='bold')

cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d',
            cmap='Greens', ax=axes[1],
            xticklabels=['Pred Fail', 'Pred Success'],
            yticklabels=['Act Fail', 'Act Success'])
axes[1].set_title(
    f'Random Forest\nAccuracy: {rf_accuracy*100:.1f}%',
    fontweight='bold')

plt.tight_layout()
plt.savefig('../data/confusion_matrices_v2.png')
plt.show()
print("✅ Confusion matrices saved!")

# ============================================================
# STEP 10: Success rate analysis charts
# ============================================================
print("\nGenerating insight charts...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Success rate by phase
phase_success = df_ml.groupby(
    'Phase')['Outcome'].mean() * 100
axes[0].bar(phase_success.index,
            phase_success.values,
            color=['#2196F3', '#4CAF50', '#FF9800'])
axes[0].set_title('Real Success Rate by Phase',
                  fontweight='bold')
axes[0].set_ylabel('Success Rate %')
for i, (phase, rate) in enumerate(phase_success.items()):
    axes[0].text(i, rate + 0.5, f'{rate:.1f}%',
                ha='center', fontweight='bold')

# Success rate by org class
org_success = df_ml.groupby(
    'Organization Class')['Outcome'].mean() * 100
org_success = org_success.sort_values(ascending=True)
axes[1].barh(org_success.index,
             org_success.values,
             color='#9C27B0')
axes[1].set_title('Success Rate by Org Class',
                  fontweight='bold')
axes[1].set_xlabel('Success Rate %')

# Success rate by disease category
disease_cats = {
    'Cancer':          df_ml['is_cancer'].mean(),
    'Healthy Vol.':    df_ml['is_healthy'].mean(),
    'Mental Health':   df_ml['is_mental_health'].mean(),
    'Diabetes':        df_ml['is_diabetes'].mean(),
    'Infectious':      df_ml['is_infectious'].mean(),
    'Cardiovascular':  df_ml['is_cardiovascular'].mean(),
    'Pain':            df_ml['is_pain'].mean(),
}

# Calculate success rate per disease category
disease_success = {}
for name, col in [
    ('Cancer', 'is_cancer'),
    ('Healthy', 'is_healthy'),
    ('Mental', 'is_mental_health'),
    ('Diabetes', 'is_diabetes'),
    ('Infectious', 'is_infectious'),
    ('Cardiovasc.', 'is_cardiovascular'),
    ('Pain', 'is_pain'),
]:
    subset = df_ml[df_ml[col] == 1]
    if len(subset) > 0:
        disease_success[name] = subset['Outcome'].mean() * 100

ds = pd.Series(disease_success).sort_values(ascending=True)
axes[2].barh(ds.index, ds.values, color='#FF5722')
axes[2].set_title('Success Rate by Disease Category',
                  fontweight='bold')
axes[2].set_xlabel('Success Rate %')

plt.tight_layout()
plt.savefig('../data/success_rate_analysis.png')
plt.show()
print("✅ Insight charts saved!")

# ============================================================
# STEP 11: Pick and save best model
# ============================================================
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"\nLogistic Regression → "
      f"Accuracy: {lr_accuracy*100:.2f}% | AUC: {lr_auc:.3f}")
print(f"Random Forest       → "
      f"Accuracy: {rf_accuracy*100:.2f}% | AUC: {rf_auc:.3f}")

best_model = rf_model if rf_auc > lr_auc else lr_model
best_name = "Random Forest" if rf_auc > lr_auc else "LR"
best_acc = rf_accuracy if rf_auc > lr_auc else lr_accuracy
best_auc = rf_auc if rf_auc > lr_auc else lr_auc

print(f"\n🏆 BEST MODEL: {best_name}")
print(f"   Accuracy : {best_acc*100:.2f}%")
print(f"   ROC-AUC  : {best_auc:.3f}")

if best_acc >= 0.70:
    print("   ✅ BRD Target ACHIEVED!")
else:
    print("   ⚠️  Honest limitation — dataset lacks"
          " drug-specific features")
    print("   📝 Documented for case study")

# Save
with open('../data/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

model_info = {
    'model_name': best_name,
    'accuracy': best_acc,
    'roc_auc': best_auc,
    'features': feature_columns
}
with open('../data/model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print("\n✅ Model saved!")
print("✅ Model info saved!")

print("\n" + "="*60)
print("ML PIPELINE v3 COMPLETE!")
print("="*60)