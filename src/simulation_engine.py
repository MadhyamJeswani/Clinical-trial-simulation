# ============================================================
# CLINICAL TRIAL SIMULATION - CAPGEMINI INTERNSHIP 2026
# File: simulation_engine.py
# Purpose: Upgraded simulation engine v3
#          - Uses real historical priors
#          - Integrates synthetic patient generator
#          - Simulates efficacy scores
#          - Phase learnings carry forward
#          - Threshold based GO/NO-GO
# Author: Madhyam
# ============================================================

import random
import numpy as np
import pandas as pd
import json
from scipy import stats
from patient_generator import PatientGenerator
import os
random.seed(42)
np.random.seed(42)

print("✅ Simulation engine v3 libraries imported!")

# ============================================================
# STEP 1: Load real historical priors
# ============================================================
def load_priors(disease='general'):
    """Load real data-driven priors from JSON."""
    try:
        # Find data folder relative to this file's location
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, '..', 'data', 'simulation_priors.json')
        with open(data_path, 'r') as f:
            all_priors = json.load(f)
        priors = all_priors.get(
            disease,
            all_priors['general']
        )
        print(f"✅ Loaded real priors for: {disease}")
        return priors
    except FileNotFoundError:
        print("⚠️ Priors file not found. "
              "Using defaults.")
        return None

# ============================================================
# STEP 2: Safety thresholds
# ============================================================
SAFETY_THRESHOLDS = {
    'Phase 1': {
        'max_ae_rate': 0.35,
        'max_dropout_rate': 0.25,
        'min_response_rate': 0.20,
        'ae_dropout_multiplier': 2.0
    },
    'Phase 2': {
        'max_ae_rate': 0.25,
        'max_dropout_rate': 0.30,
        'min_response_rate': 0.30,
        'ae_dropout_multiplier': 1.8
    },
    'Phase 3': {
        'max_ae_rate': 0.15,
        'max_dropout_rate': 0.20,
        'min_response_rate': 0.40,
        'ae_dropout_multiplier': 1.5
    }
}

# ============================================================
# STEP 3: PhaseLearnings class
# Carries insights between phases
# ============================================================
class PhaseLearnings:

    def __init__(self):
        self.previous_ae_rate = None
        self.previous_dropout_rate = None
        self.previous_response_rate = None
        self.adjustment_factor = 1.0
        self.warnings = []
        self.phase_history = []

    def update(self, ae_rate, dropout_rate,
               response_rate, phase_name):
        self.previous_ae_rate = ae_rate
        self.previous_dropout_rate = dropout_rate
        self.previous_response_rate = response_rate
        self.warnings = []

        # Store phase history
        self.phase_history.append({
            'phase': phase_name,
            'ae_rate': ae_rate,
            'dropout_rate': dropout_rate,
            'response_rate': response_rate
        })

        print(f"\n  📚 LEARNINGS FROM {phase_name}:")

        # AE adjustment
        if ae_rate > 0.25:
            self.adjustment_factor = 1.3
            w = (f"⚠️  High AE ({ae_rate*100:.1f}%) "
                 f"→ Phase {len(self.phase_history)+1} "
                 f"increases monitoring")
            self.warnings.append(w)
            print(f"  {w}")
        elif ae_rate < 0.10:
            self.adjustment_factor = 0.85
            print(f"  ✅ Low AE ({ae_rate*100:.1f}%) "
                  f"→ Proceeding with standard monitoring")
        else:
            self.adjustment_factor = 1.0
            print(f"  ✅ Normal AE ({ae_rate*100:.1f}%) "
                  f"→ Proceeding normally")

        # Response rate warning
        if response_rate < 0.40:
            w = (f"⚠️  Low response rate "
                 f"({response_rate*100:.1f}%) "
                 f"→ Consider dose adjustment")
            self.warnings.append(w)
            print(f"  {w}")

        # Dropout warning
        if dropout_rate > 0.20:
            w = (f"⚠️  High dropout "
                 f"({dropout_rate*100:.1f}%) "
                 f"→ Next phase enrolls extra patients")
            self.warnings.append(w)
            print(f"  {w}")

# ============================================================
# STEP 4: Core simulation function
# ============================================================
def simulate_phase(phase_name, drug_name,
                   disease_category, learnings,
                   priors):
    """
    Simulates one complete clinical trial phase.
    Now uses:
    - Real historical priors
    - Synthetic patient generator
    - Efficacy simulation
    - Phase learnings
    """

    print(f"\n{'='*60}")
    print(f"  SIMULATING {phase_name.upper()}")
    print(f"  Drug: {drug_name} | "
          f"Disease: {disease_category}")
    print(f"{'='*60}")

    thresholds = SAFETY_THRESHOLDS[phase_name]
    phase_priors = priors.get(phase_name, {})

    # Get data-driven rates
    base_ae_rate = phase_priors.get(
        'ae_rate', 0.25)
    base_dropout_rate = phase_priors.get(
        'dropout_rate', 0.10)
    base_success_rate = phase_priors.get(
        'success_rate', 0.75)

    # Apply learnings adjustment
    adjusted_ae_rate = (base_ae_rate *
                        learnings.adjustment_factor)
    adjusted_ae_rate = min(adjusted_ae_rate, 0.45)

    # Patient enrollment numbers per phase
    enrollment_ranges = {
        'Phase 1': (20, 80),
        'Phase 2': (100, 300),
        'Phase 3': (1000, 3000)
    }

    min_p, max_p = enrollment_ranges[phase_name]

    # Extra enrollment if previous dropout was high
    extra = 0
    if (learnings.previous_dropout_rate and
            learnings.previous_dropout_rate > 0.20):
        extra = int(min_p * 0.20)
        print(f"  📈 +{extra} extra patients "
              f"(high previous dropout)")

    num_patients = random.randint(
        min_p, max_p) + extra

    print(f"  Enrolling {num_patients} patients...")
    print(f"  AE rate (data-driven): "
          f"{adjusted_ae_rate*100:.1f}%")
    print(f"  Success rate prior: "
          f"{base_success_rate*100:.1f}%")

    # ── Generate synthetic patients ──────────────────
    generator = PatientGenerator(
        disease_category=disease_category,
        phase=phase_name
    )
    patients = generator.generate_cohort(num_patients)

    # ── Simulate each patient's journey ─────────────
    for patient in patients:

        # 1. Simulate adverse event
        if random.random() < adjusted_ae_rate:
            patient.had_adverse_event = True
            patient.ae_severity = (
                1 if random.random() < 0.70 else 2)
            # Add AE description
            ae_types = {
                'cancer': ['Nausea', 'Fatigue',
                           'Neutropenia',
                           'Alopecia', 'Vomiting'],
                'infectious': ['Fever', 'Rash',
                               'Liver enzyme elevation'],
                'cardiovascular': ['Hypotension',
                                   'Bradycardia',
                                   'Oedema'],
                'general': ['Headache', 'Dizziness',
                            'Fatigue', 'Nausea']
            }
            ae_list = ae_types.get(
                disease_category,
                ae_types['general'])
            patient.ae_description = random.choice(
                ae_list)

        # 2. Simulate dropout (AE influences dropout!)
        if patient.ae_severity == 2:
            eff_dropout = (base_dropout_rate *
                          thresholds[
                              'ae_dropout_multiplier'
                          ] * 2)
        elif patient.ae_severity == 1:
            eff_dropout = (base_dropout_rate *
                          thresholds[
                              'ae_dropout_multiplier'
                          ])
        else:
            eff_dropout = base_dropout_rate

        eff_dropout = min(eff_dropout, 0.95)

        if random.random() < eff_dropout:
            patient.dropped_out = True
            patient.enrolled = False
        else:
            patient.completed = True

    # ── Calculate phase results ──────────────────────
    total = len(patients)
    dropouts = sum(1 for p in patients
                   if p.dropped_out)
    completed = sum(1 for p in patients
                    if p.completed)
    ae_count = sum(1 for p in patients
                   if p.had_adverse_event)
    mild_ae = sum(1 for p in patients
                  if p.ae_severity == 1)
    severe_ae = sum(1 for p in patients
                    if p.ae_severity == 2)

    # Efficacy calculations (NEW!)
    completers = [p for p in patients
                  if p.completed]
    if completers:
        response_rate = np.mean([
            p.drug_response_score
            for p in completers
        ])
        endpoint_rate = np.mean([
            int(p.endpoint_achieved)
            for p in completers
        ])
        avg_baseline = np.mean([
            p.baseline_score for p in completers
        ])
        avg_followup = np.mean([
            p.followup_score for p in completers
        ])

        # Calculate p-value (t-test baseline vs followup)
        if len(completers) > 1:
            baselines = [p.baseline_score
                         for p in completers]
            followups = [p.followup_score
                         for p in completers]
            t_stat, p_value = stats.ttest_rel(
                baselines, followups)
            p_value = round(p_value, 4)
        else:
            p_value = 1.0
    else:
        response_rate = 0.0
        endpoint_rate = 0.0
        avg_baseline = 0.0
        avg_followup = 0.0
        p_value = 1.0

    # Patient demographics summary
    ages = [p.age for p in patients]
    females = sum(1 for p in patients
                  if p.gender == 'Female')

    actual_ae_rate = round(ae_count / total, 3)
    actual_dropout_rate = round(dropouts / total, 3)
    actual_response_rate = round(response_rate, 3)

    # ── Print results ────────────────────────────────
    print(f"\n  📊 PHASE RESULTS:")
    print(f"  Enrolled        : {total}")
    print(f"  Completed       : {completed}")
    print(f"  Dropped out     : {dropouts} "
          f"({actual_dropout_rate*100:.1f}%)")
    print(f"  Adverse Events  : {ae_count} "
          f"({actual_ae_rate*100:.1f}%)")
    print(f"    Mild AE       : {mild_ae}")
    print(f"    Severe AE     : {severe_ae}")
    print(f"\n  📈 EFFICACY RESULTS:")
    print(f"  Response Rate   : "
          f"{actual_response_rate*100:.1f}%")
    print(f"  Endpoint Rate   : "
          f"{endpoint_rate*100:.1f}%")
    print(f"  Baseline Score  : {avg_baseline:.1f}")
    print(f"  Followup Score  : {avg_followup:.1f}")
    print(f"  Improvement     : "
          f"{((avg_baseline-avg_followup)/avg_baseline*100):.1f}%")
    print(f"  P-value         : {p_value} "
          f"({'✅ Significant' if p_value < 0.05 else '❌ Not significant'})")
    print(f"\n  👥 DEMOGRAPHICS:")
    print(f"  Age mean        : {np.mean(ages):.1f} yrs")
    print(f"  Female/Male     : {females}/{total-females}")

    # ── GO/NO-GO Decision ────────────────────────────
    print(f"\n  🎯 GO/NO-GO ANALYSIS:")
    print(f"  AE rate      : {actual_ae_rate*100:.1f}% "
          f"(max: {thresholds['max_ae_rate']*100:.0f}%)")
    print(f"  Dropout rate : "
          f"{actual_dropout_rate*100:.1f}% "
          f"(max: {thresholds['max_dropout_rate']*100:.0f}%)")
    print(f"  Response rate: "
          f"{actual_response_rate*100:.1f}% "
          f"(min: "
          f"{thresholds['min_response_rate']*100:.0f}%)")

    # Decision logic
    if actual_ae_rate > thresholds['max_ae_rate']:
        decision = "NO-GO"
        reason = (f"AE rate {actual_ae_rate*100:.1f}% "
                  f"exceeds threshold "
                  f"{thresholds['max_ae_rate']*100:.0f}%")

    elif (actual_dropout_rate >
          thresholds['max_dropout_rate']):
        decision = "NO-GO"
        reason = (f"Dropout {actual_dropout_rate*100:.1f}%"
                  f" exceeds threshold "
                  f"{thresholds['max_dropout_rate']*100:.0f}%")

    elif (actual_response_rate <
          thresholds['min_response_rate'] and
          phase_name != 'Phase 1'):
        decision = "NO-GO"
        reason = (f"Response rate "
                  f"{actual_response_rate*100:.1f}% "
                  f"below minimum "
                  f"{thresholds['min_response_rate']*100:.0f}%")

    else:
        # Probability based on data-driven success rate
        if random.random() < base_success_rate:
            decision = "GO"
            reason = ("All thresholds passed. "
                      "Trial outcome positive.")
        else:
            decision = "NO-GO"
            reason = ("Thresholds passed but "
                      "trial outcome negative.")

    if decision == "GO":
        print(f"\n  ✅ GO! {phase_name} PASSED!")
    else:
        print(f"\n  ❌ NO-GO! {phase_name} FAILED!")
    print(f"  Reason: {reason}")

    # Update learnings
    learnings.update(
        actual_ae_rate,
        actual_dropout_rate,
        actual_response_rate,
        phase_name
    )

    # ── Return complete results ──────────────────────
    return {
        'phase': phase_name,
        'drug_name': drug_name,
        'disease_category': disease_category,
        'total_enrolled': total,
        'total_completed': completed,
        'total_dropouts': dropouts,
        'dropout_rate_%': round(
            actual_dropout_rate * 100, 1),
        'total_ae': ae_count,
        'mild_ae': mild_ae,
        'severe_ae': severe_ae,
        'ae_rate_%': round(actual_ae_rate * 100, 1),
        'response_rate_%': round(
            actual_response_rate * 100, 1),
        'endpoint_rate_%': round(
            endpoint_rate * 100, 1),
        'avg_baseline_score': round(avg_baseline, 1),
        'avg_followup_score': round(avg_followup, 1),
        'improvement_%': round(
            (avg_baseline - avg_followup) /
            avg_baseline * 100
            if avg_baseline > 0 else 0, 1),
        'p_value': p_value,
        'statistically_significant': p_value < 0.05,
        'age_mean': round(np.mean(ages), 1),
        'female_count': females,
        'male_count': total - females,
        'decision': decision,
        'reason': reason,
        'warnings': learnings.warnings.copy()
    }

# ============================================================
# STEP 5: Full trial simulation
# ============================================================
def run_full_trial(drug_name,
                   disease_category='general'):
    """
    Runs complete Phase 1 → 2 → 3 simulation.
    Uses real priors + synthetic patients + efficacy.
    """

    print(f"\n{'#'*60}")
    print(f"  CLINICAL TRIAL SIMULATION v3")
    print(f"  Drug: {drug_name}")
    print(f"  Disease: {disease_category}")
    print(f"{'#'*60}")

    # Load real priors for this disease
    priors = load_priors(disease_category)
    if priors is None:
        priors = load_priors('general')

    all_results = []
    learnings = PhaseLearnings()

    for phase in ['Phase 1', 'Phase 2', 'Phase 3']:
        result = simulate_phase(
            phase_name=phase,
            drug_name=drug_name,
            disease_category=disease_category,
            learnings=learnings,
            priors=priors
        )
        all_results.append(result)

        if result['decision'] == 'NO-GO':
            print(f"\n🛑 TRIAL STOPPED at {phase}")
            print(f"   Reason: {result['reason']}")
            break

    # Final verdict
    if (len(all_results) == 3 and
            all_results[-1]['decision'] == 'GO'):
        print(f"\n🏆 TRIAL COMPLETE!")
        print(f"   Drug '{drug_name}' passed "
              f"all 3 phases!")
        print(f"   Recommended for regulatory "
              f"approval!")
    else:
        stopped = all_results[-1]['phase']
        print(f"\n💊 '{drug_name}' discontinued "
              f"at {stopped}")

    # Summary table
    results_df = pd.DataFrame(all_results)
    print(f"\n📋 TRIAL SUMMARY:")
    cols = ['phase', 'total_enrolled',
            'ae_rate_%', 'dropout_rate_%',
            'response_rate_%', 'p_value',
            'decision']
    print(results_df[cols].to_string(index=False))

    return all_results

# ============================================================
# STEP 6: Run test simulations
# ============================================================
if __name__ == "__main__":

    print("🔬 CLINICAL TRIAL SIMULATION SYSTEM v3")
    print("   Capgemini Internship 2026 — Madhyam")
    print("   Real priors + Synthetic patients "
          "+ Efficacy")

    # Test 1: Cancer drug
    results_cancer = run_full_trial(
        drug_name="OncoCure-X1",
        disease_category="cancer"
    )

    print("\n" + "="*60)

    # Test 2: Infectious disease drug
    results_infect = run_full_trial(
        drug_name="InfectoShield-V2",
        disease_category="infectious"
    )

    print("\n" + "="*60)

    # Save results
    all_results = pd.DataFrame(
        results_cancer + results_infect
    )
    all_results.to_csv(
        '../data/simulation_results_v3.csv',
        index=False
    )
    print("\n✅ Results saved to "
          "simulation_results_v3.csv!")

    print("\n" + "="*60)
    print("SIMULATION ENGINE v3 COMPLETE!")
    print("="*60)