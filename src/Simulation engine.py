# ============================================================
# CLINICAL TRIAL SIMULATION - CAPGEMINI INTERNSHIP 2026
# File: simulation_engine.py
# Purpose: Improved Clinical Trial Simulation Engine
#          - Threshold based go/no-go decisions
#          - Adverse events influence dropouts
#          - Phase learnings carry forward
# Author: Madhyam
# ============================================================

# ============================================================
# STEP 1: Import libraries
# ============================================================
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# ============================================================
# STEP 2: Phase Parameters
# These are BASE parameters - they will be adjusted
# dynamically based on what happens in previous phases!
# ============================================================
PHASE_PARAMS = {
    'Phase 1': {
        'min_patients': 20,
        'max_patients': 80,
        'success_rate': 0.63,
        'base_dropout_rate': 0.05,
        'base_adverse_event_rate': 0.30,
        # THRESHOLDS - if exceeded, automatic NO-GO
        'max_allowed_adverse_rate': 0.35,  # 35% max AE allowed
        'max_allowed_dropout_rate': 0.25,  # 25% max dropout allowed
        # How much an adverse event increases dropout chance
        'ae_dropout_multiplier': 2.0
    },
    'Phase 2': {
        'min_patients': 100,
        'max_patients': 300,
        'success_rate': 0.77,
        'base_dropout_rate': 0.15,
        'base_adverse_event_rate': 0.20,
        'max_allowed_adverse_rate': 0.25,
        'max_allowed_dropout_rate': 0.30,
        'ae_dropout_multiplier': 1.8
    },
    'Phase 3': {
        'min_patients': 1000,
        'max_patients': 3000,
        'success_rate': 0.84,
        'base_dropout_rate': 0.10,
        'base_adverse_event_rate': 0.10,
        'max_allowed_adverse_rate': 0.15,
        'max_allowed_dropout_rate': 0.20,
        'ae_dropout_multiplier': 1.5
    }
}

# ============================================================
# STEP 3: Patient Class (Improved)
# Now adverse events directly influence dropout probability
# ============================================================
class Patient:

    def __init__(self, patient_id, phase):
        self.patient_id = patient_id
        self.phase = phase
        self.age = random.randint(18, 75)
        self.enrolled = True
        self.dropped_out = False
        self.had_adverse_event = False
        self.completed = False
        # NEW: severity of adverse event (0=none, 1=mild, 2=severe)
        self.ae_severity = 0

    def simulate_adverse_event(self, adverse_event_rate):
        # Simulate whether patient has adverse event
        if random.random() < adverse_event_rate:
            self.had_adverse_event = True
            # Randomly assign severity
            # 70% chance mild, 30% chance severe
            self.ae_severity = 1 if random.random() < 0.70 else 2

    def simulate_dropout(self, base_dropout_rate, ae_dropout_multiplier):
        # NEW LOGIC: adverse events INCREASE dropout probability!
        # If patient had severe AE → much higher dropout chance
        # If patient had mild AE → slightly higher dropout chance
        # If no AE → normal dropout chance

        if self.ae_severity == 2:
            # Severe adverse event → multiply dropout rate significantly
            effective_dropout_rate = (base_dropout_rate *
                                     ae_dropout_multiplier * 2)
        elif self.ae_severity == 1:
            # Mild adverse event → multiply dropout rate slightly
            effective_dropout_rate = (base_dropout_rate *
                                     ae_dropout_multiplier)
        else:
            # No adverse event → normal dropout rate
            effective_dropout_rate = base_dropout_rate

        # Cap dropout rate at 95% maximum
        # (some patients always stay no matter what)
        effective_dropout_rate = min(effective_dropout_rate, 0.95)

        if random.random() < effective_dropout_rate:
            self.dropped_out = True
            self.enrolled = False

    def __repr__(self):
        severity_text = ['None', 'Mild', 'Severe'][self.ae_severity]
        return (f"Patient {self.patient_id} | "
                f"Age: {self.age} | "
                f"AE: {severity_text} | "
                f"Dropped Out: {self.dropped_out}")

# ============================================================
# STEP 4: Phase Learnings class
# This carries important information from one phase to the next
# ============================================================
class PhaseLearnings:
    # This class acts as a MEMORY between phases
    # Whatever happened in Phase 1 is remembered and
    # used to adjust Phase 2 and Phase 3

    def __init__(self):
        self.previous_ae_rate = None      # AE rate from last phase
        self.previous_dropout_rate = None # dropout rate from last phase
        self.adjustment_factor = 1.0      # how much to adjust next phase
        self.warnings = []                # list of warnings to carry forward

    def update(self, ae_rate, dropout_rate, phase_name):
        # Called after each phase completes
        # Updates the learnings for next phase

        self.previous_ae_rate = ae_rate
        self.previous_dropout_rate = dropout_rate
        self.warnings = []

        print(f"\n  📚 PHASE LEARNINGS FROM {phase_name}:")

        # If previous phase had HIGH adverse events
        # → next phase will be MORE CAUTIOUS
        if ae_rate > 0.25:
            self.adjustment_factor = 1.3
            warning = (f"⚠️  High AE rate ({ae_rate*100:.1f}%) detected! "
                      f"Next phase will increase monitoring intensity.")
            self.warnings.append(warning)
            print(f"  {warning}")

        # If previous phase had LOW adverse events
        # → next phase can be slightly more relaxed
        elif ae_rate < 0.10:
            self.adjustment_factor = 0.8
            msg = (f"✅ Low AE rate ({ae_rate*100:.1f}%) detected! "
                  f"Next phase proceeds with standard monitoring.")
            print(f"  {msg}")

        else:
            self.adjustment_factor = 1.0
            msg = (f"✅ Normal AE rate ({ae_rate*100:.1f}%). "
                  f"Proceeding normally.")
            print(f"  {msg}")

        # If previous phase had HIGH dropouts
        # → next phase enrolls MORE patients to compensate
        if dropout_rate > 0.20:
            warning = (f"⚠️  High dropout rate ({dropout_rate*100:.1f}%)! "
                      f"Next phase will enroll extra patients.")
            self.warnings.append(warning)
            print(f"  {warning}")

# ============================================================
# STEP 5: Improved simulate_phase function
# Now uses learnings from previous phase!
# ============================================================
def simulate_phase(phase_name, drug_name, learnings):

    print(f"\n{'='*60}")
    print(f"  SIMULATING {phase_name.upper()} — Drug: {drug_name}")
    print(f"{'='*60}")

    params = PHASE_PARAMS[phase_name]

    # --------------------------------------------------------
    # Apply learnings from previous phase
    # --------------------------------------------------------
    # Adjust adverse event rate based on previous phase
    adjusted_ae_rate = (params['base_adverse_event_rate'] *
                       learnings.adjustment_factor)

    # If previous phase had high dropouts → enroll more patients
    extra_enrollment = 0
    if (learnings.previous_dropout_rate and
            learnings.previous_dropout_rate > 0.20):
        extra_enrollment = int(params['min_patients'] * 0.20)
        print(f"  📈 Enrolling {extra_enrollment} extra patients "
              f"due to high dropout in previous phase")

    # Decide enrollment count
    num_patients = (random.randint(
        params['min_patients'],
        params['max_patients']
    ) + extra_enrollment)

    print(f"  Enrolling {num_patients} patients...")
    print(f"  Adjusted AE monitoring rate: "
          f"{adjusted_ae_rate*100:.1f}%")

    # --------------------------------------------------------
    # Create and simulate all patients
    # --------------------------------------------------------
    patients = []
    for i in range(num_patients):
        patient = Patient(
            patient_id=f"P{str(i+1).zfill(4)}",
            phase=phase_name
        )

        # STEP A: First simulate adverse event
        patient.simulate_adverse_event(adjusted_ae_rate)

        # STEP B: Then simulate dropout
        # (dropout is NOW influenced by adverse event!)
        patient.simulate_dropout(
            params['base_dropout_rate'],
            params['ae_dropout_multiplier']
        )

        # STEP C: If not dropped out → completed
        if not patient.dropped_out:
            patient.completed = True

        patients.append(patient)

    # --------------------------------------------------------
    # Calculate results
    # --------------------------------------------------------
    total_enrolled = len(patients)
    total_dropouts = sum(1 for p in patients if p.dropped_out)
    total_completed = sum(1 for p in patients if p.completed)
    total_ae = sum(1 for p in patients if p.had_adverse_event)
    total_severe_ae = sum(1 for p in patients if p.ae_severity == 2)
    total_mild_ae = sum(1 for p in patients if p.ae_severity == 1)

    actual_dropout_rate = round(total_dropouts / total_enrolled, 3)
    actual_ae_rate = round(total_ae / total_enrolled, 3)

    # --------------------------------------------------------
    # Print detailed results
    # --------------------------------------------------------
    print(f"\n  📊 PHASE RESULTS:")
    print(f"  Total Patients Enrolled  : {total_enrolled}")
    print(f"  Completed Trial          : {total_completed}")
    print(f"  Dropped Out              : {total_dropouts} "
          f"({actual_dropout_rate*100:.1f}%)")
    print(f"  Total Adverse Events     : {total_ae} "
          f"({actual_ae_rate*100:.1f}%)")
    print(f"    → Mild AE              : {total_mild_ae}")
    print(f"    → Severe AE            : {total_severe_ae}")

    # --------------------------------------------------------
    # THRESHOLD BASED GO/NO-GO DECISION
    # This is the KEY improvement!
    # We now check thresholds FIRST before random probability
    # --------------------------------------------------------
    print(f"\n  🎯 GO/NO-GO DECISION ANALYSIS:")
    print(f"  Adverse Event Rate : {actual_ae_rate*100:.1f}% "
          f"(max allowed: "
          f"{params['max_allowed_adverse_rate']*100:.1f}%)")
    print(f"  Dropout Rate       : {actual_dropout_rate*100:.1f}% "
          f"(max allowed: "
          f"{params['max_allowed_dropout_rate']*100:.1f}%)")

    # Check thresholds first
    if actual_ae_rate > params['max_allowed_adverse_rate']:
        decision = "NO-GO"
        reason = (f"Adverse event rate {actual_ae_rate*100:.1f}% "
                 f"exceeds maximum allowed "
                 f"{params['max_allowed_adverse_rate']*100:.1f}%")
        print(f"\n  ❌ NO-GO! Threshold breached!")
        print(f"  Reason: {reason}")

    elif actual_dropout_rate > params['max_allowed_dropout_rate']:
        decision = "NO-GO"
        reason = (f"Dropout rate {actual_dropout_rate*100:.1f}% "
                 f"exceeds maximum allowed "
                 f"{params['max_allowed_dropout_rate']*100:.1f}%")
        print(f"\n  ❌ NO-GO! Threshold breached!")
        print(f"  Reason: {reason}")

    else:
        # Thresholds passed → now use probability
        trial_succeeded = random.random() < params['success_rate']
        if trial_succeeded:
            decision = "GO"
            reason = "All thresholds passed. Trial outcome positive."
            print(f"\n  ✅ GO! {phase_name} PASSED!")
            print(f"  Drug '{drug_name}' proceeds to next phase.")
        else:
            decision = "NO-GO"
            reason = "Thresholds passed but trial outcome negative."
            print(f"\n  ❌ NO-GO! {phase_name} FAILED!")
            print(f"  Drug '{drug_name}' cannot proceed.")

    # --------------------------------------------------------
    # Update learnings for next phase
    # --------------------------------------------------------
    learnings.update(actual_ae_rate, actual_dropout_rate, phase_name)

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------
    return {
        'phase': phase_name,
        'drug_name': drug_name,
        'total_enrolled': total_enrolled,
        'total_completed': total_completed,
        'total_dropouts': total_dropouts,
        'dropout_rate_%': round(actual_dropout_rate * 100, 1),
        'total_ae': total_ae,
        'mild_ae': total_mild_ae,
        'severe_ae': total_severe_ae,
        'ae_rate_%': round(actual_ae_rate * 100, 1),
        'decision': decision,
        'reason': reason
    }

# ============================================================
# STEP 6: Full Trial Simulation (Improved)
# ============================================================
def run_full_trial(drug_name):

    print(f"\n{'#'*60}")
    print(f"  FULL CLINICAL TRIAL SIMULATION")
    print(f"  Drug: {drug_name}")
    print(f"{'#'*60}")

    all_results = []

    # Create a fresh learnings object for this trial
    # It starts empty and gets filled after each phase
    learnings = PhaseLearnings()

    for phase in ['Phase 1', 'Phase 2', 'Phase 3']:
        result = simulate_phase(phase, drug_name, learnings)
        all_results.append(result)

        if result['decision'] == 'NO-GO':
            print(f"\n🛑 TRIAL STOPPED at {phase}")
            print(f"   Reason: {result['reason']}")
            break

    # Final verdict
    if len(all_results) == 3 and all_results[-1]['decision'] == 'GO':
        print(f"\n🏆 TRIAL COMPLETE!")
        print(f"   Drug '{drug_name}' passed all 3 phases!")
        print(f"   Recommended for regulatory approval!")
    elif all_results[-1]['decision'] == 'NO-GO':
        stopped_at = all_results[-1]['phase']
        print(f"\n💊 Drug '{drug_name}' discontinued at {stopped_at}")

    # Print summary table
    results_df = pd.DataFrame(all_results)
    print(f"\n📋 TRIAL SUMMARY:")
    print(results_df[[
        'phase', 'total_enrolled', 'dropout_rate_%',
        'ae_rate_%', 'decision', 'reason'
    ]].to_string(index=False))

    return results_df

# ============================================================
# STEP 7: Run simulations
# ============================================================
if __name__ == "__main__":

    print("🔬 CLINICAL TRIAL SIMULATION SYSTEM v2.0")
    print("   Capgemini Internship 2026 — Madhyam")
    print("   Improved: Threshold decisions + AE→Dropout + "
          "Phase learnings")

    # Simulate 3 different drugs
    results1 = run_full_trial("DrugX-2026")
    print("\n" + "="*60)

    results2 = run_full_trial("CureMax-001")
    print("\n" + "="*60)

    results3 = run_full_trial("FailSafe-Test")
    print("\n" + "="*60)

    # Save all results
    all_results = pd.concat([results1, results2, results3])
    all_results.to_csv('../data/simulation_results_v2.csv', index=False)
    print("\n✅ All results saved to simulation_results_v2.csv!")