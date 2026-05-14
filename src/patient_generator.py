# ============================================================
# CLINICAL TRIAL SIMULATION - CAPGEMINI INTERNSHIP 2026
# File: patient_generator.py
# Purpose: Generate realistic synthetic patients
#          grounded in disease-specific demographics
#          Uses Faker + real clinical knowledge
# Author: Madhyam
# ============================================================

import random
import numpy as np
from faker import Faker
from dataclasses import dataclass
from typing import List

fake = Faker()

random.seed(42)
np.random.seed(42)

print("✅ Patient generator libraries imported!")

# ============================================================
# STEP 1: Disease-specific patient profiles
# Based on real clinical trial demographics
# ============================================================

DISEASE_PROFILES = {
    'cancer': {
        'age_range': (35, 75),
        'age_mean': 58,
        'age_std': 12,
        'female_ratio': 0.55,
        'common_comorbidities': [
            'Hypertension',
            'Diabetes Type 2',
            'Anaemia',
            'Fatigue',
            'Nausea'
        ],
        'comorbidity_prob': 0.65,
        'baseline_severity_range': (4, 9),
        'typical_biomarkers': {
            'CA-125': (0, 500),
            'CEA': (0, 100),
            'PSA': (0, 50)
        }
    },

    'infectious': {
        'age_range': (18, 65),
        'age_mean': 38,
        'age_std': 15,
        'female_ratio': 0.48,
        'common_comorbidities': [
            'Immunodeficiency',
            'Malnutrition',
            'Fever',
            'Fatigue'
        ],
        'comorbidity_prob': 0.45,
        'baseline_severity_range': (3, 8),
        'typical_biomarkers': {
            'CD4_count': (50, 500),
            'Viral_load': (0, 100000),
            'CRP': (0, 50)
        }
    },

    'cardiovascular': {
        'age_range': (45, 80),
        'age_mean': 65,
        'age_std': 10,
        'female_ratio': 0.42,
        'common_comorbidities': [
            'Hypertension',
            'Diabetes',
            'Hyperlipidemia',
            'Obesity',
            'Previous MI'
        ],
        'comorbidity_prob': 0.75,
        'baseline_severity_range': (4, 8),
        'typical_biomarkers': {
            'Systolic_BP': (120, 180),
            'LDL': (100, 250),
            'Troponin': (0, 2)
        }
    },

    'diabetes': {
        'age_range': (30, 75),
        'age_mean': 55,
        'age_std': 12,
        'female_ratio': 0.50,
        'common_comorbidities': [
            'Obesity',
            'Hypertension',
            'Neuropathy',
            'Retinopathy'
        ],
        'comorbidity_prob': 0.70,
        'baseline_severity_range': (3, 7),
        'typical_biomarkers': {
            'HbA1c': (6.5, 12.0),
            'Fasting_glucose': (126, 300),
            'BMI': (25, 40)
        }
    },

    'mental_health': {
        'age_range': (18, 65),
        'age_mean': 38,
        'age_std': 14,
        'female_ratio': 0.58,
        'common_comorbidities': [
            'Anxiety',
            'Sleep disorder',
            'Substance use',
            'Social isolation'
        ],
        'comorbidity_prob': 0.60,
        'baseline_severity_range': (4, 9),
        'typical_biomarkers': {
            'GAF_score': (20, 70),
            'PHQ9_score': (10, 27),
            'PANSS_score': (60, 120)
        }
    },

    'healthy': {
        'age_range': (18, 55),
        'age_mean': 30,
        'age_std': 10,
        'female_ratio': 0.50,
        'common_comorbidities': [],
        'comorbidity_prob': 0.05,
        'baseline_severity_range': (0, 2),
        'typical_biomarkers': {
            'BMI': (18, 25),
            'Blood_pressure': (110, 130),
            'Heart_rate': (60, 80)
        }
    },

    'general': {
        'age_range': (18, 75),
        'age_mean': 50,
        'age_std': 15,
        'female_ratio': 0.50,
        'common_comorbidities': [
            'Hypertension',
            'Diabetes',
            'Obesity'
        ],
        'comorbidity_prob': 0.40,
        'baseline_severity_range': (3, 8),
        'typical_biomarkers': {
            'BMI': (18, 35),
            'Blood_pressure': (110, 150),
        }
    }
}

# ============================================================
# STEP 2: Patient dataclass
# A proper structured patient object!
# ============================================================

@dataclass
class SyntheticPatient:

    # Identity
    patient_id: str
    name: str
    age: int
    gender: str
    weight_kg: float
    height_cm: float

    # Disease context
    disease_category: str
    baseline_severity: int
    comorbidities: List[str]
    biomarkers: dict

    # Trial tracking
    phase: str
    enrolled: bool = True
    dropped_out: bool = False
    had_adverse_event: bool = False
    ae_severity: int = 0
    ae_description: str = ""
    completed: bool = False

    # Efficacy tracking
    drug_response_score: float = 0.0
    endpoint_achieved: bool = False
    baseline_score: float = 0.0
    followup_score: float = 0.0

    def bmi(self):
        return round(
            self.weight_kg /
            ((self.height_cm / 100) ** 2),
            1
        )

    def summary(self):
        return (
            f"Patient {self.patient_id} | "
            f"{self.age}yr {self.gender} | "
            f"BMI:{self.bmi()} | "
            f"Severity:{self.baseline_severity}/10 | "
            f"Comorbidities:{len(self.comorbidities)}"
        )

# ============================================================
# STEP 3: Patient Generator class
# ============================================================

class PatientGenerator:

    def __init__(self,
                 disease_category='general',
                 phase='Phase 1'):

        self.disease = disease_category
        self.phase = phase

        self.profile = DISEASE_PROFILES.get(
            disease_category,
            DISEASE_PROFILES['general']
        )

        self.patient_count = 0

    def generate_age(self):
        """Generate age from disease-specific distribution."""

        age = int(
            np.random.normal(
                self.profile['age_mean'],
                self.profile['age_std']
            )
        )

        min_age, max_age = self.profile['age_range']

        return max(min_age, min(max_age, age))

    def generate_weight_height(self, age, gender):
        """Generate realistic weight and height."""

        if gender == 'Female':
            height = np.random.normal(162, 7)
            weight = np.random.normal(65, 12)
        else:
            height = np.random.normal(175, 8)
            weight = np.random.normal(80, 15)

        # Age adjustment
        if age > 60:
            height -= np.random.uniform(0, 3)

        return round(weight, 1), round(height, 1)

    def generate_comorbidities(self):
        """Generate disease-appropriate comorbidities."""

        comorbidities = []

        if random.random() < self.profile['comorbidity_prob']:

            available = self.profile['common_comorbidities']

            if available:
                n = random.randint(
                    1,
                    min(3, len(available))
                )

                comorbidities = random.sample(
                    available,
                    n
                )

        return comorbidities

    def generate_biomarkers(self):
        """Generate disease-specific biomarker values."""

        biomarkers = {}

        for marker, (low, high) in self.profile[
            'typical_biomarkers'
        ].items():

            biomarkers[marker] = round(
                random.uniform(low, high),
                2
            )

        return biomarkers

    def generate_baseline_severity(self):
        """Generate disease severity score."""

        low, high = self.profile[
            'baseline_severity_range'
        ]

        return random.randint(low, high)

    def generate_drug_response(
        self,
        baseline_severity,
        comorbidities
    ):
        """
        Generate drug response score.
        Higher baseline severity + more comorbidities
        = lower drug response
        """

        # Base response probability
        base_response = 0.65

        # Severity penalty
        severity_penalty = (
            (baseline_severity - 5) * 0.04
        )

        # Comorbidity penalty
        comorbidity_penalty = (
            len(comorbidities) * 0.05
        )

        # Phase modifier
        phase_modifier = {
            'Phase 1': 0.10,
            'Phase 2': 0.00,
            'Phase 3': -0.05
        }.get(self.phase, 0.0)

        response_prob = (
            base_response
            - severity_penalty
            - comorbidity_penalty
            + phase_modifier
            + np.random.normal(0, 0.1)
        )

        return max(
            0.0,
            min(1.0, round(response_prob, 3))
        )

    def generate_patient(self):
        """Generate one complete synthetic patient."""

        self.patient_count += 1

        # Generate demographics
        age = self.generate_age()

        gender = (
            'Female'
            if random.random() < self.profile['female_ratio']
            else 'Male'
        )

        weight, height = self.generate_weight_height(
            age,
            gender
        )

        # Generate clinical profile
        comorbidities = self.generate_comorbidities()

        biomarkers = self.generate_biomarkers()

        baseline_severity = (
            self.generate_baseline_severity()
        )

        # Generate drug response
        drug_response = self.generate_drug_response(
            baseline_severity,
            comorbidities
        )

        # Baseline and followup scores
        baseline_score = round(
            baseline_severity * 10
            + np.random.normal(0, 5),
            1
        )

        followup_score = round(
            baseline_score *
            (1 - drug_response * 0.5),
            1
        )

        patient = SyntheticPatient(
            patient_id=f"P{str(self.patient_count).zfill(4)}",
            name=fake.name(),
            age=age,
            gender=gender,
            weight_kg=weight,
            height_cm=height,
            disease_category=self.disease,
            baseline_severity=baseline_severity,
            comorbidities=comorbidities,
            biomarkers=biomarkers,
            phase=self.phase,
            drug_response_score=drug_response,
            baseline_score=baseline_score,
            followup_score=followup_score,
            endpoint_achieved=drug_response > 0.5
        )

        return patient

    def generate_cohort(self, n_patients):
        """Generate a full cohort of patients."""

        return [
            self.generate_patient()
            for _ in range(n_patients)
        ]

# ============================================================
# STEP 4: Test the generator
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("SYNTHETIC PATIENT GENERATOR TEST")
    print("=" * 60)

    # Test each disease category
    for disease in [
        'cancer',
        'infectious',
        'healthy',
        'general'
    ]:

        print(f"\n{'─' * 50}")
        print(f"Disease: {disease.upper()} | Phase 2")
        print(f"{'─' * 50}")

        generator = PatientGenerator(
            disease_category=disease,
            phase='Phase 2'
        )

        # Generate 5 sample patients
        cohort = generator.generate_cohort(5)

        for patient in cohort:

            print(patient.summary())

            print(
                f"  Biomarkers: "
                f"{patient.biomarkers}"
            )

            print(
                f"  Comorbidities: "
                f"{patient.comorbidities or 'None'}"
            )

            print(
                f"  Drug response: "
                f"{patient.drug_response_score:.2f} | "
                f"Endpoint: "
                f"{patient.endpoint_achieved}"
            )

            print(
                f"  Baseline→Followup: "
                f"{patient.baseline_score}"
                f"→{patient.followup_score}"
            )

    # Full cohort statistics
    print(f"\n{'=' * 60}")
    print("COHORT STATISTICS — Cancer Phase 3")
    print(f"{'=' * 60}")

    cancer_gen = PatientGenerator(
        disease_category='cancer',
        phase='Phase 3'
    )

    cancer_cohort = cancer_gen.generate_cohort(100)

    ages = [p.age for p in cancer_cohort]

    responses = [
        p.drug_response_score
        for p in cancer_cohort
    ]

    endpoints = [
        p.endpoint_achieved
        for p in cancer_cohort
    ]

    comorbidity_counts = [
        len(p.comorbidities)
        for p in cancer_cohort
    ]

    print(f"Total patients: {len(cancer_cohort)}")

    print(
        f"Age: mean={np.mean(ages):.1f}, "
        f"std={np.std(ages):.1f}"
    )

    print(
        f"Gender F/M: "
        f"{sum(1 for p in cancer_cohort if p.gender == 'Female')}/"
        f"{sum(1 for p in cancer_cohort if p.gender == 'Male')}"
    )

    print(
        f"Drug response: "
        f"mean={np.mean(responses):.3f}, "
        f"std={np.std(responses):.3f}"
    )

    print(
        f"Endpoint achieved: "
        f"{sum(endpoints)}/100 "
        f"({sum(endpoints) / len(cancer_cohort) * 100:.1f}%)"
    )

    print(
        f"Avg comorbidities: "
        f"{np.mean(comorbidity_counts):.1f}"
    )

    print(f"\n✅ Patient generator working perfectly!")
    print(f"{'=' * 60}")