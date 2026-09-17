# ============================================================
# CLINICAL TRIAL SIMULATION - CAPGEMINI INTERNSHIP 2026
# File: ai_agent.py
# Purpose: Dr. ARIA v2 — Operational Intelligence
#          - Receives ML predictions + simulation results
#          - Reasons independently
#          - Challenges simulation decisions
#          - Returns structured JSON
#          - Feedback loop capable
# Author: Madhyam
# ============================================================

import os
import json
import pickle
import pandas as pd
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

# Load from multiple possible locations
from pathlib import Path
import os

# Try different paths
possible_paths = [
    Path('../.env'),
    Path('../../.env'),
    Path('.env'),
    Path(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '.env'
    ))
]

for env_path in possible_paths:
    if env_path.exists():
        base_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(base_dir, '..', '.env')
        load_dotenv(dotenv_path=env_path)
        break

print("✅ Libraries imported!")

# ============================================================
# STEP 1: Initialize Groq client
# ============================================================
api_key = os.getenv('GROQ_API_KEY')
if not api_key:
    # Hardcode as fallback for dashboard
    api_key = os.getenv('GROQ_API_KEY')


else:
    client = Groq(api_key=api_key)
    print("✅ Groq client initialized!")

# ============================================================
# STEP 2: Load ML models
# ============================================================
def load_ml_models():
    """Load saved ML models for predictions."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(base_dir, '..', 'data', 'outcome_model.pkl'), 'rb') as f:
            outcome_model = pickle.load(f)

        with open(os.path.join(base_dir, '..', 'data', 'dropout_model.pkl'), 'rb') as f:
            dropout_model = pickle.load(f)

        with open(os.path.join(base_dir, '..', 'data', 'model_info.pkl'), 'rb') as f:
            model_info = pickle.load(f)

        print("✅ ML models loaded!")
        return outcome_model, dropout_model, model_info

    except FileNotFoundError:
        print("⚠️ ML models not found. Running without ML predictions.")
        return None, None, None

outcome_model, dropout_model, model_info = load_ml_models()

# ============================================================
# STEP 3: Feature encoding maps
# (Must match ml_model.py exactly!)
# ============================================================
PHASE_MAP = {'Phase 1': 1, 'Phase 2': 2, 'Phase 3': 3}
ORG_SUCCESS_MAP = {
    'FED': 0.864, 'INDUSTRY': 0.848, 'NIH': 0.855,
    'NETWORK': 0.806, 'OTHER': 0.767,
    'OTHER_GOV': 0.853, 'UNKNOWN': 0.820,
}
ERA_MAP = {2010: 2, 2015: 3, 2020: 4}

def encode_features_for_ml(phase_result,
                            disease_category,
                            org_class='INDUSTRY',
                            start_year=2020):
    """
    Encodes phase result into ML features.
    Returns DataFrame ready for model prediction.
    """
    phase_num = PHASE_MAP.get(
        phase_result['phase'], 1)
    org_success = ORG_SUCCESS_MAP.get(
        org_class, 0.820)

    # Trial era
    if start_year >= 2015:
        trial_era = 4
    elif start_year >= 2010:
        trial_era = 3
    elif start_year >= 2005:
        trial_era = 2
    else:
        trial_era = 1

    # Disease flags
    is_cancer = int(disease_category == 'cancer')
    is_healthy = int(disease_category == 'healthy')
    is_mental = int(
        disease_category == 'mental_health')
    is_diabetes = int(
        disease_category == 'diabetes')
    is_infectious = int(
        disease_category == 'infectious')
    is_cardio = int(
        disease_category == 'cardiovascular')

    features = pd.DataFrame([{
        'phase_number': phase_num,
        'is_big_pharma': int(
            org_class == 'INDUSTRY'),
        'is_top_cancer_center': 0,
        'org_success_rate': org_success,
        'is_cancer': is_cancer,
        'is_healthy': is_healthy,
        'is_mental_health': is_mental,
        'is_diabetes': is_diabetes,
        'is_infectious': is_infectious,
        'is_cardiovascular': is_cardio,
        'is_pain': 0,
        'is_drug': 1,
        'is_biological': 0,
        'num_conditions': 1,
        'is_treatment': 1,
        'is_prevention': 0,
        'is_basic_science': 0,
        'trial_era': trial_era,
        'age_encoded': 1
    }])
    return features

# ============================================================
# STEP 4: Get ML predictions
# ============================================================
def get_ml_predictions(phase_result,
                       disease_category,
                       org_class='INDUSTRY'):
    """
    Gets ML model predictions for current phase.
    Returns dict with probabilities and risk scores.
    """
    if outcome_model is None:
        return {
            'predicted_success_prob': 0.65,
            'predicted_dropout_risk': 0.20,
            'ml_recommendation': 'GO',
            'ml_confidence': 0.50,
            'ml_available': False
        }

    features = encode_features_for_ml(
        phase_result, disease_category, org_class)

    # Success probability
    success_prob = outcome_model.predict_proba(
        features)[0][1]

    # Dropout risk
    dropout_risk = dropout_model.predict_proba(
        features)[0][1]

    # Confidence = how far from 0.5 (uncertain)
    confidence = abs(success_prob - 0.5) * 2

    return {
        'predicted_success_prob': round(
            success_prob, 3),
        'predicted_dropout_risk': round(
            dropout_risk, 3),
        'ml_recommendation': (
            'GO' if success_prob > 0.55
            else 'NO-GO'),
        'ml_confidence': round(confidence, 3),
        'ml_available': True
    }

# ============================================================
# STEP 5: Dr. ARIA System Prompt v2
# Now an OPERATIONAL INTELLIGENCE system!
# ============================================================
SYSTEM_PROMPT_V2 = """
You are Dr. ARIA (Advanced Research Intelligence Agent),
an expert AI clinical trial manager and independent
safety monitor with 20 years of pharmaceutical experience.

YOUR ROLE HAS CHANGED. You are NOT just a narrator.
You are an INDEPENDENT DECISION MAKER who:

1. RECEIVES simulation results AND ML predictions
2. REASONS over BOTH independently
3. CHALLENGES the simulation decision if warranted
4. RECOMMENDS protocol modifications when needed
5. TRIGGERS feedback loops for dose adjustments

CRITICAL RULES:
- If ML prediction conflicts with simulation decision
  → FLAG this prominently as "CONFLICTED"
- If ML success probability < 0.55 → recommend caution
  even if simulation says GO
- If response rate < 40% in Phase 2/3 → recommend
  dose escalation or protocol modification
- If severe AE > 10% → recommend dose reduction
- Always provide YOUR OWN independent recommendation
  separate from the simulation decision

RESPONSE FORMAT — You MUST respond in this exact JSON:
{
    "phase_summary": "2-3 sentence summary",
    "simulation_decision": "GO or NO-GO",
    "ml_prediction": "GO or NO-GO",
    "ml_simulation_alignment": "ALIGNED or CONFLICTED",
    "aria_recommendation": "GO or NO-GO or CONDITIONAL GO",
    "confidence_level": "HIGH or MEDIUM or LOW",
    "risk_level": "LOW or MEDIUM or HIGH or CRITICAL",
    "key_findings": ["finding1", "finding2", "finding3"],
    "safety_concerns": ["concern1"] or [],
    "efficacy_assessment": "assessment text",
    "protocol_modifications": ["mod1"] or [],
    "monitoring_requirements": ["req1", "req2"],
    "dose_adjustment_needed": true or false,
    "proceed_to_next_phase": true or false,
    "case_study_paragraph": "professional paragraph"
}

Respond ONLY with the JSON. No prose before or after.
"""

# ============================================================
# STEP 6: Core analysis function
# ============================================================
def analyze_phase_v2(phase_result,
                     ml_predictions,
                     previous_phases=None,
                     drug_name="Unknown",
                     disease_category="general"):
    """
    Dr. ARIA v2 — analyzes phase with ML integration.
    Returns structured JSON response.
    """

    # Build previous context
    prev_context = ""
    if previous_phases and len(previous_phases) > 0:
        prev_context = "\n\nPREVIOUS PHASE HISTORY:"
        for prev in previous_phases:
            prev_context += f"""
{prev['phase']}:
- Enrolled: {prev['total_enrolled']}
- AE rate: {prev['ae_rate_%']}%
  (Mild: {prev['mild_ae']}, Severe: {prev['severe_ae']})
- Dropout: {prev['dropout_rate_%']}%
- Response rate: {prev['response_rate_%']}%
- Endpoint rate: {prev['endpoint_rate_%']}%
- Improvement: {prev['improvement_%']}%
- P-value: {prev['p_value']}
- Decision: {prev['decision']}
"""

    # Build user prompt with ALL data
    user_prompt = f"""
DRUG: {drug_name}
DISEASE CATEGORY: {disease_category}

CURRENT PHASE: {phase_result['phase']}

SIMULATION RESULTS:
- Patients enrolled: {phase_result['total_enrolled']}
- Completed: {phase_result['total_completed']}
- Dropout rate: {phase_result['dropout_rate_%']}%
- Total AE: {phase_result['total_ae']}
  Mild: {phase_result['mild_ae']}
  Severe: {phase_result['severe_ae']}
- AE rate: {phase_result['ae_rate_%']}%
- Simulation decision: {phase_result['decision']}
- Decision reason: {phase_result['reason']}

EFFICACY DATA:
- Response rate: {phase_result['response_rate_%']}%
- Endpoint achievement: {phase_result['endpoint_rate_%']}%
- Baseline score: {phase_result['avg_baseline_score']}
- Followup score: {phase_result['avg_followup_score']}
- Improvement: {phase_result['improvement_%']}%
- P-value: {phase_result['p_value']}
- Statistically significant: {phase_result['statistically_significant']}

DEMOGRAPHICS:
- Mean age: {phase_result['age_mean']} years
- Female/Male: {phase_result['female_count']}/{phase_result['male_count']}

ML MODEL PREDICTIONS:
- Predicted success probability: {ml_predictions['predicted_success_prob']}
- Predicted dropout risk: {ml_predictions['predicted_dropout_risk']}
- ML recommendation: {ml_predictions['ml_recommendation']}
- ML confidence: {ml_predictions['ml_confidence']}

WARNINGS FROM SIMULATION:
{chr(10).join(phase_result.get('warnings', ['None']))}
{prev_context}

Provide your independent analysis in the required JSON format.
"""

    print(f"\n🤖 Dr. ARIA v2 analyzing "
          f"{phase_result['phase']}...")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system",
             "content": SYSTEM_PROMPT_V2},
            {"role": "user",
             "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=800,
    )

    raw_response = response.choices[0].message.content

    # Parse JSON response
    try:
        # Clean response if needed
        clean = raw_response.strip()
        if clean.startswith("```"):
            clean = clean.split(
                "```")[1].replace("json", "").strip()
        aria_analysis = json.loads(clean)
        return aria_analysis
    except json.JSONDecodeError:
        # If JSON parsing fails return structured error
        return {
            "phase_summary": raw_response[:200],
            "aria_recommendation": phase_result[
                'decision'],
            "ml_simulation_alignment": "UNKNOWN",
            "risk_level": "MEDIUM",
            "key_findings": ["JSON parsing failed"],
            "safety_concerns": [],
            "protocol_modifications": [],
            "monitoring_requirements": [],
            "dose_adjustment_needed": False,
            "proceed_to_next_phase": phase_result[
                'decision'] == 'GO',
            "case_study_paragraph":
                "Analysis unavailable."
        }

# ============================================================
# STEP 7: Final verdict function
# ============================================================
def get_final_verdict_v2(all_results,
                         drug_name,
                         disease_category):
    """Gets final verdict after all phases."""

    phases_summary = ""
    for r in all_results:
        if 'phase' in r and r['phase'] != 'FINAL':
            phases_summary += f"""
{r['phase']}:
- AE: {r.get('ae_rate_%', 'N/A')}%
- Response: {r.get('response_rate_%', 'N/A')}%
- Improvement: {r.get('improvement_%', 'N/A')}%
- P-value: {r.get('p_value', 'N/A')}
- Decision: {r.get('decision', 'N/A')}
- ARIA: {r.get('aria_analysis', {}).get('aria_recommendation', 'N/A')}
"""

    final_prompt = f"""
COMPLETE TRIAL SUMMARY FOR: {drug_name}
DISEASE: {disease_category}

{phases_summary}

Provide final verdict in this JSON format:
{{
    "overall_assessment": "2-3 sentence summary",
    "key_safety_findings": ["finding1", "finding2"],
    "efficacy_conclusion": "efficacy assessment",
    "final_recommendation": "APPROVE FOR REGULATORY SUBMISSION or DISCONTINUE or REQUIRES FURTHER STUDY",
    "regulatory_readiness": "READY or NOT READY or CONDITIONAL",
    "case_study_paragraph": "professional 3-4 sentence paragraph for case study document",
    "lessons_learned": ["lesson1", "lesson2"]
}}

Respond ONLY with JSON.
"""

    print("\n🤖 Dr. ARIA preparing final verdict...")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system",
             "content": SYSTEM_PROMPT_V2},
            {"role": "user",
             "content": final_prompt}
        ],
        temperature=0.2,
        max_tokens=600,
    )

    raw = response.choices[0].message.content
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split(
                "```")[1].replace("json", "").strip()
        return json.loads(clean)
    except:
        return {"overall_assessment": raw,
                "final_recommendation": "REQUIRES FURTHER STUDY"}

# ============================================================
# STEP 8: Complete trial with AI
# ============================================================
def run_trial_with_aria(drug_name,
                        simulation_results,
                        disease_category='general',
                        org_class='INDUSTRY'):
    """
    Runs Dr. ARIA analysis on simulation results.
    Integrates ML predictions into agent reasoning.
    """

    print(f"\n{'#'*60}")
    print(f"  DR. ARIA v2 ANALYSIS")
    print(f"  Drug: {drug_name} | "
          f"Disease: {disease_category}")
    print(f"{'#'*60}")

    enriched_results = []
    previous_phases = []

    for phase_result in simulation_results:

        # Get ML predictions for this phase
        ml_preds = get_ml_predictions(
            phase_result,
            disease_category,
            org_class
        )

        # Get ARIA analysis
        aria_analysis = analyze_phase_v2(
            phase_result=phase_result,
            ml_predictions=ml_preds,
            previous_phases=previous_phases,
            drug_name=drug_name,
            disease_category=disease_category
        )

        # Add to result
        enriched = phase_result.copy()
        enriched['ml_predictions'] = ml_preds
        enriched['aria_analysis'] = aria_analysis
        enriched_results.append(enriched)
        previous_phases.append(phase_result)

        # Display ARIA analysis
        print(f"\n{'='*60}")
        print(f"DR. ARIA — {phase_result['phase']}")
        print(f"{'='*60}")
        print(f"Simulation : "
              f"{phase_result['decision']}")
        print(f"ML Model   : "
              f"{ml_preds['ml_recommendation']} "
              f"(prob: {ml_preds['predicted_success_prob']})")
        print(f"ARIA       : "
              f"{aria_analysis.get('aria_recommendation', 'N/A')}")
        print(f"Alignment  : "
              f"{aria_analysis.get('ml_simulation_alignment', 'N/A')}")
        print(f"Risk Level : "
              f"{aria_analysis.get('risk_level', 'N/A')}")
        print(f"Confidence : "
              f"{aria_analysis.get('confidence_level', 'N/A')}")

        print(f"\nKey Findings:")
        for finding in aria_analysis.get(
                'key_findings', []):
            print(f"  • {finding}")

        if aria_analysis.get('safety_concerns'):
            print(f"\nSafety Concerns:")
            for concern in aria_analysis[
                    'safety_concerns']:
                print(f"  ⚠️  {concern}")

        if aria_analysis.get(
                'protocol_modifications'):
            print(f"\nProtocol Modifications:")
            for mod in aria_analysis[
                    'protocol_modifications']:
                print(f"  🔧 {mod}")

        print(f"\nEfficacy: "
              f"{aria_analysis.get('efficacy_assessment', 'N/A')}")
        print(f"\nCase Study:")
        print(f"  {aria_analysis.get('case_study_paragraph', 'N/A')}")

        # Stop if NO-GO
        if phase_result['decision'] == 'NO-GO':
            print(f"\n🛑 Trial stopped.")
            break

    # Final verdict
    final = get_final_verdict_v2(
        enriched_results,
        drug_name,
        disease_category
    )

    print(f"\n{'='*60}")
    print("DR. ARIA — FINAL VERDICT")
    print(f"{'='*60}")
    print(f"Assessment : "
          f"{final.get('overall_assessment', 'N/A')}")
    print(f"Recommendation: "
          f"{final.get('final_recommendation', 'N/A')}")
    print(f"Regulatory : "
          f"{final.get('regulatory_readiness', 'N/A')}")
    print(f"\nCase Study Paragraph:")
    print(f"  {final.get('case_study_paragraph', 'N/A')}")

    if final.get('lessons_learned'):
        print(f"\nLessons Learned:")
        for lesson in final['lessons_learned']:
            print(f"  📌 {lesson}")

    enriched_results.append({
        'phase': 'FINAL VERDICT',
        'drug_name': drug_name,
        'final_verdict': final
    })

    return enriched_results

# ============================================================
# STEP 9: Test with simulation results
# ============================================================
if __name__ == "__main__":

    print("\n🔬 DR. ARIA v2 — OPERATIONAL INTELLIGENCE")
    print("   Capgemini Internship 2026 — Madhyam")
    print("   ML-integrated + JSON + Independent reasoning")

    # Test Case 1 — Successful cancer trial
    print("\n" + "="*60)
    print("TEST: OncoCure-X1 (Cancer — 3 phase success)")
    print("="*60)

    # Using realistic simulation results
    oncox_results = [
        {
            'phase': 'Phase 1',
            'drug_name': 'OncoCure-X1',
            'total_enrolled': 60,
            'total_completed': 53,
            'total_dropouts': 7,
            'dropout_rate_%': 11.7,
            'total_ae': 18,
            'mild_ae': 10,
            'severe_ae': 8,
            'ae_rate_%': 30.0,
            'response_rate_%': 61.8,
            'endpoint_rate_%': 86.8,
            'avg_baseline_score': 63.4,
            'avg_followup_score': 44.2,
            'improvement_%': 30.3,
            'p_value': 0.0,
            'statistically_significant': True,
            'age_mean': 56.1,
            'female_count': 35,
            'male_count': 25,
            'decision': 'GO',
            'reason': 'All thresholds passed.',
            'warnings': ['High AE 30% detected']
        },
        {
            'phase': 'Phase 2',
            'drug_name': 'OncoCure-X1',
            'total_enrolled': 250,
            'total_completed': 198,
            'total_dropouts': 52,
            'dropout_rate_%': 20.8,
            'total_ae': 55,
            'mild_ae': 38,
            'severe_ae': 17,
            'ae_rate_%': 22.0,
            'response_rate_%': 54.2,
            'endpoint_rate_%': 61.5,
            'avg_baseline_score': 67.1,
            'avg_followup_score': 49.8,
            'improvement_%': 25.8,
            'p_value': 0.001,
            'statistically_significant': True,
            'age_mean': 57.2,
            'female_count': 148,
            'male_count': 102,
            'decision': 'GO',
            'reason': 'All thresholds passed.',
            'warnings': ['High dropout 20.8%']
        },
        {
            'phase': 'Phase 3',
            'drug_name': 'OncoCure-X1',
            'total_enrolled': 1850,
            'total_completed': 1650,
            'total_dropouts': 200,
            'dropout_rate_%': 10.8,
            'total_ae': 185,
            'mild_ae': 148,
            'severe_ae': 37,
            'ae_rate_%': 10.0,
            'response_rate_%': 58.3,
            'endpoint_rate_%': 64.2,
            'avg_baseline_score': 65.8,
            'avg_followup_score': 47.2,
            'improvement_%': 28.3,
            'p_value': 0.0001,
            'statistically_significant': True,
            'age_mean': 57.8,
            'female_count': 1050,
            'male_count': 800,
            'decision': 'GO',
            'reason': 'All thresholds passed.',
            'warnings': []
        }
    ]

    results = run_trial_with_aria(
        drug_name="OncoCure-X1",
        simulation_results=oncox_results,
        disease_category="cancer",
        org_class="INDUSTRY"
    )

    # Save analyses
    print("\n\nSaving analyses...")
    analyses = []
    for r in results:
        if 'aria_analysis' in r:
            analyses.append({
                'drug': r.get('drug_name', 'Unknown'),
                'phase': r['phase'],
                'simulation_decision': r.get(
                    'decision', 'N/A'),
                'ml_recommendation': r.get(
                    'ml_predictions', {}).get(
                    'ml_recommendation', 'N/A'),
                'aria_recommendation': r.get(
                    'aria_analysis', {}).get(
                    'aria_recommendation', 'N/A'),
                'alignment': r.get(
                    'aria_analysis', {}).get(
                    'ml_simulation_alignment', 'N/A'),
                'risk_level': r.get(
                    'aria_analysis', {}).get(
                    'risk_level', 'N/A'),
                'case_study': r.get(
                    'aria_analysis', {}).get(
                    'case_study_paragraph', 'N/A')
            })

    pd.DataFrame(analyses).to_csv(
        '../data/aria_analyses_v2.csv',
        index=False
    )
    print("✅ Analyses saved to aria_analyses_v2.csv!")

    print("\n" + "="*60)
    print("DR. ARIA v2 COMPLETE!")
    print("="*60)