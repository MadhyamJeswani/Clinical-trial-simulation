# ============================================================
# CLINICAL TRIAL SIMULATION - CAPGEMINI INTERNSHIP 2026
# File: ai_agent.py
# Purpose: AI Trial Manager Agent using Groq API
#          - Analyses phase results
#          - Produces GO/NO-GO recommendations
#          - Provides natural language justification
#          - Acts as intelligent trial manager
# Author: Madhyam
# ============================================================

# ============================================================
# STEP 1: Import libraries
# ============================================================
import os
import json
from groq import Groq
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env file
# This reads our GROQ_API_KEY safely
from pathlib import Path
load_dotenv(dotenv_path=Path('../.env'))

print("✅ Libraries imported!")

# ============================================================
# STEP 2: Initialize Groq client
# ============================================================
api_key = os.getenv('GROQ_API_KEY')

if not api_key:
    print("❌ ERROR: GROQ_API_KEY not found!")
    print("Make sure your .env file has: GROQ_API_KEY=your_key")
    exit()

client = Groq(api_key=api_key)
print("✅ Groq client initialized!")

# ============================================================
# STEP 3: Define the AI Agent system prompt
# This tells the AI WHO it is and HOW to behave
# ============================================================
SYSTEM_PROMPT = """
You are Dr. ARIA (Advanced Research Intelligence Agent), 
an expert AI clinical trial manager with 20 years of 
experience in pharmaceutical drug development.

Your role is to:
1. Analyze clinical trial phase results
2. Provide GO/NO-GO recommendations with clear reasoning
3. Identify risks and safety concerns
4. Suggest monitoring protocols for next phase
5. Write professional medical assessments

Your analysis style:
- Always start with a brief phase summary
- Clearly state your GO/NO-GO recommendation
- Explain the key factors behind your decision
- Highlight any safety concerns
- Suggest what to watch for in the next phase
- Use professional but clear language
- Be concise but thorough (150-200 words per analysis)

Safety thresholds you know:
- Phase 1: AE rate must be below 35%, dropout below 25%
- Phase 2: AE rate must be below 25%, dropout below 30%  
- Phase 3: AE rate must be below 15%, dropout below 20%

You make decisions based on both:
1. Hard thresholds (automatic NO-GO if exceeded)
2. Trends and patterns across phases
3. Clinical significance of findings
"""


# ============================================================
# STEP 4: Core AI Agent function
# This sends phase results to Groq and gets recommendation
# ============================================================
def analyze_phase(phase_result, previous_phases=None,
                  drug_name="Unknown Drug"):
    """
    Sends phase results to AI agent and gets recommendation.

    Parameters:
    - phase_result: dictionary with current phase results
    - previous_phases: list of previous phase results
    - drug_name: name of the drug being tested

    Returns:
    - ai_recommendation: string with full AI analysis
    """

    # Build context from previous phases
    previous_context = ""
    if previous_phases and len(previous_phases) > 0:
        previous_context = "\n\nPREVIOUS PHASE HISTORY:"
        for prev in previous_phases:
            previous_context += f"""
Phase: {prev['phase']}
- Patients enrolled: {prev['total_enrolled']}
- Completed: {prev['total_completed']}
- Dropout rate: {prev['dropout_rate_%']}%
- Adverse events: {prev['total_ae']} ({prev['ae_rate_%']}%)
- Mild AE: {prev['mild_ae']} | Severe AE: {prev['severe_ae']}
- Decision: {prev['decision']}
- Reason: {prev['reason']}
"""

    # Build the user prompt with current phase data
    user_prompt = f"""
DRUG UNDER EVALUATION: {drug_name}

CURRENT PHASE RESULTS:
Phase: {phase_result['phase']}
Patients enrolled: {phase_result['total_enrolled']}
Patients completed: {phase_result['total_completed']}
Patients dropped out: {phase_result['total_dropouts']}
Dropout rate: {phase_result['dropout_rate_%']}%
Total adverse events: {phase_result['total_ae']}
- Mild adverse events: {phase_result['mild_ae']}
- Severe adverse events: {phase_result['severe_ae']}
Adverse event rate: {phase_result['ae_rate_%']}%
Simulation decision: {phase_result['decision']}
Decision reason: {phase_result['reason']}
{previous_context}

Please provide your professional clinical assessment and 
GO/NO-GO recommendation for this phase. If this is not
Phase 1, compare trends with previous phases.
"""

    # Send to Groq API
    print(f"\n🤖 Dr. ARIA analyzing {phase_result['phase']}...")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Groq's fast Llama 3 model
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.3,  # lower = more consistent/professional
        max_tokens=400,  # limit response length
    )

    # Extract the text response
    ai_recommendation = response.choices[0].message.content
    return ai_recommendation


# ============================================================
# STEP 5: Final Trial Summary function
# Gets overall verdict after all phases complete
# ============================================================
def get_final_verdict(all_phase_results, drug_name):
    """
    Gets AI agent's final verdict on the entire trial.

    Parameters:
    - all_phase_results: list of all phase result dicts
    - drug_name: name of the drug

    Returns:
    - final_verdict: string with complete trial assessment
    """

    # Build complete trial summary
    phases_summary = ""
    for result in all_phase_results:
        phases_summary += f"""
{result['phase']}:
- Enrolled: {result['total_enrolled']} patients
- Dropout: {result['dropout_rate_%']}%
- AE Rate: {result['ae_rate_%']}%
- Decision: {result['decision']}
"""

    final_prompt = f"""
COMPLETE CLINICAL TRIAL SUMMARY FOR: {drug_name}

{phases_summary}

Please provide:
1. Overall trial assessment (2-3 sentences)
2. Key safety findings across all phases
3. Efficacy signals observed
4. Final recommendation: APPROVE FOR REGULATORY 
   SUBMISSION / DISCONTINUE / REQUIRES FURTHER STUDY
5. One paragraph suitable for a case study document

Be professional and concise.
"""

    print("\n🤖 Dr. ARIA preparing final verdict...")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": final_prompt
            }
        ],
        temperature=0.3,
        max_tokens=500,
    )

    return response.choices[0].message.content


# ============================================================
# STEP 6: Complete Trial with AI Agent
# Combines simulation engine + AI agent together
# ============================================================
def run_trial_with_ai(drug_name, phase_results_list):
    """
    Runs AI analysis on a complete set of phase results.

    This function takes simulation results and adds
    AI agent analysis to each phase.

    Parameters:
    - drug_name: name of the drug
    - phase_results_list: list of phase result dicts
                          from simulation engine

    Returns:
    - enriched_results: list of results with AI analysis added
    """

    print(f"\n{'#' * 60}")
    print(f"  AI AGENT ANALYSIS: {drug_name}")
    print(f"{'#' * 60}")

    enriched_results = []
    previous_phases = []

    for i, phase_result in enumerate(phase_results_list):

        # Get AI analysis for this phase
        ai_analysis = analyze_phase(
            phase_result=phase_result,
            previous_phases=previous_phases,
            drug_name=drug_name
        )

        # Add AI analysis to result
        enriched = phase_result.copy()
        enriched['ai_analysis'] = ai_analysis

        enriched_results.append(enriched)
        previous_phases.append(phase_result)

        # Display the AI analysis
        print(f"\n{'=' * 60}")
        print(f"DR. ARIA'S ANALYSIS — {phase_result['phase']}")
        print(f"{'=' * 60}")
        print(ai_analysis)

        # If NO-GO stop processing further phases
        if phase_result['decision'] == 'NO-GO':
            print(f"\n🛑 Trial stopped. No further phases.")
            break

    # Get final verdict if trial completed
    if (len(enriched_results) == 3 and
            enriched_results[-1]['decision'] == 'GO'):
        final_verdict = get_final_verdict(
            phase_results_list, drug_name
        )
        print(f"\n{'=' * 60}")
        print("DR. ARIA'S FINAL VERDICT")
        print(f"{'=' * 60}")
        print(final_verdict)

        # Add final verdict to results
        enriched_results.append({
            'phase': 'FINAL VERDICT',
            'drug_name': drug_name,
            'ai_analysis': final_verdict
        })

    elif enriched_results[-1]['decision'] == 'NO-GO':
        stopped_phase = enriched_results[-1]['phase']
        final_verdict = get_final_verdict(
            [r for r in enriched_results
             if r.get('phase') != 'FINAL VERDICT'],
            drug_name
        )
        print(f"\n{'=' * 60}")
        print(f"DR. ARIA'S FINAL VERDICT (Discontinued)")
        print(f"{'=' * 60}")
        print(final_verdict)

        enriched_results.append({
            'phase': 'FINAL VERDICT',
            'drug_name': drug_name,
            'ai_analysis': final_verdict
        })

    return enriched_results


# ============================================================
# STEP 7: Test the AI Agent
# We use pre-defined simulation results to test
# (no need to run simulation engine again)
# ============================================================
if __name__ == "__main__":

    print("🔬 CLINICAL TRIAL AI AGENT — Dr. ARIA")
    print("   Capgemini Internship 2026 — Madhyam")
    print("   Powered by Groq (Llama 3)")

    # ── Test Case 1: Successful drug (all phases pass) ──
    print("\n" + "=" * 60)
    print("TEST CASE 1: DrugX-2026 (Successful Trial)")
    print("=" * 60)

    drugx_results = [
        {
            'phase': 'Phase 1',
            'drug_name': 'DrugX-2026',
            'total_enrolled': 60,
            'total_completed': 57,
            'total_dropouts': 3,
            'dropout_rate_%': 5.0,
            'total_ae': 19,
            'mild_ae': 17,
            'severe_ae': 2,
            'ae_rate_%': 31.7,
            'decision': 'GO',
            'reason': 'All thresholds passed. Trial outcome positive.'
        },
        {
            'phase': 'Phase 2',
            'drug_name': 'DrugX-2026',
            'total_enrolled': 162,
            'total_completed': 131,
            'total_dropouts': 31,
            'dropout_rate_%': 19.1,
            'total_ae': 39,
            'mild_ae': 32,
            'severe_ae': 7,
            'ae_rate_%': 24.1,
            'decision': 'GO',
            'reason': 'All thresholds passed. Trial outcome positive.'
        },
        {
            'phase': 'Phase 3',
            'drug_name': 'DrugX-2026',
            'total_enrolled': 1379,
            'total_completed': 1249,
            'total_dropouts': 130,
            'dropout_rate_%': 9.4,
            'total_ae': 138,
            'mild_ae': 100,
            'severe_ae': 38,
            'ae_rate_%': 10.0,
            'decision': 'GO',
            'reason': 'All thresholds passed. Trial outcome positive.'
        }
    ]

    results1 = run_trial_with_ai("DrugX-2026", drugx_results)

    print("\n" + "=" * 60)

    # ── Test Case 2: Failed drug (stopped at Phase 1) ──
    print("\nTEST CASE 2: CureMax-001 (Failed Trial)")
    print("=" * 60)

    curemax_results = [
        {
            'phase': 'Phase 1',
            'drug_name': 'CureMax-001',
            'total_enrolled': 78,
            'total_completed': 73,
            'total_dropouts': 5,
            'dropout_rate_%': 6.4,
            'total_ae': 29,
            'mild_ae': 20,
            'severe_ae': 9,
            'ae_rate_%': 37.2,
            'decision': 'NO-GO',
            'reason': 'Adverse event rate 37.2% exceeds '
                      'maximum allowed 35.0%'
        }
    ]

    results2 = run_trial_with_ai("CureMax-001", curemax_results)

    # ── Save all AI analyses ──
    print("\n\nSaving AI analyses...")

    all_analyses = []
    for result in results1 + results2:
        if 'ai_analysis' in result:
            all_analyses.append({
                'drug': result.get('drug_name', 'Unknown'),
                'phase': result['phase'],
                'decision': result.get('decision', 'N/A'),
                'ai_analysis': result['ai_analysis']
            })

    analyses_df = pd.DataFrame(all_analyses)
    analyses_df.to_csv(
        '../data/ai_analyses.csv',
        index=False
    )
    print("✅ AI analyses saved to ai_analyses.csv!")

    print("\n" + "=" * 60)
    print("AI AGENT COMPLETE!")
    print("Dr. ARIA successfully analyzed all trials!")
    print("=" * 60)