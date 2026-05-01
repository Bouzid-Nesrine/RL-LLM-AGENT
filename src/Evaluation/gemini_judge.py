"""
gemini_judge.py
Uses Gemini as an LLM judge to score conversation quality across 5 qualitative dimensions.
Requires: pip install google-generativeai
Set your API key: export GEMINI_API_KEY="your-key-here"
"""

import json
import os
import time
import re
from pathlib import Path

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai not installed. Run: pip install google-generativeai")


GEMINI_MODEL = "gemini-1.5-flash"  # fast + cheap; change to gemini-1.5-pro for higher quality

JUDGE_DIMENSIONS = {
    "naturalness": "How natural and human-like does the agent's language sound? (1=robotic/scripted, 5=fully natural)",
    "empathy": "How well does the agent acknowledge and respond to the customer's emotional state? (1=cold/dismissive, 5=highly empathetic)",
    "task_helpfulness": "How well does the agent actually move toward resolving the customer's issue? (1=unhelpful/derailing, 5=directly addresses the problem)",
    "coherence": "Are the agent's responses logically consistent with the conversation context? (1=contradictory/irrelevant, 5=perfectly coherent)",
    "policy_compliance": "Does the agent follow correct support policies (e.g., escalate fraud, gather info before resolving)? (1=violates key policies, 5=fully compliant)"
}


def format_conversation_for_judge(result: dict) -> str:
    """Format a conversation trace into a readable string for the judge."""
    lines = [f"Issue Type: {result['issue_type']}"]
    lines.append(f"Initial Customer Sentiment: {result['initial_sentiment']}")
    lines.append(f"Agent System: {result['agent_name']}")
    lines.append("")
    lines.append(f"Customer: {result['opening_message']}")
    lines.append("")
    
    for turn in result.get("conversation_history", []):
        lines.append(f"[Turn {turn['turn']}]")
        lines.append(f"Agent ({turn['action']}): {turn['agent_says']}")
        lines.append(f"Customer: {turn['customer_says']}")
        lines.append(f"[Sentiment after: {turn['sentiment_after']}]")
        lines.append("")
    
    return "\n".join(lines)


def judge_conversation(result: dict, genai_client) -> dict:
    """
    Ask Gemini to score a single conversation on all 5 dimensions.
    Returns dict with scores and explanations.
    """
    conversation_text = format_conversation_for_judge(result)
    
    prompt = f"""You are an expert evaluator of customer support AI systems. You will evaluate a conversation between an AI customer support agent and a customer.

CONVERSATION TO EVALUATE:
{conversation_text}

OUTCOME: 
- Success: {result['success']}
- Total turns: {result['turns']}  
- Terminal action: {result['terminal_action']}
- Final customer sentiment: {result['final_sentiment']}

Please score this conversation on exactly these 5 dimensions. For each dimension, give:
1. A score from 1 to 5
2. A 1-sentence justification

DIMENSIONS:
1. naturalness - {JUDGE_DIMENSIONS['naturalness']}
2. empathy - {JUDGE_DIMENSIONS['empathy']}
3. task_helpfulness - {JUDGE_DIMENSIONS['task_helpfulness']}
4. coherence - {JUDGE_DIMENSIONS['coherence']}
5. policy_compliance - {JUDGE_DIMENSIONS['policy_compliance']}

Respond ONLY in this exact JSON format, no preamble:
{{
  "naturalness": {{"score": <1-5>, "reason": "<one sentence>"}},
  "empathy": {{"score": <1-5>, "reason": "<one sentence>"}},
  "task_helpfulness": {{"score": <1-5>, "reason": "<one sentence>"}},
  "coherence": {{"score": <1-5>, "reason": "<one sentence>"}},
  "policy_compliance": {{"score": <1-5>, "reason": "<one sentence>"}},
  "overall_comment": "<2-3 sentence overall assessment of this agent's performance>"
}}"""

    try:
        model = genai_client.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai_client.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=600
            )
        )
        raw = response.text.strip()
        
        # Clean up potential markdown fences
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*", "", raw)
        
        scores = json.loads(raw)
        scores["scenario_id"] = result["scenario_id"]
        scores["agent_name"] = result["agent_name"]
        
        # Add composite score
        dims = ["naturalness", "empathy", "task_helpfulness", "coherence", "policy_compliance"]
        scores["composite_score"] = round(
            sum(scores[d]["score"] for d in dims) / len(dims), 2
        )
        
        return scores
    
    except json.JSONDecodeError as e:
        print(f"    ⚠️  JSON parse error for {result['scenario_id']}/{result['agent_name']}: {e}")
        return _fallback_scores(result)
    except Exception as e:
        print(f"    ⚠️  Gemini error for {result['scenario_id']}/{result['agent_name']}: {e}")
        return _fallback_scores(result)


def _fallback_scores(result: dict) -> dict:
    """Return null scores when Gemini is unavailable or errors."""
    return {
        "scenario_id": result["scenario_id"],
        "agent_name": result["agent_name"],
        "naturalness": {"score": None, "reason": "Evaluation failed"},
        "empathy": {"score": None, "reason": "Evaluation failed"},
        "task_helpfulness": {"score": None, "reason": "Evaluation failed"},
        "coherence": {"score": None, "reason": "Evaluation failed"},
        "policy_compliance": {"score": None, "reason": "Evaluation failed"},
        "overall_comment": "Gemini evaluation unavailable",
        "composite_score": None
    }


def run_gemini_evaluation(raw_results_path: str,
                          output_path: str = "results/qualitative_scores.json",
                          api_key: str = None,
                          delay_between_calls: float = 1.5) -> list:
    """
    Load raw results, run Gemini judge on each conversation.
    Returns list of qualitative score dicts.
    """
    if not GEMINI_AVAILABLE:
        print("❌ google-generativeai not available. Install it and set GEMINI_API_KEY.")
        return []

    # Configure Gemini
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        print("❌ GEMINI_API_KEY not set. Export it or pass api_key argument.")
        return []
    
    genai.configure(api_key=key)
    print(f"✅ Gemini configured with model: {GEMINI_MODEL}")

    # Load results
    with open(raw_results_path) as f:
        results = json.load(f)

    # Only judge conversations that have history (skip empty runs)
    results_to_judge = [r for r in results if r.get("conversation_history")]
    print(f"📊 Judging {len(results_to_judge)} conversations...")

    all_scores = []
    for i, result in enumerate(results_to_judge):
        print(f"  [{i+1}/{len(results_to_judge)}] {result['scenario_id']} | {result['agent_name']}")
        scores = judge_conversation(result, genai)
        all_scores.append(scores)
        
        valid = scores.get("composite_score")
        if valid:
            print(f"    composite={valid:.2f} | naturalness={scores['naturalness']['score']} | empathy={scores['empathy']['score']}")
        
        time.sleep(delay_between_calls)  # rate limit

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_scores, f, indent=2)

    print(f"\n✅ Qualitative scores saved to {output_path}")
    return all_scores


if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else None
    run_gemini_evaluation(
        raw_results_path="results/raw_results.json",
        output_path="results/qualitative_scores.json",
        api_key=key
    )