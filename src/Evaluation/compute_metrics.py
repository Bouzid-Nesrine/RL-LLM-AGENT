"""
compute_metrics.py
Aggregates raw results into structured quantitative metrics for all 3 agents.
"""

import json
from collections import defaultdict
from pathlib import Path


AGENT_ORDER = ["Mistral-Only", "Rule-Based + Mistral", "RL + Mistral"]
AGENT_COLORS = {
    "Mistral-Only": "#6C757D",
    "Rule-Based + Mistral": "#0D6EFD",
    "RL + Mistral": "#198754"
}
AGENT_SHORT = {
    "Mistral-Only": "Mistral",
    "Rule-Based + Mistral": "Rule-Based",
    "RL + Mistral": "RL+Mistral"
}

DIFFICULTY_ORDER = ["easy", "medium", "hard"]
SENTIMENT_SCORE = {
    "satisfied": 2, "neutral": 1, "frustrated": 0, "angry": -1, "very_angry": -2
}


def load_results(raw_path: str, qual_path: str = None) -> tuple:
    with open(raw_path) as f:
        raw = json.load(f)
    qual = []
    if qual_path and Path(qual_path).exists():
        with open(qual_path) as f:
            qual = json.load(f)
    return raw, qual


def compute_agent_summary(results: list) -> dict:
    """Per-agent aggregate metrics."""
    by_agent = defaultdict(list)
    for r in results:
        by_agent[r["agent_name"]].append(r)

    summary = {}
    for agent in AGENT_ORDER:
        runs = by_agent.get(agent, [])
        if not runs:
            continue
        n = len(runs)
        successes = [r for r in runs if r["success"]]
        
        summary[agent] = {
            "n_scenarios": n,
            "success_rate": round(len(successes) / n * 100, 1),
            "avg_reward": round(sum(r["total_reward"] for r in runs) / n, 3),
            "avg_turns": round(sum(r["turns"] for r in runs) / n, 2),
            "avg_sentiment_delta": round(
                sum(r["sentiment_delta"] for r in runs) / n, 3
            ),
            "escalation_rate": round(
                sum(1 for r in runs if r["terminal_action"] == "escalate") / n * 100, 1
            ),
            "resolution_rate": round(
                sum(1 for r in runs if r["terminal_action"] == "resolve") / n * 100, 1
            ),
            "hit_max_turns_rate": round(
                sum(1 for r in runs if r["hit_max_turns"]) / n * 100, 1
            ),
            "slots_complete_rate": round(
                sum(1 for r in runs if r["slots_complete"]) / n * 100, 1
            ),
            "avg_elapsed_sec": round(sum(r["elapsed_seconds"] for r in runs) / n, 2)
        }
    return summary


def compute_per_issue(results: list) -> dict:
    """Success rate per issue type per agent."""
    by_agent_issue = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_agent_issue[r["agent_name"]][r["issue_type"]].append(r)

    issues = sorted(set(r["issue_type"] for r in results))
    per_issue = {}
    for agent in AGENT_ORDER:
        per_issue[agent] = {}
        for issue in issues:
            runs = by_agent_issue[agent].get(issue, [])
            if runs:
                per_issue[agent][issue] = round(
                    sum(1 for r in runs if r["success"]) / len(runs) * 100, 1
                )
            else:
                per_issue[agent][issue] = None
    return per_issue, issues


def compute_per_sentiment(results: list) -> dict:
    """Success rate per initial sentiment per agent."""
    sentiments = ["neutral", "frustrated", "angry", "very_angry"]
    by_agent_sent = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_agent_sent[r["agent_name"]][r["initial_sentiment"]].append(r)

    per_sent = {}
    for agent in AGENT_ORDER:
        per_sent[agent] = {}
        for s in sentiments:
            runs = by_agent_sent[agent].get(s, [])
            if runs:
                per_sent[agent][s] = round(
                    sum(1 for r in runs if r["success"]) / len(runs) * 100, 1
                )
            else:
                per_sent[agent][s] = None
    return per_sent, sentiments


def compute_per_difficulty(results: list) -> dict:
    """Success rate per difficulty per agent."""
    by_agent_diff = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_agent_diff[r["agent_name"]][r["difficulty"]].append(r)

    per_diff = {}
    for agent in AGENT_ORDER:
        per_diff[agent] = {}
        for d in DIFFICULTY_ORDER:
            runs = by_agent_diff[agent].get(d, [])
            if runs:
                per_diff[agent][d] = round(
                    sum(1 for r in runs if r["success"]) / len(runs) * 100, 1
                )
            else:
                per_diff[agent][d] = None
    return per_diff


def compute_action_distribution(results: list) -> dict:
    """Action usage frequency per agent."""
    by_agent = defaultdict(list)
    for r in results:
        by_agent[r["agent_name"]].extend(r["actions_taken"])

    dist = {}
    from collections import Counter
    for agent in AGENT_ORDER:
        acts = by_agent[agent]
        total = len(acts)
        counts = Counter(acts)
        dist[agent] = {
            a: round(counts.get(a, 0) / total * 100, 1) if total > 0 else 0
            for a in ["greet","ask_info","confirm_details","provide_info",
                      "tool_query","resolve","escalate","apologize","end_chat"]
        }
    return dist


def compute_qualitative_summary(qual_scores: list) -> dict:
    """Average Gemini judge scores per agent."""
    by_agent = defaultdict(list)
    for s in qual_scores:
        if s.get("composite_score") is not None:
            by_agent[s["agent_name"]].append(s)

    dims = ["naturalness", "empathy", "task_helpfulness", "coherence", "policy_compliance"]
    qual_summary = {}
    for agent in AGENT_ORDER:
        scores = by_agent.get(agent, [])
        if not scores:
            qual_summary[agent] = {d: None for d in dims}
            qual_summary[agent]["composite"] = None
            continue
        qual_summary[agent] = {
            d: round(sum(s[d]["score"] for s in scores if s[d]["score"]) / len(scores), 2)
            for d in dims
        }
        qual_summary[agent]["composite"] = round(
            sum(qual_summary[agent][d] for d in dims if qual_summary[agent][d]) / len(dims), 2
        )
    return qual_summary


def compute_sentiment_recovery(results: list) -> dict:
    """Track sentiment trajectory per agent across turns."""
    by_agent = defaultdict(lambda: defaultdict(list))
    for r in results:
        for h in r.get("conversation_history", []):
            turn = h["turn"]
            score = SENTIMENT_SCORE.get(h["sentiment_after"], 0)
            by_agent[r["agent_name"]][turn].append(score)

    recovery = {}
    for agent in AGENT_ORDER:
        turns_data = by_agent[agent]
        max_turn = max(turns_data.keys()) if turns_data else 0
        recovery[agent] = {
            t: round(sum(turns_data[t]) / len(turns_data[t]), 3)
            for t in range(1, max_turn + 1)
            if turns_data.get(t)
        }
    return recovery


def compile_all_metrics(raw_path: str, qual_path: str = None,
                         output_path: str = "results/metrics.json") -> dict:
    """Main function: compile everything into a single metrics dict."""
    raw, qual = load_results(raw_path, qual_path)

    per_issue, issues = compute_per_issue(raw)
    per_sentiment, sentiments = compute_per_sentiment(raw)

    metrics = {
        "agent_summary": compute_agent_summary(raw),
        "per_issue_type": per_issue,
        "issue_types": issues,
        "per_sentiment": per_sentiment,
        "sentiments": sentiments,
        "per_difficulty": compute_per_difficulty(raw),
        "action_distribution": compute_action_distribution(raw),
        "qualitative_summary": compute_qualitative_summary(qual) if qual else {},
        "sentiment_recovery": compute_sentiment_recovery(raw),
        "agent_colors": AGENT_COLORS,
        "agent_short": AGENT_SHORT,
        "agent_order": AGENT_ORDER,
        "raw_results": raw,
        "qualitative_scores": qual
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Metrics compiled → {output_path}")
    return metrics


if __name__ == "__main__":
    compile_all_metrics(
        raw_path="results/raw_results.json",
        qual_path="results/qualitative_scores.json",
        output_path="results/metrics.json"
    )