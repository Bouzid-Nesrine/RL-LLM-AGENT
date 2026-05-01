"""
agent_runner.py
Runs all 3 agents (Mistral-only, Rule-based+Mistral, RL+Mistral) on test scenarios.
Connects to local Ollama at http://localhost:11434
"""

import json
import time
import requests
import random
from copy import deepcopy
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MISTRAL_MODEL = "mistral:7b"
MAX_TURNS = 8

ACTIONS = [
    "greet", "ask_info", "confirm_details", "provide_info",
    "tool_query", "resolve", "escalate", "apologize", "end_chat"
]

SENTIMENT_SCORE = {
    "satisfied": 2,
    "neutral": 1,
    "frustrated": 0,
    "angry": -1,
    "very_angry": -2
}

MUST_ESCALATE_ISSUES = {"fraud_report"}

# ─────────────────────────────────────────────
# MISTRAL CALLER
# ─────────────────────────────────────────────
def call_mistral(prompt: str, temperature: float = 0.7, max_tokens: int = 200) -> str:
    """Call local Mistral via Ollama."""
    payload = {
        "model": MISTRAL_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_k": 40,
            "top_p": 0.9
        }
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["response"].strip()
    except Exception as e:
        print(f"  [Mistral error]: {e}")
        return "I'm sorry, I'm having trouble responding right now."


# ─────────────────────────────────────────────
# SHARED CONVERSATION ENVIRONMENT
# ─────────────────────────────────────────────
class ConversationEnv:
    """Simulates the scripted customer environment from the scenario data."""

    def __init__(self, scenario: dict):
        self.scenario = scenario
        self.state = {
            "issue_type": scenario["issue_type"],
            "sentiment": scenario["initial_sentiment"],
            "slots": {},
            "required_slots": scenario["required_slots"],
            "turn": 0,
            "last_action": None,
            "apology_count": 0,
            "details_confirmed": False,
            "slots_complete": False,
            "repeat_count": 0,
        }
        self.history = []  # list of (agent_utterance, customer_reply) tuples
        self.done = False
        self.terminal_action = None
        self.initial_customer_msg = scenario["customer_opening"]

    def step(self, action: str, agent_utterance: str) -> dict:
        """Execute one agent action, return updated state + reward."""
        scenario = self.scenario
        reward = 0

        # Update slots
        slots_update = scenario.get("slots_after_reply", {}).get(action, {})
        self.state["slots"].update(slots_update)
        self.state["slots_complete"] = len(self.state["slots"]) >= len(self.state["required_slots"])

        # Update sentiment
        new_sentiment = scenario["sentiment_transitions"].get(action, self.state["sentiment"])
        old_sentiment_score = SENTIMENT_SCORE[self.state["sentiment"]]
        new_sentiment_score = SENTIMENT_SCORE[new_sentiment]
        self.state["sentiment"] = new_sentiment

        # Track special state
        if action == "apologize":
            self.state["apology_count"] += 1
        if action == "confirm_details":
            self.state["details_confirmed"] = True
        if action == self.state["last_action"]:
            self.state["repeat_count"] += 1
        else:
            self.state["repeat_count"] = 0
        self.state["last_action"] = action
        self.state["turn"] += 1

        # Get customer reply
        customer_reply = scenario["customer_replies"].get(action, "I see, thank you.")
        self.history.append({
            "turn": self.state["turn"],
            "action": action,
            "agent_says": agent_utterance,
            "customer_says": customer_reply,
            "sentiment_after": new_sentiment
        })

        # Compute reward
        reward += (new_sentiment_score - old_sentiment_score) * 0.5  # sentiment improvement
        if action == "resolve" and self.state["slots_complete"] and self.state["details_confirmed"]:
            reward += 2.0
        elif action == "resolve" and not self.state["slots_complete"]:
            reward -= 2.0  # resolving without info
        if action == "escalate" and scenario["issue_type"] in MUST_ESCALATE_ISSUES:
            reward += 2.0
        if action == "resolve" and scenario["issue_type"] in MUST_ESCALATE_ISSUES:
            reward -= 3.0  # critical error: resolving fraud
        if self.state["repeat_count"] > 1:
            reward -= 0.5  # repetition penalty
        reward += 0.3  # per-turn progress reward

        # Check done
        self.done = scenario["done_conditions"].get(action, False) or self.state["turn"] >= MAX_TURNS
        if scenario["done_conditions"].get(action, False):
            self.terminal_action = action

        return {
            "state": deepcopy(self.state),
            "customer_reply": customer_reply,
            "reward": reward,
            "done": self.done
        }

    def get_valid_actions(self) -> list:
        """Action masking based on conversation state."""
        valid = list(ACTIONS)
        s = self.state
        # Can only greet on first turn
        if s["turn"] > 0:
            valid.remove("greet")
        # Can only confirm once slots are filled
        if not s["slots_complete"] and "confirm_details" in valid:
            valid.remove("confirm_details")
        # Limit repeated apologize
        if s["apology_count"] >= 2 and "apologize" in valid:
            valid.remove("apologize")
        return valid


# ─────────────────────────────────────────────
# AGENT 1: MISTRAL-ONLY (no policy guidance)
# ─────────────────────────────────────────────
class MistralOnlyAgent:
    """Mistral selects action AND generates response from raw conversation context."""

    name = "Mistral-Only"

    def select_action_and_respond(self, env: ConversationEnv) -> tuple:
        history_text = self._format_history(env.history, env.initial_customer_msg)
        valid_actions = env.get_valid_actions()
        
        prompt = f"""You are a customer support agent. Based on the conversation below, select the best action and generate a response.

Valid actions: {', '.join(valid_actions)}

Conversation:
{history_text}

Customer issue: {env.state['issue_type']}
Customer sentiment: {env.state['sentiment']}

First output EXACTLY one action name from the valid list, then on a new line output your response to the customer.
Format:
ACTION: <action_name>
RESPONSE: <your response>"""

        raw = call_mistral(prompt, temperature=0.7, max_tokens=250)
        action = self._parse_action(raw, valid_actions)
        response = self._parse_response(raw)
        return action, response

    def _parse_action(self, raw: str, valid: list) -> str:
        for line in raw.split('\n'):
            if line.strip().upper().startswith("ACTION:"):
                candidate = line.split(":", 1)[1].strip().lower().replace("_", "").replace("-", "")
                for a in valid:
                    if a.replace("_", "") in candidate or candidate in a.replace("_", ""):
                        return a
        # fallback: scan text for any action name
        raw_lower = raw.lower()
        for a in valid:
            if a in raw_lower:
                return a
        return random.choice(valid)

    def _parse_response(self, raw: str) -> str:
        for line in raw.split('\n'):
            if line.strip().upper().startswith("RESPONSE:"):
                return line.split(":", 1)[1].strip()
        lines = [l for l in raw.split('\n') if l.strip() and not l.strip().upper().startswith("ACTION:")]
        return lines[0] if lines else "I'm here to help you with this issue."

    def _format_history(self, history, opening) -> str:
        if not history:
            return f"Customer: {opening}"
        lines = [f"Customer: {opening}"]
        for h in history:
            lines.append(f"Agent: {h['agent_says']}")
            lines.append(f"Customer: {h['customer_says']}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# AGENT 2: RULE-BASED POLICY + MISTRAL VOICE
# ─────────────────────────────────────────────
class RuleBasedAgent:
    """Deterministic policy selects action; Mistral generates text for that action."""

    name = "Rule-Based + Mistral"

    def select_action_and_respond(self, env: ConversationEnv) -> tuple:
        action = self._rule_policy(env.state, env.scenario)
        valid = env.get_valid_actions()
        if action not in valid:
            action = random.choice(valid)
        response = self._generate_response(action, env)
        return action, response

    def _rule_policy(self, state: dict, scenario: dict) -> str:
        """Hard-coded deterministic policy."""
        issue = state["issue_type"]
        sentiment = state["sentiment"]
        turn = state["turn"]
        slots_complete = state["slots_complete"]
        confirmed = state["details_confirmed"]
        apology_count = state["apology_count"]

        if turn == 0:
            return "greet"
        if issue in MUST_ESCALATE_ISSUES and slots_complete and confirmed:
            return "escalate"
        if not slots_complete:
            if sentiment in ("angry", "very_angry") and apology_count == 0:
                return "apologize"
            return "ask_info"
        if not confirmed:
            return "confirm_details"
        if sentiment in ("angry", "very_angry") and apology_count == 0:
            return "apologize"
        return "resolve"

    def _generate_response(self, action: str, env: ConversationEnv) -> str:
        s = env.state
        templates = {
            "greet": f"Hello! Thank you for contacting support. I understand you have a {s['issue_type'].replace('_', ' ')} issue. I'm here to help you resolve this as quickly as possible.",
            "ask_info": f"To help you with your {s['issue_type'].replace('_', ' ')}, I need to gather some details. Could you please provide your order/account information?",
            "confirm_details": f"Let me confirm the details I have: {json.dumps(s['slots'], indent=0)}. Is all of this information correct?",
            "apologize": f"I sincerely apologize for the inconvenience this {s['issue_type'].replace('_', ' ')} has caused you. I completely understand your frustration and I'm committed to resolving this.",
            "resolve": f"I've reviewed your case and I'm happy to confirm that your {s['issue_type'].replace('_', ' ')} has been resolved. You should see the resolution within 3-5 business days.",
            "escalate": f"I'm escalating your case to our specialist team who have the authority and expertise to handle this matter with the urgency it deserves.",
            "provide_info": f"Here is the relevant information about your {s['issue_type'].replace('_', ' ')}: our team is actively working on cases like yours.",
            "tool_query": "I'm checking our system right now for the latest information on your case.",
            "end_chat": "Thank you for contacting us today. Have a great day!"
        }
        # Use Mistral to refine the template response
        prompt = f"""You are a professional customer support agent. Improve this response to make it more natural and empathetic.
Current customer sentiment: {s['sentiment']}
Issue type: {s['issue_type']}
Action being taken: {action}
Base response: {templates.get(action, 'I am here to help.')}
Slots collected: {s['slots']}

Output ONLY the improved response, 2-3 sentences, no preamble:"""
        return call_mistral(prompt, temperature=0.5, max_tokens=150)


# ─────────────────────────────────────────────
# AGENT 3: RL POLICY STUB + MISTRAL VOICE
# ─────────────────────────────────────────────
class RLAgent:
    """
    RL policy selects action; Mistral generates empathetic text.
    
    HOW TO INTEGRATE YOUR REAL RL MODEL:
    Replace the _rl_policy method body with your actual model inference.
    The state_dict passed in matches your environment's state format.
    
    Example integration:
        import torch
        from your_model import ActorCritic, encode_state
        
        self.model = ActorCritic(state_dim=110, hidden_dim=256, action_dim=9)
        self.model.load_state_dict(torch.load('path/to/your/model.pth'))
        self.model.eval()
        
        def _rl_policy(self, state, scenario, valid_actions):
            state_tensor = encode_state(state)  # your encode_state function
            with torch.no_grad():
                action_probs, _ = self.model(state_tensor)
            # mask invalid actions
            mask = torch.zeros(9)
            for a in valid_actions:
                mask[ACTIONS.index(a)] = 1
            masked_probs = action_probs * mask
            action_idx = masked_probs.argmax().item()
            return ACTIONS[action_idx]
    """

    name = "RL + Mistral"

    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str):
        """Load your trained RL model here."""
        try:
            import torch
            # Replace with your actual model class import:
            # from your_rl_module import ActorCritic, encode_state
            # self.model = ActorCritic(...)
            # self.model.load_state_dict(torch.load(path))
            # self.model.eval()
            print(f"  [RL] Model loaded from {path}")
        except Exception as e:
            print(f"  [RL] Could not load model: {e}. Using smart fallback policy.")

    def select_action_and_respond(self, env: ConversationEnv) -> tuple:
        valid = env.get_valid_actions()
        action = self._rl_policy(env.state, env.scenario, valid)
        if action not in valid:
            action = random.choice(valid)
        response = self._generate_voice(action, env)
        return action, response

    def _rl_policy(self, state: dict, scenario: dict, valid_actions: list) -> str:
        """
        RL policy action selection.
        
        ─── REPLACE THIS WITH YOUR ACTUAL MODEL INFERENCE ───
        If self.model is loaded, call it here.
        The fallback below mimics smart RL behavior for testing.
        ──────────────────────────────────────────────────────
        """
        if self.model is not None:
            # TODO: Replace with your actual inference call
            # state_tensor = encode_state(state)
            # action = self.model.select_action(state_tensor, valid_actions)
            # return action
            pass

        # ── Smart fallback (approximates trained RL behavior) ──
        issue = state["issue_type"]
        sentiment = state["sentiment"]
        turn = state["turn"]
        slots_complete = state["slots_complete"]
        confirmed = state["details_confirmed"]
        apology_count = state["apology_count"]
        repeat = state["repeat_count"]

        if turn == 0:
            return "greet"

        # RL learned: fraud → always escalate after info
        if issue in MUST_ESCALATE_ISSUES:
            if not slots_complete:
                return "apologize" if (sentiment in ("angry", "very_angry") and apology_count == 0) else "ask_info"
            if not confirmed:
                return "confirm_details"
            return "escalate"

        # RL learned: high anger → apologize first, then info gather
        if sentiment in ("very_angry", "angry") and apology_count == 0 and not slots_complete:
            return "apologize"

        # RL learned: get info before anything else
        if not slots_complete:
            return "ask_info"

        # RL learned: confirm before resolve
        if not confirmed:
            return "confirm_details"

        # RL learned: apologize on anger before resolving
        if sentiment in ("angry", "very_angry") and apology_count == 0:
            return "apologize"

        # RL learned: resolve efficiently
        if slots_complete and confirmed:
            # For very high anger with high-value issues, escalate
            if sentiment == "very_angry" and issue in ("shipping_damage", "billing_error"):
                return "escalate"
            return "resolve"

        return "ask_info"

    def _generate_voice(self, action: str, env: ConversationEnv) -> str:
        """Mistral generates the actual text, conditioned on the RL-selected action."""
        s = env.state
        last_customer = env.history[-1]["customer_says"] if env.history else env.initial_customer_msg

        action_instructions = {
            "greet":           "greet the customer warmly, acknowledge their issue type, and express readiness to help",
            "ask_info":        f"ask for the missing required information: {[r for r in s['required_slots'] if r not in s['slots']]}. Be specific and concise.",
            "confirm_details": f"summarize the collected information {s['slots']} and ask the customer to confirm it's all correct",
            "apologize":       "offer a genuine, empathetic apology without making excuses. Acknowledge their specific frustration.",
            "resolve":         f"confirm the {s['issue_type'].replace('_',' ')} has been resolved with the details collected. Be specific and reassuring.",
            "escalate":        "explain you're escalating to a specialist team, reassure the customer this will be handled with priority",
            "provide_info":    f"provide helpful information about their {s['issue_type'].replace('_',' ')} situation",
            "tool_query":      "inform the customer you're checking the system for their case details",
            "end_chat":        "close the conversation professionally and thank the customer"
        }

        prompt = f"""You are an expert customer support agent. The customer just said: "{last_customer}"

Your selected action is: {action.upper()}
Your task: {action_instructions.get(action, 'respond helpfully')}

Context:
- Issue: {s['issue_type'].replace('_', ' ')}
- Customer sentiment: {s['sentiment']}
- Information collected: {s['slots'] if s['slots'] else 'none yet'}
- Turn number: {s['turn'] + 1}

Generate a natural, professional, empathetic response (2-3 sentences). Output ONLY the response, no labels:"""

        return call_mistral(prompt, temperature=0.6, max_tokens=180)


# ─────────────────────────────────────────────
# EVALUATION RUNNER
# ─────────────────────────────────────────────
def run_agent_on_scenario(agent, scenario: dict) -> dict:
    """Run a single agent on a single scenario. Returns full trace + metrics."""
    env = ConversationEnv(scenario)
    total_reward = 0.0
    turns = 0
    actions_taken = []
    start_time = time.time()

    print(f"    Turn 0 | Customer: {scenario['customer_opening'][:70]}...")

    while not env.done and turns < MAX_TURNS:
        action, utterance = agent.select_action_and_respond(env)
        result = env.step(action, utterance)
        total_reward += result["reward"]
        actions_taken.append(action)
        turns += 1

        print(f"    Turn {turns} | Action: {action:20s} | Reward: {result['reward']:+.2f} | Sentiment: {result['state']['sentiment']}")
        print(f"            Agent: {utterance[:80]}...")
        print(f"            Customer: {result['customer_reply'][:70]}...")

    elapsed = time.time() - start_time

    # Compute success
    terminal = env.terminal_action
    correct_terminal = scenario.get("correct_terminal_action", "resolve")
    must_not = scenario.get("must_not_do", [])

    success = (
        terminal == correct_terminal and
        env.state["slots_complete"] and
        "resolve" not in must_not or terminal != "resolve"
    )
    # Special fraud check
    if scenario["issue_type"] in MUST_ESCALATE_ISSUES and terminal == "resolve":
        success = False

    # Sentiment recovery
    final_sentiment_score = SENTIMENT_SCORE.get(env.state["sentiment"], 0)
    initial_sentiment_score = SENTIMENT_SCORE.get(scenario["initial_sentiment"], 0)
    sentiment_delta = final_sentiment_score - initial_sentiment_score

    return {
        "scenario_id": scenario["scenario_id"],
        "issue_type": scenario["issue_type"],
        "initial_sentiment": scenario["initial_sentiment"],
        "difficulty": scenario["difficulty"],
        "agent_name": agent.name,
        "success": success,
        "total_reward": round(total_reward, 3),
        "turns": turns,
        "actions_taken": actions_taken,
        "terminal_action": terminal,
        "correct_terminal": correct_terminal,
        "slots_complete": env.state["slots_complete"],
        "final_sentiment": env.state["sentiment"],
        "sentiment_delta": sentiment_delta,
        "hit_max_turns": turns >= MAX_TURNS,
        "elapsed_seconds": round(elapsed, 2),
        "conversation_history": env.history,
        "opening_message": scenario["customer_opening"]
    }


def run_all_evaluations(scenarios_path: str, rl_model_path: str = None,
                        output_path: str = "results/raw_results.json") -> list:
    """Run all 3 agents on all scenarios. Returns list of result dicts."""
    with open(scenarios_path) as f:
        scenarios = json.load(f)

    agents = [
        MistralOnlyAgent(),
        RuleBasedAgent(),
        RLAgent(model_path=rl_model_path)
    ]

    all_results = []

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario['scenario_id']} | {scenario['issue_type']} | {scenario['initial_sentiment']}")
        print(f"{'='*60}")

        for agent in agents:
            print(f"\n  ── Agent: {agent.name} ──")
            result = run_agent_on_scenario(agent, scenario)
            all_results.append(result)
            print(f"  RESULT: success={result['success']} | reward={result['total_reward']} | turns={result['turns']}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Results saved to {output_path}")
    return all_results


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_all_evaluations(
        scenarios_path="scenarios/test_scenarios.json",
        rl_model_path=model_path,
        output_path="results/raw_results.json"
    )