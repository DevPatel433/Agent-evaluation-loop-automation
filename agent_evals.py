"""
Agent evals: TRACK -> MEASURE -> IMPROVE

  AI AGENT -> EVALUATE (accuracy / hallucination / quality) -> IMPROVE -> back to AI AGENT

Run offline with the built-in mock agent:   python agent_evals.py
Run against a real model:                   python agent_evals.py --live
  (needs `pip install anthropic` and ANTHROPIC_API_KEY set)
"""
import argparse, json, re, time
from dataclasses import dataclass, asdict

# ---------------------------------------------------------------- test set
CONTEXT = (
    "Acme Corp refund policy: Customers may return items within 30 days of delivery "
    "for a full refund. Digital goods are non-refundable. Shipping is free on orders "
    "over 50 dollars. Support is open Monday to Friday, 9am to 5pm GMT."
)

CASES = [
    {"q": "How long do I have to return an item?", "expect": ["30 days"]},
    {"q": "Can I get a refund on an ebook?", "expect": ["non-refundable"]},
    {"q": "When is shipping free?", "expect": ["50"]},
    {"q": "What are the support hours?", "expect": ["9am", "5pm"]},
    {"q": "Which days is support open?", "expect": ["monday", "friday"]},
    {"q": "Do I get a full refund on returns?", "expect": ["full refund"]},
]

BASE_PROMPT = "You are a helpful customer support assistant."

# Improvement rules, keyed by the metric they fix.
RULES = {
    "hallucination": "Answer ONLY from the provided context. If it is missing, say you don't know.",
    "quality": "Be concise: one or two direct sentences, no hedging.",
    "accuracy": "Double-check every number and day against the context before answering.",
}

# ---------------------------------------------------------------- agents
class MockAgent:
    """Deterministic stand-in so the loop runs offline. Gets better as rules are added."""
    GOLD = [
        "Returns are accepted within 30 days of delivery.",
        "Digital goods such as ebooks are non-refundable.",
        "Shipping is free on orders over 50 dollars.",
        "Support is open 9am to 5pm GMT.",
        "Support is open Monday to Friday.",
        "Yes, returns get a full refund.",
    ]
    def __call__(self, system, question, context):
        i = next(n for n, c in enumerate(CASES) if c["q"] == question)
        ans = self.GOLD[i]
        if "ONLY from the provided context" not in system and i % 2 == 0:
            ans += " According to a 2021 industry study, most customers prefer store credit."
        if "concise" not in system and i % 3 == 0:
            ans = "I think, as an AI, maybe this is right: " + ans + " " + "Please let me know if anything else is unclear. " * 4
        if "Double-check" not in system and i == 4:
            ans = "Support is open Tuesday to Saturday."
        return ans

class LiveAgent:
    def __init__(self, model="claude-sonnet-4-5"):
        import anthropic
        self.client, self.model = anthropic.Anthropic(), model
    def __call__(self, system, question, context):
        r = self.client.messages.create(
            model=self.model, max_tokens=300, system=system,
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}])
        return r.content[0].text

# ---------------------------------------------------------------- evaluators
def words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3 or w.isdigit()}

def accuracy(answer, expect):
    """1.0 if every expected fact appears in the answer."""
    return float(all(e.lower() in answer.lower() for e in expect))

def hallucination(answer, question, context):
    """Share of sentences not supported by the context (lower is better)."""
    support = words(context) | words(question)
    # filler/hedging isn't a factual claim (quality() penalises it instead)
    answer = re.sub(r"^.*?(as an ai|i think)[^:]*:\s*", "", answer, flags=re.I)
    sents = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip())
             if words(s) and not re.search(r"let me know|anything else", s, re.I)]
    if not sents:
        return 0.0
    bad = sum(1 for s in sents if len(words(s) & support) / len(words(s)) < 0.6)
    return bad / len(sents)

def quality(answer):
    """Heuristic rubric: concise, direct, no hedging/filler (0-1)."""
    score = 1.0
    if len(answer.split()) > 40: score -= 0.4
    if re.search(r"\b(as an ai|i think|maybe|perhaps)\b", answer, re.I): score -= 0.3
    if re.search(r"let me know|anything else", answer, re.I): score -= 0.2
    return max(score, 0.0)

# ---------------------------------------------------------------- loop
@dataclass
class Result:
    q: str; answer: str; accuracy: float; hallucination: float; quality: float

def run_suite(agent, system):
    out = []
    for c in CASES:
        a = agent(system, c["q"], CONTEXT)
        out.append(Result(c["q"], a, accuracy(a, c["expect"]),
                          hallucination(a, c["q"], CONTEXT), quality(a)))
    return out

def summarize(res):
    n = len(res)
    return {"accuracy": sum(r.accuracy for r in res) / n,
            "hallucination": sum(r.hallucination for r in res) / n,
            "quality": sum(r.quality for r in res) / n}

def worst_metric(m):
    # normalise to "badness" so all three are comparable
    bad = {"accuracy": 1 - m["accuracy"], "hallucination": m["hallucination"], "quality": 1 - m["quality"]}
    return max(bad, key=bad.get), bad

def passed(m, acc=0.95, hal=0.05, qual=0.9):
    return m["accuracy"] >= acc and m["hallucination"] <= hal and m["quality"] >= qual

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default="eval_history.json")
    args = ap.parse_args()

    agent = LiveAgent() if args.live else MockAgent()
    system, history = BASE_PROMPT, []

    for rnd in range(args.rounds + 1):
        res = run_suite(agent, system)                      # AI AGENT -> EVALUATE
        m = summarize(res)
        history.append({"round": rnd, "prompt": system, "metrics": m,
                        "results": [asdict(r) for r in res], "ts": time.time()})  # TRACK
        print(f"Round {rnd}: accuracy {m['accuracy']:.0%} | "
              f"hallucination {m['hallucination']:.1%} | quality {m['quality']:.0%}")
        if passed(m):
            print("Targets met."); break
        _, bad = worst_metric(m)                            # IMPROVE
        todo = [k for k in sorted(bad, key=bad.get, reverse=True) if bad[k] > 0 and RULES[k] not in system]
        if not todo:
            print("No new rule to add."); break
        metric = todo[0]
        rule = RULES[metric]
        system += " " + rule
        print(f"  -> weakest: {metric}; adding rule: {rule}")

    json.dump(history, open(args.out, "w"), indent=2)
    print(f"\nHistory saved to {args.out}")
    print("Final prompt:", system)

if __name__ == "__main__":
    main()
