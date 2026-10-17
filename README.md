# Agent Evals

A minimal **Track → Measure → Improve** loop for AI agents.

```
AI AGENT ──► EVALUATE (accuracy · hallucination · quality) ──► IMPROVE ──┐
   ▲                                                                      │
   └──────────────────────────────────────────────────────────────────────┘
```

Each round the agent answers a test suite, three metrics are scored, results are
logged to JSON, and the system prompt is patched to fix the weakest metric.
The loop stops when targets are met or no new fixes remain.

![Core loop](docs/diagrams/01_core_loop.png)

📄 Full write-up: [`docs/Agent_Evals_Report.pdf`](docs/Agent_Evals_Report.pdf) (4 pages, with all flowcharts in `docs/diagrams/`).

## Metrics

| Metric | How it's measured | Target |
|---|---|---|
| Accuracy | Every expected fact appears in the answer | ≥ 95% |
| Hallucination | Share of sentences not supported by the context | ≤ 5% |
| Quality | Rubric: concise, direct, no hedging or filler | ≥ 90% |

## Quick start

```bash
git clone <your-repo-url> && cd agent-evals
python agent_evals.py            # offline, uses a deterministic mock agent
```

![Results](docs/diagrams/05_results.png)

Example output (see `examples/sample_run.txt`):

```
Round 0: accuracy 83% | hallucination 33.3% | quality 70%
Round 1: accuracy 83% | hallucination 16.7% | quality 70%
Round 2: accuracy 83% | hallucination 16.7% | quality 100%
Round 3: accuracy 100% | hallucination 0.0% | quality 100%
Targets met.
```

### Run against a real model

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key
python agent_evals.py --live --rounds 5
```

## Use your own agent

Edit `CONTEXT` and `CASES` at the top of `agent_evals.py`, or pass any callable
`agent(system_prompt, question, context) -> str` to `run_suite`.

## Tests

```bash
python -m unittest -v
```

## Roadmap

- LLM-as-judge for hallucination and quality
- Dataset loading from JSONL
- LLM-written prompt revisions instead of fixed rules
- Regression tracking across commits

## License

MIT © 2026 Dev Patel
