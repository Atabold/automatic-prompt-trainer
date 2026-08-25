# Prompt Trainer

A workbench for tuning the system prompt of a local GGUF text-formatter model.

A small llama.cpp model cleans up messy spoken/voice-input text. A cloud LLM of your choice
judges every output against a multilingual test suite, proposes better prompts, and helps you
curate the configuration by conversation. Every run is logged, scored, and kept for comparison.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
python run.py
```

Then, in the browser:

1. **Settings** — choose a provider type, pick a provider, paste your API key, hit **Test connection**.
2. **Models** — paste a Hugging Face repo id or URL, pick a quantization, download it, set it active.
3. **Format** — try the current prompt on one piece of text.
4. **Optimizer** — run the improvement loop and watch it live.

## Providers

The judge, optimizer, and assistant all run on one OpenAI-compatible endpoint of your choosing.
Settings groups them into four types:

| Type | Providers |
| --- | --- |
| **API providers** — first-party endpoints from the labs that build the models | Mistral, Anthropic, DeepSeek, Qwen, OpenAI, Grok (xAI), Kimi (Moonshot), Meta |
| **Routers & gateways** — one key, many vendors | [Cortecs](https://cortecs.ai/) (EU-hosted, GDPR-focused, 150+ endpoints), Groq, Together AI, OpenRouter |
| **Local** — nothing leaves your machine | Ollama |
| **Custom** | any other OpenAI-compatible endpoint |

Model lists in the dropdown are a starting point — **Refresh list** pulls the provider's live
catalogue from its `/models` endpoint, which matters most for routers whose lineup changes often.

## Tabs

| Tab | What it does |
| --- | --- |
| **Format** | Run the local model on one input; shows timing, detected language, filler check. |
| **Optimizer** | Iteratively improves the prompt. Live progress, per-generation scores, per-test cards, a working Stop button. |
| **Test suite** | Inspect and edit the cases every prompt is scored against; restore any archived configuration. |
| **History** | Every past run with its baseline, best score, improvement, and event log. |
| **Models** | Download any GGUF repo from Hugging Face, choose the quant, switch or delete the active model. |
| **Assistant** | Change prompts, criteria, weights, and test cases by talking to the LLM. It edits through tools, so changes are surgical and validated. |
| **Settings** | Provider, API key, base URL, model (with per-role overrides), Hugging Face token, app log. |

## Command line

```bash
python cli.py optimize --iterations 3 --sample 30
python cli.py download https://huggingface.co/google/gemma-4-E2B-it
python cli.py download https://huggingface.co/owner/repo/blob/main/model.Q4_K_M.gguf
python cli.py models
python cli.py runs
```

## Layout

```
app/
  main.py            Gradio app composition and the status header
  core/              paths, logging, config store, secrets store
  llm/               provider registry and the OpenAI-compatible client factory
  hf/                Hugging Face download, local model registry, llama.cpp loader
  evaluation/        scoring: language, filler, LLM judge
  optimizer/         the optimization loop and per-run artifact store
  assistant/         tool schemas and the tool-calling chat loop
  ui/                theme, shared HTML components, one module per tab
data/
  config.json        prompts, criteria, weights, test cases, run settings
  secrets.json       API keys — gitignored, never archived
  seed_test_cases.py fallback suite used only when no config exists
  archive/           automatic snapshot of the config before every save
  test_runs/         one folder per run: summary, history, events
models/              downloaded .gguf weights
logs/app.log         rotating application log
```

## How scoring works

Each test case output is scored on three signals, combined using the weights in
**Test suite → Weights**:

- **Language** — the output must be in the same language as the input (`langdetect`).
- **Filler** — no filler words survive (regex across the suite's languages).
- **Judge** — a cloud LLM rates 0–100 against your `evaluation_criteria` text.

Judge results are cached per (input, expected, output), so re-evaluating an unchanged
output costs nothing. Set **Test cases per evaluation** to a small number while iterating —
it is the single biggest lever on run cost and time.

## Notes

- API keys live in `data/secrets.json`, never in `config.json` — the config is snapshotted
  into `data/archive/` on every save, and secrets must not be copied into those snapshots.
- Environment variables (`MISTRAL_API_KEY`, `HF_TOKEN`, …) still work as a fallback for
  headless use; see `.env.example`.
- Set `APP_ENV=DEV` for debug-level console logging. `logs/app.log` always gets everything.
