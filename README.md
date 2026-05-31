# Hermes Agent Optimizer

**40-70% fewer tokens** -- TF-IDF based selective memory retrieval.

## What it does

Hermes Agent appends all memory and user profile entries into the system
prompt on every turn, wasting tokens on irrelevant context. This optimizer:

- **TF-IDF matches** only memory entries relevant to the current query
- **Zero new API calls** (no embedding model, no LLM router)
- **Zero extra dependencies** (pure Python stdlib)
- **Opt-out**: set `selective_retrieval: false` in config to restore
  original behavior
- **Learning cycle preserved**: cron/curator/background_review turns
  still load full memory (hermes-agent v1.39.0+)

## Usage

```bash
# Auto-detect + patch
python optimize.py

# Specify path
python optimize.py --path /path/to/hermes-agent

# Preview changes
python optimize.py --dry-run

# Restore from last backup
python optimize.py --rollback
```

## How it works

| Step | What happens |
|------|-------------|
| 1 | `optimize.py` locates the hermes-agent installation and creates a backup |
| 2 | `select_for_query()` is added to `tools/memory_tool.py` |
| 3 | `agent/memory_router.py` is copied into the agent (TF-IDF engine) |
| 4 | `agent/system_prompt.py` excludes memory from the volatile tier when `selective_retrieval` is enabled |
| 5 | `agent/conversation_loop.py` calls `MemoryRouter.route()` on every turn |
| 6 | `hermes_cli/config.py` gets `selective_retrieval` and `selective_top_k` defaults |

## Requirements

- Python 3.8+
- Hermes Agent (installed)
- No extra pip packages

## Backups

Backups are stored in `~/.hermes-optimizer/backups/`. To restore:

```bash
python optimize.py --rollback
python optimize.py --rollback --backup-dir 20250315_120000
```
