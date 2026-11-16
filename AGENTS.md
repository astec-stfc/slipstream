# Instructions for AI agents

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing anything. It is the
canonical guide for workflow, code standards, tests, documentation and the
rules for AI-assisted contributions.

Before you finish, run both CI checks and report the real result:

```bash
python -m pytest
ruff check src tests
```

Keep personal or machine-specific agent settings in git-ignored files
(`.claude/`, `.codex/`, `.agents/`), not here.
