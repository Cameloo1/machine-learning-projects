# Reproducibility Verification

This repo contains independent ML projects with different dependency stacks, data needs, and runtime costs. Verification is contract-based instead of one-size-fits-all.

## Files

- `projects.json` defines each project path, required files, dependency files, artifact globs, install commands, and quick/full commands.
- `scripts/verify_projects.py` reads the manifest and writes local JSON/Markdown reports.
- `reports/reproducibility/` stores generated reports and is ignored by Git.
- `.verify/` stores disposable command workspaces and virtual environments and is ignored by Git.

## Safe Check

```bash
python scripts/verify_projects.py --level quick
```

This checks structure and existing artifacts only:

- required files and folders
- dependency specs
- Python syntax without importing third-party packages
- notebook JSON parseability
- artifact presence
- JSON artifact parseability

No installs or project commands run in this mode.

## Running Commands

```bash
python scripts/verify_projects.py --level quick --run-commands
```

By default, commands run in copied workspaces under `.verify/runs/<timestamp>/<project-id>/`, so generated files do not touch the normal project folders.

Use in-place mode only when you intentionally want to update project artifacts:

```bash
python scripts/verify_projects.py --level quick --run-commands --workspace-mode in-place
```

## Installing Dependencies

```bash
python scripts/verify_projects.py --level quick --install --allow-network
```

The verifier creates one virtual environment per project under `.verify/venvs/<project-id>/`.

## Full Checks

```bash
python scripts/verify_projects.py --level full --install --allow-network --run-commands
```

Full checks may download data, train models, run backtests, or produce explainability artifacts. Expect longer runtimes and larger generated outputs.

## Status Meanings

- `pass`: all applicable checks passed.
- `pass_with_skips`: checks passed, but at least one declared check was skipped.
- `fail`: at least one applicable check failed.

Skips are explicit in the report. Common reasons are disabled command execution, missing network permission, or no command declared for that level.
