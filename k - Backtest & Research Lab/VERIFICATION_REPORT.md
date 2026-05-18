# Backtest Lab Verification Snapshot

## Scope

This file records the intended verification surface for Backtest Lab. Refresh it after running the test suite and required experiments.

## Required Checks

```bash
pytest -q -ra -W error::FutureWarning -W error::UserWarning -W error::DeprecationWarning
python scripts/run_required_experiments.py
```

## Expected Evidence

- all tests pass
- required configs run successfully
- each run writes metrics, returns, weights, trades, diagnostics, metadata, and plots
- data-integrity JSON exists for runs that validate input data
- generated `report.html` files are local artifacts, not durable source docs

## Last Known Repo-Level Check

The root verifier can confirm structure and existing artifacts without installing dependencies:

```bash
python scripts\verify_projects.py --project k --level quick
```

For command execution, use:

```bash
python scripts\verify_projects.py --project k --level quick --install --allow-network --run-commands
```

## Notes

Do not claim strategy performance, benchmark superiority, or production readiness from this file alone. Use the concrete run artifacts and test output from the current machine.
