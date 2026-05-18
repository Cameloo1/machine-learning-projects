# RL Sandbox Analysis

## Summary

The sandbox demonstrates a complete RL trading workflow: data loading, feature engineering, custom environment, DQN training, validation checkpoints, and test-set evaluation.

## Current Read

The saved experiments are useful evidence that the pipeline runs end to end. They should not be interpreted as evidence of a profitable strategy. RL trading results are highly sensitive to data window, seed, reward design, costs, and market regime.

## What To Trust

- the environment contract
- generated configs and logs
- existence of saved model/checkpoint artifacts
- repeatable evaluation outputs

## What Needs More Work

- deterministic quick smoke tests
- smaller default training profile for CI
- stronger baseline comparison
- repeated-seed evaluation
- clearer artifact cleanup policy for experiment folders

## Recommended Next Step

Add a fast verification command that generates synthetic data, runs a tiny training loop, and checks for `config.json`, `training_log.csv`, `best_model.zip`, and `test_metrics.json`.
