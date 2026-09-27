# Contributing

Please report RedFlow issues in this repository, including the commit, Python/JAX versions, GPU configuration, launch command (with credentials removed), and traceback. Describe the dataset schema and whether the failure occurs before the first training step.

For changes to the training loss or data transforms, run:

```bash
uv run --no-default-groups --group dev pytest tests/test_redflow_losses.py tests/test_data_loader_chunked.py -q
```

The dataset integration test needs `TEST_REAL_ROLLOUT_DIR` pointing to a local chunked dataset; otherwise it is skipped. Include test results and any unverified GPU behavior in your pull request. Do not commit datasets, checkpoints, access tokens, or training logs.

Before submitting or packaging source, run `python scripts/prepare_public_release.py --check`. This check uses only the Python standard library, Git and Bash. It does not run model training. See [release instructions](docs/releasing.md) for creating a source snapshot without local Git history.
