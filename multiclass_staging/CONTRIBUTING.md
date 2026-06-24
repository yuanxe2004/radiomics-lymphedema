# Contributing

This repository is primarily intended to document and reproduce the reported model-development workflow.

Before committing changes:

1. Do not commit private data, trained model files, or identifiable clinical information.
2. Run a syntax check:

```bash
python -m compileall src scripts
```

3. Keep configuration changes in local YAML files rather than editing source defaults when possible.
