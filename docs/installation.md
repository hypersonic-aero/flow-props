# Installation

## From PyPI

```bash
pip install flow-props
```

To upgrade an existing installation:

```bash
pip install --upgrade flow-props
```

## From Source

```bash
git clone https://github.com/hypersonic-aero/flow-props.git
cd flow-props
pip install -e .
```

## Optional Extras

For development (tests, linting, and docs):

```bash
pip install -e ".[dev]"
```

Includes:

- [pytest](https://docs.pytest.org/) and [pytest-cov](https://pytest-cov.readthedocs.io/) for testing
- [ruff](https://docs.astral.sh/ruff/) for linting
- [zensical](https://zensical.org/) for building the documentation

## Verify Installation

```bash
flow-props --version
```

Or verify the Python import:

```python
import flow_props
print(flow_props.__version__)
```
