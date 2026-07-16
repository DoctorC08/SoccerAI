# SoccerAI

A Multi-Agent Reinforcement Learning (MARL) framework for soccer environments.
The soccer simulation core is written in C++ and exposed to Python via
[pybind11](https://pybind11.readthedocs.io/); training, buffers, and agents are
implemented in PyTorch.

## Requirements

- Python >= 3.10
- A C++17 compiler (needed to build the `soccer_sim` extension)
- Docker (optional, but the recommended way to get a reproducible environment)

## Docker (recommended)

### Build

```bash
docker build -t soccerai .
```

**Behind a corporate TLS-intercepting proxy:** if the build fails with
`SSL: CERTIFICATE_VERIFY_FAILED` / `self-signed certificate in certificate chain`,
the slim base image doesn't trust your proxy's CA. Pass trusted hosts to pip:

```bash
docker build -t soccerai \
  --build-arg PIP_EXTRA="--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org" \
  .
```

### Run

The image's entrypoint is `python -m`, so the argument you pass is the module to
run. The default command runs the test suite:

```bash
# Run tests (the default command)
docker run --rm soccerai

# Run pytest explicitly
docker run --rm soccerai pytest

# Run any module, e.g. a training script under src/examples
docker run --rm soccerai src.examples.grid_A2C
```

To iterate on the code without rebuilding, mount your working tree over the
editable install:

```bash
docker run --rm -v "$PWD":/workspace -w /workspace soccerai pytest
```

## Local install

```bash
# Install with dev dependencies (pytest) in editable mode.
# This also compiles the C++ soccer_sim extension.
pip install -e ".[dev]"
```

### Run locally

```bash
# Tests
pytest

# Examples (see the note above — these currently need a fix before they run)
python -m src.examples.grid_A2C
```

## Tests

Tests live in [src/tests](src/tests) and run with pytest:

```bash
pytest
```
