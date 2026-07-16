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

**Behind a TLS-intercepting proxy:** if the build fails with
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

The mount hides the Linux `soccer_sim` extension compiled inside the image, so
code that imports it (e.g. `src.examples.grid_A2C`) needs a Linux `.so` present
in your tree. Compile one into the mount once (and again only when
`soccer_env.cpp` changes); Python edits stay live after that:

```bash
# one-time: build the extension into your working tree
docker run --rm -v "$PWD":/workspace -w /workspace \
  --entrypoint python soccerai setup.py build_ext --inplace

# then iterate freely (add -e WANDB_API_KEY for online logging)
docker run --rm -v "$PWD":/workspace -w /workspace soccerai src.examples.grid_A2C
```

## Running with Weights & Biases

Training scripts (e.g. `src.examples.grid_A2C`) log to [Weights & Biases](https://wandb.ai).
A container is isolated, so authenticate by passing your API key as an environment variable — `wandb` reads `WANDB_API_KEY` automatically and logs in non-interactively.


```bash
export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx 

docker run --rm -e WANDB_API_KEY soccerai src.examples.grid_A2C
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

# Examples (set WANDB_API_KEY first, or WANDB_MODE=offline / disabled — see above)
python -m src.examples.grid_A2C
```

## Tests

Tests live in [src/tests](src/tests) and run with pytest:

```bash
pytest
```
