# Baseline status

Each baseline was checked in its own Python 3.12 environment next to the
newest JAX. The rungs are:

1. installs next to current JAX;
2. every module imports (`python tools/import_check.py <directory>`);
3. its unit tests pass;
4. a few training steps run on small or synthetic data.

Only files under `jaxpruner/` were changed to get here. Nothing was run on
ImageNet, C4 or Atari at full scale.

| Baseline | Rung reached | Versions | Command for the highest rung | What stops the next rung |
|---|---|---|---|---|
