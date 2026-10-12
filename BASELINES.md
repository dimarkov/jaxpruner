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
| `baselines/scenic` | 4 | jax 0.11.2, flax 0.12.10, tensorflow 2.21.0 | `TFDS_DATA_DIR=/tmp/jaxpruner-smoke/tfds-scenic .venv-scenic/bin/python -m jaxpruner.baselines.scenic.main --config=jaxpruner/baselines/scenic/configs/imagenet_resnet_config.py:runlocal --config.dataset_name=cifar10 --config.batch_size=256 --config.num_training_epochs=1 --config.sparsity_config.algorithm=magnitude --config.sparsity_config.update_freq=10 --config.sparsity_config.update_start_step=50 --config.sparsity_config.update_end_step=150 --workdir=/tmp/jaxpruner-smoke/scenic` | — (195 steps, exit 0 in 4 of 4 runs). The environment also needs `tensorboard` and `importlib_resources`, which the install does not pull in. Only the ResNet config was run: `model_name=plainvit` cannot be imported, because the scenic package that pip builds leaves out `scenic/projects/baselines/plainvit`, so `main.py` now imports it only when that model is asked for. |
| `projects/activation_sparsity` | 4, not reliably | jax 0.11.2, flax 0.12.10, tensorflow 2.21.0 | `TFDS_DATA_DIR=/tmp/jaxpruner-smoke/tfds-scenic .venv-scenic/bin/python -m jaxpruner.projects.activation_sparsity.scenic.main --config=jaxpruner/projects/activation_sparsity/scenic/configs/imagenet_resnet_config.py:runlocal --config.dataset_name=cifar10 --config.batch_size=256 --config.num_training_epochs=1 --workdir=/tmp/jaxpruner-smoke/activation_sparsity` | The command ran 195 steps and exited 0 in 3 of 6 runs; the other 3 died after step 101 inside the pmapped train step, with a segmentation fault or `JaxRuntimeError: INTERNAL: CUDA error: ... CUDA_ERROR_LAUNCH_FAILED` (kernel log: NVRM Xid 13), cause not found. With `--config.log_eval_steps=1000` added, 7 of 8 runs exited 0. Same environment additions as `baselines/scenic`. |
