# Baseline status

Each baseline was checked in its own Python 3.12 environment next to the
newest JAX. The rungs are:

1. installs next to current JAX;
2. every module imports (`python tools/import_check.py <directory>`);
3. its unit tests pass;
4. a few training steps run on small or synthetic data.

Only files under `jaxpruner/` were changed to get here. Nothing was run on
ImageNet, C4 or Atari at full scale.

## Environments

Each environment is a git-ignored directory `.venv-*` in the repository root,
built with [uv](https://docs.astral.sh/uv/) and Python 3.12. The commands below
are the ones that were run, in this order, from the repository root; every
command after the first install adds a package that the first install does not
bring in. Dependencies are not pinned, and scenic and t5x are installed from
the head of their repositories, so a rebuild on another day can resolve to
other versions. The versions each framework environment had are in the table
under [Results](#results).

### Notebook environment (`.venv-nb`)

```bash
uv venv --python 3.12 .venv-nb
VIRTUAL_ENV=.venv-nb uv pip install -e ".[test,notebooks]" "jax[cuda12]" tensorflow tensorflow-datasets
VIRTUAL_ENV=.venv-nb uv pip install importlib_resources
```

tensorflow-datasets 4.9.10 uses `importlib_resources` without installing it;
`colabs/mnist_pruning.ipynb` needs it to load MNIST.

Notebooks executed in it: `colabs/quick_start.ipynb`, `colabs/deep_dive.ipynb`,
`colabs/mnist_pruning.ipynb` and
`jaxpruner/projects/bigsparse/sparsity-scaling.ipynb`.

### scenic (`.venv-scenic`)

```bash
uv venv --python 3.12 .venv-scenic
VIRTUAL_ENV=.venv-scenic uv pip install -e ".[test,notebooks]" "jax[cuda12]" "scenic @ git+https://github.com/google-research/scenic"
VIRTUAL_ENV=.venv-scenic uv pip install tensorboard
VIRTUAL_ENV=.venv-scenic uv pip install importlib_resources
```

scenic resolved to commit `8c113c5` and optax to 0.2.9.dev0 from git.
`tensorboard` is needed by `clu.metric_writers` and `importlib_resources` by
tensorflow-datasets.

Notebook executed in it: `colabs/sparse_model_vit.ipynb`.

### fedjax (`.venv-fedjax`)

```bash
uv venv --python 3.12 .venv-fedjax
VIRTUAL_ENV=.venv-fedjax uv pip install -e ".[test,notebooks]" "jax[cuda12]" fedjax
VIRTUAL_ENV=.venv-fedjax uv pip install tensorflow
VIRTUAL_ENV=.venv-fedjax uv pip install tensorboard
```

The trainer imports `tensorflow.io.gfile`, and the logger of fedjax calls
`tf.summary.scalar`, which needs `tensorboard`. For a time the environment held
fedjax from the head of its repository; it was put back to the PyPI release
with `VIRTUAL_ENV=.venv-fedjax uv pip install --reinstall-package fedjax "fedjax==0.0.17"`,
the release that the second command above installs.

Notebook executed in it: `colabs/fedjax_jaxpruner.ipynb`.

### t5x (`.venv-t5x`)

```bash
uv venv --python 3.12 .venv-t5x
VIRTUAL_ENV=.venv-t5x uv pip install -e ".[test]" "jax[cuda12]" "t5x @ git+https://github.com/google-research/t5x"
VIRTUAL_ENV=.venv-t5x uv pip install tensorboard
VIRTUAL_ENV=.venv-t5x uv pip install --no-deps t5
VIRTUAL_ENV=.venv-t5x uv pip install babel editdistance immutabledict "mesh-tensorflow[transformer]" nltk pandas rouge-score sacrebleu scikit-learn scipy sentencepiece six transformers
```

t5x resolved to commit `2045b33`, with flax, optax and seqio from git. t5x
imports `tensorboard` without declaring it. The gin files import
`t5.data.mixtures`; `t5` is installed without its dependencies, followed by
those dependencies by name, because a plain `uv pip install t5` would add
`seqio-nightly` on top of the `seqio` that t5x installs.

No notebook was executed in it.

### dopamine (`.venv-dopamine`)

```bash
uv venv --python 3.12 .venv-dopamine
VIRTUAL_ENV=.venv-dopamine uv pip install -e ".[test]" "jax[cuda12]" dopamine-rl
VIRTUAL_ENV=.venv-dopamine uv pip install --no-deps baselines==0.1.5
VIRTUAL_ENV=.venv-dopamine uv pip uninstall tensorflow-probability
VIRTUAL_ENV=.venv-dopamine uv pip install tfp-nightly tensorboard
```

dopamine-rl 4.1.2 imports `baselines` without declaring it, and a plain install
of `baselines` fails to build its pinned `gym==0.21.0`, hence `--no-deps`. The
released tensorflow-probability 0.25.0 does not import with jax 0.11.2; it is
uninstalled first because it and `tfp-nightly` own the same package directory.
The three further installs were added to the existing environment; the
sequence was not repeated from an empty one.

No notebook was executed in it.

## Results

| Baseline | Rung reached | Versions | Command for the highest rung | What stops the next rung |
|---|---|---|---|---|
| `baselines/scenic` | 4 | jax 0.11.2, flax 0.12.10, tensorflow 2.21.0 | `TFDS_DATA_DIR=/tmp/jaxpruner-smoke/tfds-scenic .venv-scenic/bin/python -m jaxpruner.baselines.scenic.main --config=jaxpruner/baselines/scenic/configs/imagenet_resnet_config.py:runlocal --config.dataset_name=cifar10 --config.batch_size=256 --config.num_training_epochs=1 --config.sparsity_config.algorithm=magnitude --config.sparsity_config.update_freq=10 --config.sparsity_config.update_start_step=50 --config.sparsity_config.update_end_step=150 --workdir=/tmp/jaxpruner-smoke/scenic` | — (195 steps, exit 0 in 4 of 4 runs). The environment also needs `tensorboard` and `importlib_resources`, which the install does not pull in. Only the ResNet config was run: `model_name=plainvit` cannot be imported, because the scenic package that pip builds leaves out `scenic/projects/baselines/plainvit`, so `main.py` now imports it only when that model is asked for. |
| `projects/activation_sparsity` | 4, not reliably | jax 0.11.2, flax 0.12.10, tensorflow 2.21.0 | `TFDS_DATA_DIR=/tmp/jaxpruner-smoke/tfds-scenic .venv-scenic/bin/python -m jaxpruner.projects.activation_sparsity.scenic.main --config=jaxpruner/projects/activation_sparsity/scenic/configs/imagenet_resnet_config.py:runlocal --config.dataset_name=cifar10 --config.batch_size=256 --config.num_training_epochs=1 --workdir=/tmp/jaxpruner-smoke/activation_sparsity` | The command ran 195 steps and exited 0 in 3 of 6 runs; the other 3 died after step 101 inside the pmapped train step, with a segmentation fault or `JaxRuntimeError: INTERNAL: CUDA error: ... CUDA_ERROR_LAUNCH_FAILED` (kernel log: NVRM Xid 13), cause not found. With `--config.log_eval_steps=1000` added, 7 of 8 runs exited 0. Same environment additions as `baselines/scenic`. |
| `baselines/fedjax` | 4 (rung 3 not applicable: the baseline has no unit tests) | Python 3.12.13, jax 0.11.2, optax 0.2.8, fedjax 0.0.17 (the PyPI release), tensorflow 2.21.0, tensorboard 2.21.0. tensorflow and tensorboard are not dependencies of fedjax or of this package and were installed by hand: `uv pip install tensorflow tensorboard`. fedjax 0.0.17 still calls `jax.tree_map`, which current JAX no longer has, in `fedjax.tree_util.tree_mean`; the trainer and the notebook `colabs/fedjax_jaxpruner.ipynb` compute that weighted mean themselves and do not call it. The notebook was executed in the same environment. | `.venv-fedjax/bin/python -m jaxpruner.baselines.fedjax.trainer --algorithm=magnitude --sparsity=0.8 --update_start_step=2 --num_rounds=20 --workdir=/tmp/jaxpruner-smoke/fedjax` (exit 0 after 20 rounds; downloads the EMNIST sqlite files, 155 MB, to `~/.cache/fedjax` on the first run) | — |
| `baselines/t5x` | 4 (20 training steps on a synthetic seqio task, `tiny_c4_pretrain.gin` with `sparse_pretrain.gin`: exit 0, `_total_sparsity=0.797482` at step 20 against a 0.8 target). Rung 2: 3 of 3 modules import and all 6 gin files parse. Rung 3 is not applicable: there are no unit tests. | Python 3.12.13, jax 0.11.2, jaxlib 0.11.2, flax 0.12.10 (git), optax 0.2.9.dev0 (git), chex 0.1.92, tensorflow 2.21.0, seqio 0.0.20 (git), t5x 0.0.0 (git 2045b33), t5 0.9.4, gin-config 0.5.0. Environment: `uv venv --python 3.12 .venv-t5x`; `VIRTUAL_ENV=.venv-t5x uv pip install -e ".[test]" "jax[cuda12]" "t5x @ git+https://github.com/google-research/t5x"`; then `VIRTUAL_ENV=.venv-t5x uv pip install tensorboard` (t5x imports it without declaring it); then, for the gin files, which import `t5.data.mixtures`: `VIRTUAL_ENV=.venv-t5x uv pip install --no-deps t5` and `VIRTUAL_ENV=.venv-t5x uv pip install babel editdistance immutabledict "mesh-tensorflow[transformer]" nltk pandas rouge-score sacrebleu scikit-learn scipy sentencepiece six transformers` (a plain `uv pip install t5` would add `seqio-nightly` on top of the `seqio` that t5x installs). | `.venv-t5x/bin/python -c "import runpy, seqio, tensorflow as tf; f = seqio.Feature(seqio.SentencePieceVocabulary('gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model')); s = tf.strings.as_string; seqio.TaskRegistry.add('synthetic_text', source=seqio.FunctionDataSource(lambda split, shuffle_files=False, seed=None: tf.data.Dataset.range(512).map(lambda i: {'inputs': 'copy the number ' + s(i) + ' and then ' + s(i), 'targets': 'the number ' + s(i) + ' ' + s(i)}), splits=['train', 'validation']), preprocessors=[seqio.preprocessors.tokenize, seqio.preprocessors.append_eos], output_features={'inputs': f, 'targets': f}, metric_fns=[]); runpy.run_module('t5x.train', run_name='__main__', alter_sys=True)" --gin.MODEL_DIR="'/tmp/jaxpruner-smoke/t5x-fix1/baseline2'" --gin.MIXTURE_OR_TASK_NAME="'synthetic_text'" --gin.USE_CACHED_TASKS=False --gin.TRAIN_STEPS=20 --gin.BATCH_SIZE=8 --gin.train.eval_period=10 --gin.train.eval_steps=2 --gin_file=jaxpruner/baselines/t5x/examples/tiny_c4_pretrain.gin --gin_file=jaxpruner/baselines/t5x/sparse_pretrain.gin "--gin.TASK_FEATURE_LENGTHS={'inputs':32,'targets':32}" --gin.create_updater_from_config.update_start_step=2 --gin.create_updater_from_config.update_end_step=12 --gin.create_updater_from_config.update_freq=2 --gin.warmup_cosine_decay_schedule.warmup_steps=5` (run from the repository root. The `-c` part registers a synthetic seqio task of 512 short strings and then runs `t5x.train` as `__main__`; the vocabulary is the public one the gin files bind, read from `gs://`, about 0.8 MB, so the run needs network access. The flags shorten the run: 20 steps, batch 8, 32 tokens, pruning over steps 2 to 12, and 5 warmup steps because the 1000 in the gin file exceed the 20 steps.) Rung 2: `JAX_PLATFORMS=cpu .venv-t5x/bin/python tools/import_check.py jaxpruner/baselines/t5x` and, for the gin files, `JAX_PLATFORMS=cpu .venv-t5x/bin/python -c "import glob, os, sys, gin, t5x; from t5x import train; sys.modules['__main__'] = train; gin.add_config_file_search_path(os.path.dirname(os.path.dirname(t5x.__file__))); gin.add_config_file_search_path('.'); [(gin.clear_config(), gin.parse_config_file(p, skip_unknown=False), print('parsed', p)) for p in sorted(glob.glob('jaxpruner/baselines/t5x/**/*.gin', recursive=True))]"` (the gin files bind `__main__.train`, so the check puts `t5x.train` in the place of `__main__`, as `python -m t5x.train` does) | Rung 4 is the top of the ladder. Not checked: training on C4 (`c4_v220_span_corruption`), which is hundreds of gigabytes and not on this machine, so nothing here says the baseline reaches its published quality. In the 20 synthetic steps the training loss did not fall (in one run 1638.66 at step 10 and 1659.29 at step 20, accuracy 0; the loss differs from run to run because the data order is not seeded, the sparsity does not), with a peak learning rate of 1e-4 reached after 5 steps and 80% of the weights pruned by step 12. |
| `projects/bigsparse/t5x` | 4 (20 training steps on the same synthetic task, exit 0 for each of: `small_c4_pretrain.gin` with `sparse_pretrain.gin`, `_total_sparsity=0.797239` at step 20; with `sparse_pretrain_mag.gin`, 0.59043; with `sparse_pretrain_acdc.gin`, 0.48391; `decoder_base_c4_pretrain.gin`, shrunk, with `sparse_pretrain.gin`, 0.797539). Rung 2: 4 of 4 modules import and 8 of 8 gin files parse. Rung 3 is not applicable: there are no unit tests. | Same environment as `baselines/t5x`. | `.venv-t5x/bin/python -c "import runpy, seqio, tensorflow as tf; f = seqio.Feature(seqio.SentencePieceVocabulary('gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model')); s = tf.strings.as_string; seqio.TaskRegistry.add('synthetic_text', source=seqio.FunctionDataSource(lambda split, shuffle_files=False, seed=None: tf.data.Dataset.range(512).map(lambda i: {'inputs': 'copy the number ' + s(i) + ' and then ' + s(i), 'targets': 'the number ' + s(i) + ' ' + s(i)}), splits=['train', 'validation']), preprocessors=[seqio.preprocessors.tokenize, seqio.preprocessors.append_eos], output_features={'inputs': f, 'targets': f}, metric_fns=[]); runpy.run_module('t5x.train', run_name='__main__', alter_sys=True)" --gin.MODEL_DIR="'/tmp/jaxpruner-smoke/t5x-fix1/big'" --gin.MIXTURE_OR_TASK_NAME="'synthetic_text'" --gin.USE_CACHED_TASKS=False --gin.TRAIN_STEPS=20 --gin.BATCH_SIZE=8 --gin.train.eval_period=10 --gin.train.eval_steps=2 --gin_file=jaxpruner/projects/bigsparse/t5x/configs/small_c4_pretrain.gin --gin_file=jaxpruner/projects/bigsparse/t5x/configs/sparse_pretrain.gin "--gin.TASK_FEATURE_LENGTHS={'inputs':32,'targets':32}" --gin.create_updater_from_config.update_start_step=2 --gin.create_updater_from_config.update_end_step=12 --gin.create_updater_from_config.update_freq=2 --gin.warmup_cosine_decay_schedule.warmup_steps=5` (see the `baselines/t5x` row for what the parts do). The other three runs use the same command up to `--gin.train.eval_steps=2`, with their own `MODEL_DIR`, followed by: for `sparse_pretrain_mag.gin`, `--gin_file=jaxpruner/projects/bigsparse/t5x/configs/small_c4_pretrain.gin --gin_file=jaxpruner/projects/bigsparse/t5x/configs/sparse_pretrain_mag.gin "--gin.TASK_FEATURE_LENGTHS={'inputs':32,'targets':32}" --gin.create_updater_from_config.update_start_step=2 --gin.create_updater_from_config.update_end_step=12 --gin.create_updater_from_config.update_freq=2 --gin.create_updater_from_config.sparsity=0.8` (the file ships with sparsity 0); for `sparse_pretrain_acdc.gin`, `--gin_file=jaxpruner/projects/bigsparse/t5x/configs/small_c4_pretrain.gin --gin_file=jaxpruner/projects/bigsparse/t5x/configs/sparse_pretrain_acdc.gin "--gin.TASK_FEATURE_LENGTHS={'inputs':32,'targets':32}" --gin.create_updater_from_config.sparsity=0.8 --gin.create_updater_from_config.init_dense_steps_end=4 --gin.create_updater_from_config.final_sparse_steps_start=16 --gin.create_updater_from_config.cycle_sparse_steps=4 --gin.create_updater_from_config.cycle_dense_steps=4`; for the decoder-only model, `--gin_file=jaxpruner/projects/bigsparse/t5x/configs/decoder_base_c4_pretrain.gin --gin_file=jaxpruner/projects/bigsparse/t5x/configs/sparse_pretrain.gin "--gin.TASK_FEATURE_LENGTHS={'targets':32}" --gin.create_updater_from_config.update_start_step=2 --gin.create_updater_from_config.update_end_step=12 --gin.create_updater_from_config.update_freq=2 --gin.warmup_cosine_decay_schedule.warmup_steps=5 --gin.TransformerConfig.emb_dim=64 --gin.TransformerConfig.num_heads=2 --gin.TransformerConfig.num_layers=2 --gin.TransformerConfig.head_dim=32 --gin.TransformerConfig.mlp_dim=128` (the last five flags shrink the base model to 2 layers). Rung 2: `JAX_PLATFORMS=cpu .venv-t5x/bin/python tools/import_check.py jaxpruner/projects/bigsparse/t5x` and, for the gin files, `for p in jaxpruner/projects/bigsparse/t5x/configs/*.gin; do JAX_PLATFORMS=cpu .venv-t5x/bin/python -c "import os, sys, gin, t5x; from t5x import train; sys.modules['__main__'] = train; gin.add_config_file_search_path(os.path.dirname(os.path.dirname(t5x.__file__))); gin.add_config_file_search_path('.'); gin.parse_config_file(sys.argv[1], skip_unknown=False); print('parsed', sys.argv[1])" $p 2>/dev/null; done` (one process per file: the decoder file and the encoder-decoder files both import a module as `jp_network`, and gin refuses the second one in the same process) | Rung 4 is the top of the ladder. Not checked: training on C4, as above. Training loss at steps 10 and 20, in one run each (it differs from run to run, as above): 1607.46 and 1487.16 (`sparse_pretrain.gin`), 1332.89 and 967.24 (`sparse_pretrain_mag.gin`), 1395.68 and 1058.75 (`sparse_pretrain_acdc.gin`), 2187.70 and 2091.62 (decoder). The total sparsity of 0.59043 for `sparse_pretrain_mag.gin` at a 0.8 setting comes with `embed_sparsity = 'auto'`, which gives the embedding tables a lower sparsity; whether that total is the intended one was not investigated. `configs/decoder_base_c4_pretrain.gin` was changed to bind `TransformerConfig` from `jaxpruner/projects/bigsparse/t5x/decoder_only_network.py` instead of `t5x.examples.decoder_only.network`: the t5x repository has that file but no `__init__.py` next to it, so an installed t5x leaves the module out (`ModuleNotFoundError: No module named 't5x.examples.decoder_only.network'`). The binding on `network.DecoderWrapper`, which nothing in the file used, went with it. |
| `baselines/dopamine` | 4 (rung 3 not applicable: no unit tests) | Python 3.12.13, jax 0.11.2, dopamine-rl 4.1.2, tensorflow 2.21.0, tfp-nightly 0.26.0.dev20261012 (in place of tensorflow-probability 0.25.0), baselines 0.1.5 (installed without its dependencies), tensorboard 2.21.0, flax 0.12.10, optax 0.2.8 | Environment: `uv venv --python 3.12 .venv-dopamine`, `VIRTUAL_ENV=.venv-dopamine uv pip install -e ".[test]" "jax[cuda12]" dopamine-rl`, then three further installs that the first one does not bring in: `VIRTUAL_ENV=.venv-dopamine uv pip install --no-deps baselines==0.1.5`, `VIRTUAL_ENV=.venv-dopamine uv pip uninstall tensorflow-probability`, `VIRTUAL_ENV=.venv-dopamine uv pip install tfp-nightly tensorboard`. Rung 2: `JAX_PLATFORMS=cpu .venv-dopamine/bin/python tools/import_check.py jaxpruner/baselines/dopamine` (three `OK`). Rung 4: `.venv-dopamine/bin/python -m jaxpruner.baselines.dopamine.train --base_dir=/tmp/jaxpruner-smoke/dopamine --gin_files=jaxpruner/baselines/dopamine/configs/dqn.gin --gin_bindings="atari_lib.create_atari_environment.game_name='Pong'" --gin_bindings="Runner.num_iterations=1" --gin_bindings="Runner.training_steps=500" --gin_bindings="Runner.evaluation_steps=100" --gin_bindings="Runner.max_steps_per_episode=200" --gin_bindings="JaxDQNAgent.min_replay_history=100"` (exit 0, one iteration logged) | — (rung 4 is the last). Rungs 2 and 4 depend on the three further installs; with only the first install the row stops at rung 1. Without `baselines`, `dqn_agent_pruner` and `train` fail at `dopamine/discrete_domains/atari_lib.py` line 45 with `ModuleNotFoundError: No module named 'baselines'`; a plain `uv pip install baselines` fails to build its pinned `gym==0.21.0`, hence `--no-deps`. With the released tensorflow-probability 0.25.0 they fail at `tensorflow_probability/python/internal/backend/jax/ops.py` line 681 with `AttributeError: module 'jax.interpreters.xla' has no attribute 'pytype_aval_mappings'`; the nightly pre-release imports with jax 0.11.2. Without `tensorboard`, `train` fails with `ModuleNotFoundError: No module named 'tensorboard'` (from `dopamine/jax/agents/ppo/ppo_agent.py` line 36 through `flax/metrics/tensorboard.py`). dopamine itself was not patched. `configs/dqn.gin` was updated for dopamine 4: the replay buffer is now `dopamine.jax.replay_memory.replay_buffer.ReplayBuffer` with `max_capacity` and `batch_size`. The run is 500 training steps of Pong; pruning starts at step 500000 in the config, so no weights were pruned in it. |
