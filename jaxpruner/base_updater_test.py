# coding=utf-8
# Copyright 2024 Jaxpruner Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Sparsity updater classes in base_updater.py."""
import dataclasses
import functools
from typing import NamedTuple, Optional
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import optax
from jaxpruner import base_updater
from jaxpruner import sparsity_distributions
from jaxpruner import sparsity_schedules
from jaxpruner import sparsity_types


class BaseUpdaterTest(parameterized.TestCase, absltest.TestCase):

  def setUp(self):
    super(BaseUpdaterTest, self).setUp()
    self.updater = base_updater.BaseUpdater(
        scheduler=sparsity_schedules.PolynomialSchedule(
            update_freq=2, update_start_step=4, update_end_step=9
        ),
        sparsity_distribution_fn=functools.partial(
            sparsity_distributions.uniform, sparsity=0.8
        ),
    )

  def testUpdaterInitialized(self):
    self.assertIsInstance(
        self.updater.sparsity_type, sparsity_types.Unstructured
    )
    self.assertTrue(callable(self.updater.topk_fn))

  def testOptaxWrapper(self):
    distribution_fn = mock.MagicMock()
    sparse_updater = base_updater.BaseUpdater(
        scheduler=sparsity_schedules.PolynomialSchedule(
            update_freq=1, update_start_step=0, update_end_step=1
        ),
        sparsity_distribution_fn=distribution_fn,
    )
    sparse_updater.get_initial_masks = mock.MagicMock()
    sparse_updater.maybe_update_mask = mock.MagicMock()
    sparse_updater.maybe_update_mask.return_value = ({}, {})

    inner = mock.MagicMock()
    inner.update.return_value = ({}, None)

    tx = sparse_updater.wrap_optax(inner)

    _ = tx.init({})
    distribution_fn.assert_called_once()
    sparse_updater.get_initial_masks.assert_called_once()

    sparse_updater.calculate_scores = mock.MagicMock()
    sparse_updater.calculate_scores.return_value = None

    _ = tx.update({}, base_updater.SparseState(masks=None, count=0), {})
    inner.update.assert_called_once()
    sparse_updater.calculate_scores.assert_called_once()

    # Test passing an extra argument to update().
    inner = mock.MagicMock()
    inner.update.return_value = ({}, None)
    sparse_updater.calculate_scores.reset_mock()
    inner.update.assert_not_called()

    tx = sparse_updater.wrap_optax(inner)
    extra_arg = object()
    _ = tx.update(
        {}, base_updater.SparseState(masks=None, count=0), {}, foo=extra_arg
    )
    inner.update.assert_called_once()
    _, kwargs = inner.update.call_args
    self.assertEmpty(kwargs, 'Should not pass extra args to inner')
    sparse_updater.calculate_scores.assert_called_once()
    _, kwargs = sparse_updater.calculate_scores.call_args
    expected = {'foo': extra_arg}
    self.assertEqual(kwargs, {**kwargs, **expected})

  def testInstantSparsify(self):
    distribution_fn = mock.MagicMock()
    distribution_fn.return_value = {}
    sparse_updater = base_updater.BaseUpdater(
        scheduler=sparsity_schedules.PolynomialSchedule(
            update_freq=1, update_start_step=0, update_end_step=1
        ),
        sparsity_distribution_fn=distribution_fn,
    )
    sparse_updater.calculate_scores = mock.MagicMock()
    sparse_updater.calculate_scores.return_value = {}
    sparse_updater.create_masks = mock.MagicMock()
    masks_returned = {'a': jnp.array([1, 0])}
    sparse_updater.create_masks.return_value = masks_returned
    masked_vars, masks = sparse_updater.instant_sparsify(
        {'a': jnp.array([2.0, -3.0])}
    )
    chex.assert_trees_all_close(masks, masks_returned)
    chex.assert_trees_all_close(masked_vars, {'a': jnp.array([2.0, 0.0])})
    distribution_fn.assert_called_once()
    sparse_updater.calculate_scores.assert_called_once()

    # Test passing an extra argument to instant_sparsify().
    sparse_updater.calculate_scores.reset_mock()
    extra_arg = object()
    _ = sparse_updater.instant_sparsify(
        {'a': jnp.array([2.0, -3.0])}, foo=extra_arg
    )
    sparse_updater.calculate_scores.assert_called_once()
    _, kwargs = sparse_updater.calculate_scores.call_args
    expected = {'foo': extra_arg}
    self.assertEqual(kwargs, {**kwargs, **expected})

  def testNoPruning(self):
    updater = base_updater.NoPruning()
    param_tree = {'a': 0, 'b': {'c': 0, 'd': 0}}
    expected = {'a': None, 'b': {'c': None, 'd': None}}
    self.assertEqual(
        updater.get_initial_masks(param_tree, target_sparsities=None), expected
    )
    chex.assert_trees_all_close(
        updater.instant_sparsify(param_tree)[0], param_tree
    )

  def testCustomTopkFn(self):
    def _mock_topk_fn(t, _):
      return jnp.ones_like(t)

    sparse_updater = base_updater.BaseUpdater(
        topk_fn=_mock_topk_fn,
    )
    # Test if custom topk_fn is not overrided.
    self.assertEqual(sparse_updater.topk_fn, _mock_topk_fn)


class _TransientState(NamedTuple):
  """Inner state with a field that is only set between two updates."""

  count: chex.Array
  scratch: Optional[chex.ArrayTree] = None


def _transient_optimizer(grow=False):
  """An optimizer whose update adds `state.scratch` to the gradients.

  Args:
    grow: If False, `update` clears `scratch`, so the state it returns has a
      different pytree structure than the state it was given. If True, `update`
      sets `scratch` although it was not set before.

  Returns:
    An optax gradient transformation.
  """

  def init_fn(params):
    del params
    return _TransientState(count=jnp.zeros([], jnp.int32))

  def update_fn(updates, state, params=None):
    del params
    if grow:
      return updates, _TransientState(count=state.count + 1, scratch=updates)
    updates = jax.tree.map(lambda u, s: u + s, updates, state.scratch)
    return updates, _TransientState(count=state.count + 1)

  return optax.GradientTransformation(init_fn, update_fn)


@dataclasses.dataclass
class _MagnitudeUpdater(base_updater.BaseUpdater):

  def calculate_scores(self, params, sparse_state=None, grads=None):
    del sparse_state, grads
    return jax.tree.map(jnp.abs, params)


def _magnitude_updater(skip_gradients):
  return _MagnitudeUpdater(
      scheduler=sparsity_schedules.PeriodicSchedule(
          update_freq=2, update_start_step=2, update_end_step=4
      ),
      sparsity_distribution_fn=functools.partial(
          sparsity_distributions.uniform, sparsity=0.5
      ),
      skip_gradients=skip_gradients,
  )


class WrapOptaxInnerStateTest(parameterized.TestCase, absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.params = {'w': jnp.array([[1.0, -2.0], [3.0, -4.0]])}
    self.grads = {'w': jnp.ones((2, 2))}

  @parameterized.parameters(False, True)
  def testInnerStateStructureMayChangeDuringUpdate(self, skip_gradients):
    updater = _magnitude_updater(skip_gradients)
    tx = updater.wrap_optax(_transient_optimizer())
    update = jax.jit(tx.update)
    state = tx.init(self.params)
    num_inner_updates = 0
    for step in range(6):
      is_update_step = bool(updater.scheduler.is_mask_update_iter(step))
      inner_state = state.inner_state._replace(scratch=self.grads)
      state = state._replace(inner_state=inner_state)
      updates, state = update(self.grads, state, self.params)
      self.assertIsNone(state.inner_state.scratch)
      if skip_gradients and is_update_step:
        chex.assert_trees_all_close(updates, {'w': jnp.zeros((2, 2))})
      else:
        num_inner_updates += 1
        chex.assert_trees_all_close(updates, {'w': 2 * jnp.ones((2, 2))})
      self.assertEqual(int(state.inner_state.count), num_inner_updates)
    self.assertEqual(num_inner_updates, 4 if skip_gradients else 6)

  @parameterized.product(
      skip_gradients=[False, True],
      make_inner=[lambda: optax.sgd(0.1, momentum=0.9), lambda: optax.adam(0.1)],
  )
  def testOptaxOptimizersMatchReference(self, skip_gradients, make_inner):
    updater = _magnitude_updater(skip_gradients)
    inner = make_inner()
    tx = updater.wrap_optax(inner)
    state = tx.init(self.params)
    expected_inner_state = inner.init(self.params)
    for step in range(6):
      is_update_step = bool(updater.scheduler.is_mask_update_iter(step))
      if skip_gradients and is_update_step:
        expected_updates = {'w': jnp.zeros((2, 2))}
      else:
        expected_updates, expected_inner_state = inner.update(
            self.grads, expected_inner_state, self.params
        )
      updates, state = tx.update(self.grads, state, self.params)
      chex.assert_trees_all_close(updates, expected_updates)
      chex.assert_trees_all_close(state.inner_state, expected_inner_state)
      chex.assert_trees_all_equal_dtypes(state.inner_state, expected_inner_state)

  def testSkipGradientsRejectsInnerStateThatGainsLeaves(self):
    tx = _magnitude_updater(skip_gradients=True).wrap_optax(
        _transient_optimizer(grow=True)
    )
    state = tx.init(self.params)
    with self.assertRaisesRegex(ValueError, 'skip_gradients=True needs'):
      tx.update(self.grads, state, self.params)


if __name__ == '__main__':
  absltest.main()
