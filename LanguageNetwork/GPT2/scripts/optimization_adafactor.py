# Original work Copyright 2018 The Google AI Language Team Authors.
# Modified work Copyright 2019 Rowan Zellers
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
import re
import tensorflow as tf
from utils import get_shape_list


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = AdaFactorOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    if use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)

    # You could do this, but instead we don't because a) it's slow and b) we already did the 'update clipping'
    # (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdaFactorOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])

    train_metrics = {
        'learning_rate': learning_rate,
        'minibatch_loss': loss,
        # 'minibatch_ppl': tf.math.exp(loss),
    }
    return train_op, train_metrics


class AdaFactorOptimizer(tf.compat.v1.train.Optimizer):
    """here's the optimizer we'll use"""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 clipping_rate=1.0,
                 name="AdaFactorOptimizer"):
        """Constructs a AdaFactorOptimizer."""
        super(AdaFactorOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.epsilon1 = 1e-30
        self.epsilon2 = 0.001
        self.clipping_rate = clipping_rate
        self.exclude_from_weight_decay = exclude_from_weight_decay
        self.use_locking = False

    def _use_factored(self, shape):
        return len(shape) >= 2

    def _parameter_scale(self, var):
        """Estimate the scale of the parameters from the current values.
        We include a minimum value of 0.001 to give it a chance to escape 0
        if it was zero-initialized.
        Instead of using the value, we could impute the scale from the shape,
        as initializers do.
        Args:
          var: a variable or Tensor.
        Returns:
          a Scalar
        """
        return tf.maximum(reduce_rms(var), self.epsilon2)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)
            shape_list = get_shape_list(param, expected_rank=[1, 2])

            # decay_rate = 1 - tf.pow(tf.cast(tf.train.get_or_create_global_step(), tf.float32) + 1.0, -0.8)
            decay_rate = self.beta_2
            grad_squared = tf.square(grad) + self.epsilon1

            update_scale = self.learning_rate
            # update_scale = self.learning_rate * tf.cast(self._parameter_scale(param), dtype=tf.float32)

            # HACK: Make things dependent on grad.
            # This confounds the XLA rewriter and keeps it from fusing computations
            # across different variables.  This fusion is a bad for HBM usage, since
            # it causes the gradients to persist in memory.
            grad_squared_mean = tf.reduce_mean(grad_squared)
            decay_rate += grad_squared_mean * 1e-30
            update_scale += grad_squared_mean * 1e-30

            # END HACK

            if self._use_factored(shape_list):
                num_rows, num_columns = shape_list

                vr = tf.get_variable(
                    name=param_name + "/adafactor_vr",
                    shape=[num_rows],
                    dtype=tf.float32,
                    trainable=False,
                    initializer=tf.zeros_initializer())
                vc = tf.get_variable(
                    name=param_name + "/adafactor_vc",
                    shape=[num_columns],
                    dtype=tf.float32,
                    trainable=False,
                    initializer=tf.zeros_initializer())

                next_vr = decay_rate * vr + (1 - decay_rate) * tf.reduce_mean(grad_squared, 1)
                next_vc = decay_rate * vc + (1 - decay_rate) * tf.reduce_mean(grad_squared, 0)

                long_term_mean = tf.reduce_mean(next_vr, -1, keepdims=True)
                r_factor = tf.rsqrt(next_vr / long_term_mean + self.epsilon1)
                c_factor = tf.rsqrt(next_vc + self.epsilon1)
                update = grad * tf.expand_dims(r_factor, -1) * tf.expand_dims(c_factor, -2)

                assignments.append(vr.assign(next_vr, use_locking=self.use_locking))
                assignments.append(vc.assign(next_vc, use_locking=self.use_locking))
            else:
                v = tf.get_variable(
                    name=param_name + "/adafactor_v",
                    shape=shape_list,
                    dtype=tf.float32,
                    trainable=False,
                    initializer=tf.zeros_initializer())
                next_v = decay_rate * v + (1 - decay_rate) * grad_squared

                assignments.append(v.assign(next_v, use_locking=self.use_locking))
                update = grad * tf.rsqrt(next_v + self.epsilon1)

            clipping_denom = tf.maximum(1.0, reduce_rms(update) / self.clipping_rate)
            update /= clipping_denom

            # Do weight decay
            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = update_scale * update
            next_param = param - update_with_lr

            assignments.append(param.assign(next_param, use_locking=self.use_locking))
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


def reduce_rms(x):
    return tf.sqrt(tf.reduce_mean(tf.square(x)))
