import tensorflow as tf
from keras.layers import Dense
from keras import backend as K
from keras.engine.topology import Layer
from typing import List, Tuple
from util.initializer import glorot_initializer


class MeanAggregator(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MeanAggregator, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.w_neigh = self.add_weight(
            name='w_neigh',
            shape=(input_shape[1][2], self.output_dim),
            initializer=glorot_initializer((input_shape[1][2], self.output_dim)),
            trainable=True
        )
        self.w_self = self.add_weight(
            name='w_self',
            shape=(input_shape[0][1], self.output_dim),
            initializer=glorot_initializer((input_shape[0][1], self.output_dim)),
            trainable=True
        )
        super(MeanAggregator, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, **kwargs):
        neigh_means = K.mean(x[1], axis=1)

        from_self = K.dot(x[0], self.w_self)
        from_neigh = K.dot(neigh_means, self.w_neigh)
        return K.concatenate([from_self, from_neigh], axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 2*self.output_dim


def supervised_graphsage(
        num_labels: int,
        dims: List[int],
        num_samples: List[int],
        batch_in: Tuple,
        agg
):
    batch_size, labels, *x = batch_in
    batch_size = tf.Print(batch_size, [batch_size])
    assert len(x) == len(num_samples) + 1 and len(x) == len(dims)
    nl = len(num_samples)
    num_samples += [1]
    output_dims = dims[1:]
    input_dims = dims[0:1] + [2*d for d in dims[1:-1]]

    # compose graphsage layers
    for layer in range(nl):
        agg_f = agg(output_dim=output_dims[layer])
        x = [
            agg_f([x[i], tf.reshape(x[i+1], [batch_size*num_samples[-i-1], num_samples[-i-2], input_dims[layer]])])
            for i in range(nl - layer)
        ]

    # outputs
    outs = tf.nn.l2_normalize(x[0], 1)
    preds = Dense(num_labels)(outs)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    grads_and_vars = optimizer.compute_gradients(loss)
    clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                              for grad, var in grads_and_vars]
    opt_op = optimizer.apply_gradients(clipped_grads_and_vars)
    y_preds = tf.nn.sigmoid(preds)
    y_true = labels

    return loss, opt_op, y_preds, y_true


