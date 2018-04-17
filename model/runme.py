import tensorflow as tf
import numpy as np
from model.graphsage import supervised_graphsage, MeanAggregator


def dummy_gen(batch_size, num_labels, num_samples, num_feats):

    def rar(m, n):
        return np.array(np.random.rand(m, n), dtype=np.float32)

    for i in range(100):
        yield (batch_size,
               rar(batch_size, num_labels),
               rar(batch_size, num_feats),
               rar(batch_size * num_samples[1], num_feats),
               rar(batch_size * num_samples[0] * num_samples[1], num_feats)
               )


def main():
    nb, nl, ns, nf = 10, 10, [25, 10], 50

    ds = tf.data.Dataset.from_generator(
        lambda: dummy_gen(nb, nl, ns, nf),
        (tf.int32, tf.float32, tf.float32, tf.float32, tf.float32),
        (
            tf.TensorShape(()),
            tf.TensorShape((None, nl)),
            tf.TensorShape((None, nf)),
            tf.TensorShape((None, nf)),
            tf.TensorShape((None, nf))
        )
    ).prefetch(1)
    t_batch_iter = ds.make_initializable_iterator()
    batch_in = t_batch_iter.get_next()
    loss, opt_op, preds = supervised_graphsage(
        num_labels=nl,
        dims=[nf, 120, 120],
        num_samples=ns,
        batch_in=batch_in,
        agg=MeanAggregator
    )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(10):
            sess.run(t_batch_iter.initializer)
            while True:
                try:
                    outs = sess.run([loss, opt_op, preds])
                    print("loss={:.5f}".format(outs[0]))
                except tf.errors.OutOfRangeError:
                    break

    print("Done")




if __name__ == '__main__':
    main()

