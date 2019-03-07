import tensorflow as tf

def create_train_input_fn(features, label, batch_size=32, num_epochs=10, shuffle=True, queue_capacity=10000, num_threads=1):
    return tf.estimator.inputs.numpy_input_fn(x=features,
                                       y=label,
                                       batch_size=batch_size,
                                       num_epochs=num_epochs,
                                       shuffle=shuffle,
                                       queue_capacity=queue_capacity,
                                       num_threads=num_threads
                                       )
