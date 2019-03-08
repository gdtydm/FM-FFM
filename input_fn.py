import tensorflow as tf

def create_train_input_fn(features, label, batch_size=32, num_epochs=10):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((features, label))
        dataset = dataset.shuffle(1000).repeat(num_epochs).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()
    return input_fn