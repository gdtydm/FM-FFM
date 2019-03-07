import tensorflow as tf
from util import FieldHandler, transformation_data
from model_fn import create_model_fn
from input_fn import create_train_input_fn
from hparams import create_hparams, FLAGS
from model.FM import FM
tf.logging.set_verbosity(FLAGS.logging_level)


def main(_):
    fh = FieldHandler(train_file_path=FLAGS.train_file_path,
                      category_columns=FLAGS.category_columns,
                      continuation_columns=FLAGS.continuation_columns)
    df_i, df_v, label = transformation_data(FLAGS.train_file_path, fh, label=FLAGS.label)
    hparams = create_hparams(df_i.shape[1], fh.field_nums)

    train_input_fn = create_train_input_fn({"df_i": df_i, "df_v": df_v},
                                           label=label,
                                           batch_size=hparams.batch_size,
                                           num_epochs=hparams.epoches,
                                           shuffle=True)
    
    if hparams.model == "fm":
        model_fn = create_model_fn(FM)
    elif hparams.model == "ffm":
        pass
    else:
        raise ValueError("model is ffm or fm.")
    
    print(">>>>>>>>>>>",model_fn)
    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        model_dir=FLAGS.model_path,
        config=tf.estimator.RunConfig(),
        params=hparams
    )
    if hparams.loss_type == "log_loss":
        show_dict = {
            "loss":"loss",
            "accuracy":"accuracy/value",
            "auc":"auc/value"
        }
    elif hparams.loss_type == "mse":
        show_dict = {
            "loss":"loss"
        }
    else:
        show_dict = {}
    log_hook = tf.train.LoggingTensorHook(show_dict, every_n_iter=100)
    
    estimator.train(input_fn=train_input_fn, hooks=[log_hook])


if __name__ == "__main__":
    tf.app.run()


