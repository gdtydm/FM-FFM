import tensorflow as tf
from util import FieldHandler, transformation_data, dataGenerate
from model_fn import create_model_fn
from input_fn import create_train_input_fn
from hparams import create_hparams, FLAGS
from model.FM import FM
from model.FFM import FFM

tf.logging.set_verbosity(FLAGS.logging_level)


def main(_):
    fh = FieldHandler(train_file_path=FLAGS.train_file_path,
                      category_columns=FLAGS.category_columns,
                      continuation_columns=FLAGS.continuation_columns)

    features, labels = transformation_data(file_path=FLAGS.train_file_path, field_hander=fh, label=FLAGS.label)

    # features, labels, files_dict = dataGenerate(FLAGS.train_file_path)
    hparams = create_hparams(fh.field_nums, fh.feature_nums)

    train_input_fn = create_train_input_fn(features,
                                           label=labels,
                                           batch_size=hparams.batch_size,
                                           num_epochs=hparams.epoches)

    
    if hparams.model == "fm":
        model_fn = create_model_fn(FM)
    elif hparams.model == "ffm":
        if hparams.use_deep:
            tf.logging.warning("\n\n>>>>>>>>>>> use ffm model, ignore --use_deep params. <<<<<<<<<<<<<<<\n")
        model_fn = create_model_fn(FFM)
    else:
        raise ValueError("model is ffm or fm.")
    
    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        model_dir=FLAGS.model_path,
        params=hparams,
        config=tf.estimator.RunConfig(
            tf_random_seed=hparams.seed,
            log_step_count_steps=500
        )
    )

    show_dict = {
        "loss":"loss",
        "accuracy":"accuracy/value",
        "auc":"auc/value"
    }
   
    log_hook = tf.train.LoggingTensorHook(show_dict, every_n_iter=100)
    # estimator.train(input_fn=train_input_fn, hooks=[log_hook])

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, hooks=[log_hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=train_input_fn, )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    
if __name__ == "__main__":
    tf.app.run()
