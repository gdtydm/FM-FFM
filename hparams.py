import tensorflow as tf
from collections import namedtuple



tf.flags.DEFINE_string("opt_type", "gd", "optimizer type (adm, gd, adagrad, momentum).")
tf.flags.DEFINE_string("train_file_path", "./Dataset/train.csv", "train file path.")
tf.flags.DEFINE_string("label", "Survived", "target column name.")
tf.flags.DEFINE_string("activation", "relu", "deep mid activation function(tanh, relu, tanh, sigmoid).")
tf.flags.DEFINE_float("threshold", 0.5, "bi-classification threshold." )
tf.flags.DEFINE_string("loss_type", "log_loss", "bi-classification is log_loss, regression is mse.")
tf.flags.DEFINE_string("model_path", "./checkpoint/", "save model path.")
tf.flags.DEFINE_bool("use_deep", False, "Whether to use deep or not.")
tf.flags.DEFINE_string("model", "fm", "fm or ffm.")
tf.flags.DEFINE_list("layers", [30,30], "deep mid layers.")
tf.flags.DEFINE_list("category_columns", ['Pclass',"Sex","SibSp","Parch","Embarked"], "category columns.")
tf.flags.DEFINE_list("continuation_columns", ['Fare'], "continuation columns.")
tf.flags.DEFINE_float("lr", 0.01, "learning rate.")
tf.flags.DEFINE_float("line_output_keep_dropout", 0.1, "line output keep dropout in deep schema.")
tf.flags.DEFINE_float("fm_output_keep_dropout", 0.1, "fm output keep dropout in deep schema.")
tf.flags.DEFINE_float("deep_output_keep_dropout", 0.1, "deep output keep dropout in deep schema.")
tf.flags.DEFINE_float("deep_input_keep_dropout", 0.1, "deep input keep dropout in deep schema.")
tf.flags.DEFINE_float("deep_mid_keep_dropout", 0.2, "deep mid keep dropout in deep schema.")
tf.flags.DEFINE_integer("embedding_size", 3, "field embedding size")
tf.flags.DEFINE_bool("use_batch_normal", True, "Whether to use batch normal or not.")
tf.flags.DEFINE_integer("batch_size", 64, "batch size.")
tf.flags.DEFINE_integer("epoches", 100, "epoches.")
tf.flags.DEFINE_integer("logging_level", 20, "tensorflow logging level.")


FLAGS = tf.flags.FLAGS



HParams = namedtuple(
  "HParams",
  [
    "opt_type",
    "threshold",
    "loss_type",
    "use_deep",
    "model",
    "layers",
    "lr",
    "fm_output_keep_dropout",
    "line_output_keep_dropout",
    "deep_input_keep_dropout",
    "deep_mid_keep_dropout",
    "deep_output_keep_dropout",
    "embedding_size",
    "use_batch_normal",
    "batch_size",
    "epoches",
    "columns",
    "field_nums",
    "activation"
  ])


def create_hparams(columns, field_nums):
  return HParams(
    model=FLAGS.model,
    opt_type=FLAGS.opt_type,
    threshold=FLAGS.threshold,
    loss_type=FLAGS.loss_type,
    use_deep=FLAGS.model,
    layers=FLAGS.layers,
    lr=FLAGS.lr,
    fm_output_keep_dropout=FLAGS.fm_output_keep_dropout,
    line_output_keep_dropout=FLAGS.line_output_keep_dropout,
    deep_input_keep_dropout=FLAGS.deep_input_keep_dropout,
    deep_output_keep_dropout=FLAGS.deep_output_keep_dropout,
    deep_mid_keep_dropout=FLAGS.deep_mid_keep_dropout,
    embedding_size=FLAGS.embedding_size,
    use_batch_normal=FLAGS.use_batch_normal,
    batch_size=FLAGS.batch_size,
    epoches=FLAGS.epoches,
    activation=FLAGS.activation,
    columns=columns,
    field_nums=field_nums
    
    )
