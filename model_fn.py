import tensorflow as tf
from model import FM


def create_model_fn(model):
    def model_fn(features, labels, params, mode):
        if params.opt_type == "adm":
            optimizer = tf.train.AdamOptimizer(learning_rate=params.lr)
        elif params.opt_type == "adagrad":
            optimizer = tf.train.AdagradDAOptimizer(learning_rate=params.lr)
        elif params.opt_type == "gd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=params.lr)
        elif params.opt_type == "momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate=params.lr)
        else:
            raise ValueError("opt type must in adm, adagrad, gd, momentum")


        if params.threshold:
            threshold = params.threshold
        else:
            threshold = 0.5

        df_i = features["df_i"]
        df_v = features['df_v']
        
        logits = model(hparams=params, df_i=df_i, df_v=df_v).logits
        print(">>>>>>>>>", logits)
        
        if params.loss_type == "log_loss":
            logits = tf.nn.sigmoid(logits, name="sigmoid")
            print(logits)
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            predict = tf.cast(logits > threshold, dtype=tf.float32)
            predictions = {
                "predict_pro": logits,
                "predict": predict
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        if params.loss_type == "log_loss":
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits),name="loss")
            predict = tf.cast(logits > threshold, dtype=tf.float32)
            accuracy = tf.metrics.accuracy(labels=labels, predictions=predict, name="accuracy")
            auc = tf.metrics.auc(labels, predictions=logits, name="auc")
            metrics = {
                "loss": loss,
                "accuracy": accuracy,
                "auc": auc,
                "predic_pro":logits,
                "predic":predict
            }
            
        elif params.loss_type == "mse":
            loss = tf.losses.mean_squared_error(labels, logits, name="loss")
            metrics = {
                "loss": loss,
                "predic":logits
            }
        else:
            raise ValueError("loss type is mse or log_loss")

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        opt = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar("loss", loss)
            tf.summary.histogram("predict", logits)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=opt)
    return model_fn

    
