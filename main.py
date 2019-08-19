# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 02/08/19 
# @Contact: michealabaho265@gmail.com

from base_models import train_lstm, train_cnn, svm_tuned
import tensorflow as tf
import sys
import pandas as pd

tf.logging.set_verbosity(tf.logging.ERROR)

def main(model=''):
    ebm = 'labels_outcomes_2.csv'
    ebm_data = pd.read_csv(ebm, low_memory=False)

    # defining parameters
    tf.flags.DEFINE_float("validation_percentage", 0.2, "Percentage of data to be used for validation")
    tf.flags.DEFINE_string("glovec", "/users/phd/micheala/Documents/Github/pico-back-up/glove.840B.300d.txt", "pretrained_embedding")
    tf.flags.DEFINE_float("dropout_rate_lstm", 0.2, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_integer('embedding_dim_lstm', 150, 'embedding_dim')
    tf.flags.DEFINE_string("activation", 'sigmoid', "activation")
    tf.flags.DEFINE_string("kernel", "orthogonal", "kernel")
    tf.flags.DEFINE_float("learning_rate", .001, "learning_rate")
    tf.flags.DEFINE_integer("epochs", 200, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("batch_size_lstm", 512, "batch_size")

    # Model Hyperparameters
    tf.flags.DEFINE_integer("embedding_dim_cnn", 128, "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_float("dropout_rate_cnn", 0.5, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

    # Misc Parameters
    tf.flags.DEFINE_integer("batch_size_cnn", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
    tf.flags.DEFINE_integer("checkpoint_every", 10, "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

    Flags = tf.flags.FLAGS

    if model.lower() == 'lstm':
        x, y, clas, voc, emb_matrix = train_lstm.load_vectorize_shuffle_data(load_data=ebm_data,
                                                                             glovec=Flags.glovec,
                                                                             embedding_dim=Flags.embedding_dim_lstm,
                                                                             validation_percentage=Flags.validation_percentage,
                                                                             cross_validation=True)
        mod = train_lstm.train_cross_fold(x=x[:200],
                                          y=y[:200],
                                          classes=clas,
                                          vocabularly=voc,
                                          emb_matrix=emb_matrix,
                                          activation=Flags.activation,
                                          dropout_rate=Flags.dropout_rate_lstm,
                                          embedding_dim=Flags.embedding_dim_lstm,
                                          batch_size=Flags.batch_size_lstm,
                                          kernel=Flags.kernel,
                                          learning_rate=Flags.learning_rate,
                                          epochs=Flags.epochs)
    elif model.lower() == 'cnn':
        x_train, y_train, vocab_processor, x_dev, y_dev = train_cnn.pre_process(ebm_data)
        train_cnn.train(x_train=x_train[:50],
                        y_train=y_train[:50],
                        vocabulary=vocab_processor,
                        x_val=x_dev,
                        y_val=y_dev,
                        embedding_dim=Flags.embedding_dim_cnn,
                        l2_reg_lambda=Flags.l2_reg_lambda,
                        num_checkpoints=Flags.num_checkpoints,
                        checkpoint_every=Flags.checkpoint_every,
                        evaluate_every=Flags.evaluate_every,
                        batch_size=Flags.batch_size_cnn,
                        dropout_keep_prob=Flags.dropout_rate_cnn,
                        filter_sizes=Flags.filter_sizes,
                        num_filters=Flags.num_filters,
                        allow_soft_placement=Flags.allow_soft_placement,
                        log_device_placement=Flags.log_device_placement)

    elif model.lower() == 'svm':
        X_train, X_test, y_train, y_test, class_dict = svm_tuned.data_split(data=ebm_data, test_percentage=Flags.validation_percentage)
        saved_model = svm_tuned.svm_classifier(X_train[:50], y_train.values[:50], list(class_dict))
        svm_tuned.evaluate_model(saved_model, X_test, y_test, list(class_dict))

if len(sys.argv) < 2:
    raise ValueError("Check your arguments, Either one of these are missing LSTM or CNN or SVM")

input_1 = sys.argv[1]
main(model=input_1)


