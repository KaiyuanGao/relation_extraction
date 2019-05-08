# coding=utf-8
# @author: kaiyuan
# blog: https://blog.csdn.net/Kaiyuan_sjtu
import tensorflow as tf
import numpy as np
import os
import datetime
import time
import sys
import cnn_m
from utils import load_data, write_results, inputs


tf.flags.DEFINE_string("train_file", "../data/clean_data.txt", "Data source for the training.")
tf.flags.DEFINE_string("test_file", "../data/clean_data_test.txt", "Data source for the test.")
tf.flags.DEFINE_string("vocab_file", "../data/vocab.txt", "Data source for text vocabulary.")
tf.flags.DEFINE_string("embed_file", "../data/embed50.senna.npy", "senna words embeddding as the paper mentioned")
tf.flags.DEFINE_string("senna_embed50_file", "../data/embed50.senna.npy","senna words embeddding")
tf.flags.DEFINE_string("senna_words_file", "../data/senna_words.lst", "senna words list")
tf.flags.DEFINE_string("trimmed_embed50_file", "../data/embed50.trim.npy", "trimmed senna embedding")
tf.flags.DEFINE_string("train_record", "../data/train.tfrecord", "training file of TFRecord format")
tf.flags.DEFINE_string("test_record", "../data/test.tfrecord", "Test file of TFRecord format")
tf.flags.DEFINE_string("relation_file", "../data/relations.txt", "relation file")
tf.flags.DEFINE_string("results_file", "../data/results.txt", "predicted results file")
tf.flags.DEFINE_string("logdir", "..\saved_models", "where to save the model")

tf.flags.DEFINE_integer("max_len", 96, "max length of sentences")
tf.flags.DEFINE_integer("num_relations", 19, "number of relations")
tf.flags.DEFINE_integer("word_dim", 50, "word embedding size")
tf.flags.DEFINE_integer("num_epochs", 150, "number of epochs")
tf.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.flags.DEFINE_integer("pos_num", 123, "number of position feature")
tf.flags.DEFINE_integer("pos_dim", 5, "position embedding size")
tf.flags.DEFINE_integer("num_filters", 100, "cnn number of output unit")
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
tf.flags.DEFINE_float("keep_prob", 0.5, "dropout keep probability")

tf.flags.DEFINE_boolean('test', False, 'set True to test')
#tf.flags.DEFINE_boolean('train', True, 'set True to test')

FLAGS = tf.flags.FLAGS


def train(sess, m_train, m_val):
    n = 1
    best = .0
    best_step = n
    start_time = time.time()
    orig_begin_time = start_time

    fetches = [m_train.train_op, m_train.loss, m_train.accuracy]

    while True:
        try:
            _, loss, acc = sess.run(fetches)

            epoch = n // 80
            if n % 80 == 0:
                now = time.time()
                duration = now - start_time
                start_time = now
                v_acc = sess.run(m_val.accuracy)
                if best < v_acc:
                    best = v_acc
                    best_step = n
                    m_train.save(sess, best_step)
                print("Epoch %d, loss %.2f, acc %.2f, val_acc %.4f, time %.2f" %
                      (epoch, loss, acc, v_acc, duration))
                sys.stdout.flush()
            n += 1
        except tf.errors.OutOfRangeError:
            break

    duration = time.time() - orig_begin_time
    duration /= 3600
    print('Done training, best_step: %d, best_acc: %.4f' % (best_step, best))
    print('duration: %.2f hours' % duration)
    sys.stdout.flush()

def test(sess, m_val):
    m_val.restore(sess)
    fetches = [m_val.accuracy, m_val.prediction]
    accuracy, predictions = sess.run(fetches)
    print('accuracy: %.4f' % accuracy)

    write_results(predictions, FLAGS.relation_file, FLAGS.results_file)

def main(_):
    with tf.Graph().as_default():
        train_data, test_data, word_embed = inputs()

        m_train, m_val = cnn_m.build_train_valid_model(word_embed, train_data, test_data)

        m_train.set_saver('cnn-%d-%d' % (FLAGS.num_epochs, FLAGS.word_dim))

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        config = tf.ConfigProto()

        with tf.Session(config=config) as sess:
            sess.run(init_op)
            print('='*80)
            if FLAGS.test:
                test(sess, m_val)
            else:
                train(sess, m_train, m_val)

if __name__ == '__main__':
    tf.app.run()



