# coding=utf-8
# @author: kaiyuan
# blog: https://blog.csdn.net/Kaiyuan_sjtu

import os
import re
import numpy as np
import tensorflow as tf
from collections import namedtuple
from nltk.corpus import wordnet as wn

PAD_WORD = "<pad>"

Raw_Example = namedtuple('Raw_Example', 'label entity1 entity2 sentence')
PositionPair = namedtuple('PosPair', 'first last')

FLAGS = tf.app.flags.FLAGS

def load_data(data_file):
    """load data.txt to a list of raw_examples"""
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split(' ')
            sent = words[5:]
            n = len(sent)
            if FLAGS.max_len < n:
                FLAGS.max_len = n

            label = int(words[0])
            e1 = PositionPair(int(words[1]), int(words[2]))
            e2 = PositionPair(int(words[3]), int(words[4]))
            example = Raw_Example(label, e1, e2, sent)
            data.append(example)
    print(FLAGS.max_len)
    return data

def build_vocab(train_data, test_data, vocab_file):
    """to build a vocabulay using train and test data"""
    if not os.path.exists(vocab_file):
        vocab = set()
        for example in train_data + test_data:
            for w in example.sentence:
                vocab.add(w)

        with open(vocab_file, 'w') as f:
            for w in sorted(list(vocab)):
                f.write('%s\n' % w)
            f.write('%s\n' % PAD_WORD)

def load_vocab(vocab_file):
    """load vocab txt"""
    vocab = []
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            w = line.strip()
            vocab.append(w)
    return vocab

def sent2id(raw_data, word2id):
    """convert a string to vector"""
    pad_id = word2id[PAD_WORD]
    for example in raw_data:
        for idx, word in enumerate(example.sentence):
            example.sentence[idx] = word2id[word]
        pad_n = FLAGS.max_len-len(example.sentence)
        example.sentence.extend(pad_n*[pad_id])

def load_embedding(embed_file, vocab_file):
    """load embedding file"""
    embedding = np.load(embed_file)
    word2id = {}
    words = load_vocab(vocab_file)
    for id, w in enumerate(words):
        word2id[w] = id
    return embedding, word2id

def trim_embeddings(vocab_file, pretrain_embed_file, pretrain_words_file, trimed_embed_file):
    """trim unnessary word embeddings to get a dense representation"""
    if not os.path.exists(trimed_embed_file):
        pretrain_embed, pretrain_word2id = load_embedding(pretrain_embed_file, pretrain_words_file)
        word_embed = []
        vocab = load_vocab(vocab_file)
        for w in vocab:
            if w in pretrain_word2id:
                id = pretrain_word2id[w]
                word_embed.append(pretrain_embed[id])
            else:
                vec = np.random.normal(0, 0.1, [FLAGS.word_dim])
                word_embed.append(vec)
        pad_id = -1
        word_embed[pad_id] = np.zeros([FLAGS.word_dim])
        np.save(trimed_embed_file, np.array(word_embed, dtype=np.float32))

    word_embed, vocab_file = load_embedding(trimed_embed_file, vocab_file)
    return word_embed, vocab_file


def get_lexical_feature(example):
    """exetrac lexical features"""
    def entity_context(e_idx, sent):
        """get a list [w(e-1), w(e), w(e+1)]"""
        context = []
        context.append(sent[e_idx])
        # todo: what if len(entity) > 1?
        if e_idx >= 1:
            context.append(sent[e_idx-1])
        else:
            context.append(sent[e_idx])
        if e_idx < len(sent)-1:
            context.append(sent[e_idx+1])
        else:
            context.append(sent[e_idx])
        return context
    e1_idx = example.entity1.first
    e2_idx = example.entity2.first

    context1 = entity_context(e1_idx, example.sentence)
    context2 = entity_context(e2_idx, example.sentence)
    #hyper1 = wn.synset(example.sentence[e1_idx] + '.n.01').hypernyms()
    #hyper2 = wn.synset(example.sentence[e2_idx] + '.n.01').hypernyms()

    lexical = context1 + context2 #+ hyper1 + hyper2 # six words
    return lexical

def get_position_feature(example):
    def distance(n):
        """convert relative distance to positive number"""
        ## todo: not good enough, fix
        if n < -60:
            return 0
        elif -60 <= n <= 60:
            return n+61
        return 122

    e1_idx = example.entity1.first
    e2_idx = example.entity2.first
    position1 = []
    position2 = []
    length = len(example.sentence)
    for i in range(length):
        position1.append(distance(i - e1_idx))
        position2.append(distance(i - e2_idx))

    return position1, position2

def write_to_tfrecord(raw_data, new_file):
    """
    convert the raw_data to tf.trian.SequenceExample and write to tfrecord
    :param raw_data:
    :param new_file:
    :return:
    """
    if not os.path.exists(new_file):
        writer = tf.python_io.TFRecordWriter(new_file)
        for raw_example in raw_data:
            example = get_sequence_example(raw_example)
            writer.write(example.SerializeToString())
        writer.close()

def get_sequence_example(example):
    """
    build tf.train.SequenceExample from Raw_Example
    context features: lexical, rid, direction (mtl)
    sequence features: sentence, position1, position2
    :param example:
    :return: tf.trian.SequenceExample
    """
    ex = tf.train.SequenceExample()
    lexical = get_lexical_feature(example)
    ex.context.feature['lexical'].int64_list.value.extend(lexical)

    rid = example.label
    ex.context.feature['rid'].int64_list.value.append(rid)

    for word_id in example.sentence:
        word = ex.feature_lists.feature_list['sentence'].feature.add()
        word.int64_list.value.append(word_id)

    pos1, pos2 = get_position_feature(example)
    for pos_val in pos1:
        pos = ex.feature_lists.feature_list['pos1'].feature.add()
        pos.int64_list.value.append(pos_val)
    for pos_val in pos2:
        pos = ex.feature_lists.feature_list['pos2'].feature.add()
        pos.int64_list.value.append(pos_val)

    return ex


def get_parse_tfexample(serialized_example):
    """
    parse serialized tf.train.SequenceExample to tensors
    :param serialized_example:
    :return:
    """
    context_features = {
        'lexical': tf.FixedLenFeature([6], tf.int64),
        'rid': tf.FixedLenFeature([], tf.int64)}
    sequence_features = {
        'sentence': tf.FixedLenSequenceFeature([], tf.int64),
        'pos1': tf.FixedLenSequenceFeature([], tf.int64),
        'pos2': tf.FixedLenSequenceFeature([], tf.int64)}
    context_dict, sequence_dict = tf.parse_single_sequence_example(
        serialized_example,
        context_features=context_features,
        sequence_features=sequence_features)

    sentence = sequence_dict['sentence']
    pos1 = sequence_dict['pos1']
    pos2 = sequence_dict['pos2']
    lexical = context_dict['lexical']
    rid = context_dict['rid']

    return lexical, rid, sentence, pos1, pos2

def read_tfrecord_to_batch(file, epoch, batch_size, shuffle=True):
    """generate batch_size tensors for our model"""
    with tf.device('/cpu:0'):
        dataset = tf.data.TFRecordDataset([file])  # tfrecord
        dataset = dataset.map(get_parse_tfexample)  # tensors
        dataset = dataset.repeat(epoch)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        batch = iterator.get_next()
        return batch

def inputs():
    raw_train_data = load_data(FLAGS.train_file)
    raw_test_data = load_data(FLAGS.test_file)

    # only run for the first try
    build_vocab(raw_train_data, raw_test_data, FLAGS.vocab_file)

    #word_embedding, word2id = load_embedding(FLAGS.embed_file, FLAGS.vocab_file)
    if FLAGS.word_dim == 50:
        word_embed, word2id = trim_embeddings(FLAGS.vocab_file, FLAGS.senna_embed50_file,
                                               FLAGS.senna_words_file, FLAGS.trimmed_embed50_file)
    elif FLAGS.word_dim == 300:
        word_embed, word2id = trim_embeddings(FLAGS.vocab_file, FLAGS.google_embed300_file,
                                               FLAGS.google_words_file, FLAGS.trimmed_embed300_file)

    sent2id(raw_train_data, word2id)
    sent2id(raw_test_data, word2id)

    train_record = FLAGS.train_record
    test_record = FLAGS.test_record

    write_to_tfrecord(raw_train_data, train_record)
    write_to_tfrecord(raw_test_data, test_record)

    pad_value = word2id[PAD_WORD]
    train_data = read_tfrecord_to_batch(train_record, FLAGS.num_epochs, FLAGS.batch_size,  shuffle=True)
    test_data = read_tfrecord_to_batch(test_record, FLAGS.num_epochs, 2717,  shuffle=False)

    return train_data, test_data, word_embed

def write_results(pred, relations_file, results_file):
    """write to test result to file for evaluate"""
    start_no = 8001
    relations = []
    with open(relations_file) as f:
        for line in f:
            segment = line.strip().split()
            relations.append(segment[1])

    with open(results_file, 'w', encoding='utf-8') as f:
        for idx, id in enumerate(pred):
            rel = relations[id]
            f.write('%d\t%s\n' % (start_no+idx, rel))




