# coding=utf-8
# @author: kaiyuan
# blog: https://blog.csdn.net/Kaiyuan_sjtu

"""
The file to preprocess the original raw data: SemEval2010_task8

*********************
RAW_SAMPLES ARE LIKE:
        3	"The <e1>author</e1> of a keygen uses a <e2>disassembler</e2> to look at the raw assembly code."
        Instrument-Agency(e2,e1)
        Comment:
*********************
NEW SAMPLES AFTER PROCESS ARE LIKE:
        11 1 1 7 7 the author of a keygen uses a disassembler to look at the raw assembly code
        【relation_type entity1_start_index entity1_end_index entity2_start_index entity2_end_index sentence】
*********************

"""

import numpy as np
import pandas as pd
import nltk
import re
from collections import namedtuple


PositionPair = namedtuple('PosPair', 'first last')

def rel2id(relation_path):
    dic = {}
    with open(relation_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            dic[line.split()[1]] = int(line.split()[0])
    return dic

def id2rel(relation_path):
    dic = {}
    with open(relation_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            dic[line.split()[0]] = int(line.split()[1])
    return dic

def get_new_pos(entity, sentence):
    n = len(entity)
    tokens = sentence.split()
    for i in range(len(tokens)):
        if tokens[i:i + n] == entity:
            first, last = i, i + n - 1
            return PositionPair(first, last)

def amend_str(text):
    """convert string to standard sentence"""
    text = text.lower()
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()

def get_clean_data(file_path, new_path, relation_path):
    r2id = rel2id(relation_path)
    lines = [line.strip() for line in open(file_path, 'r', encoding='utf-8')]

    with open(new_path, 'w', encoding='utf-8') as f:
        for i in range(0, len(lines), 4):
            label = r2id[lines[i+1].strip()]

            sentence = lines[i].strip().split('\t')[1]
            sentence = sentence.replace('<e1>', ' e11 ')
            sentence = sentence.replace('</e1>', ' e12 ')
            sentence = sentence.replace('<e2>', ' e21 ')
            sentence = sentence.replace('</e2>', ' e22 ')
            sentence = sentence.lower()
            sentence = sentence.replace("'", " ' ")
            #sentence = amend_str(sentence)
            tokens = nltk.word_tokenize(sentence)
            e1_s = tokens.index("e11")
            e1_e = tokens.index("e12")
            e2_s = tokens.index("e21")
            e2_e = tokens.index("e22")

            e1 = tokens[e1_s + 1:e1_e]
            e2 = tokens[e2_s + 1:e2_e]
            for t in ["e11", "e12", "e21", "e22", '-']:
                if t in tokens:
                    tokens.remove(t)
            n = len(tokens)
            sentence = ' '.join(tokens[1:n-2])
            new_e1 = get_new_pos(e1, sentence)
            new_e2 = get_new_pos(e2, sentence)

            f.write('%d %d %d %d %d %s\n' %(label, new_e1.first, new_e1.last, new_e2.first, new_e2.last, sentence))

if __name__ == '__main__':
    file_path = '../data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    test_file_path = '../data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
    new_path = '../data/clean_data.txt'
    new_path_test = '../data/clean_data_test.txt'
    relation_path = '../data/relations.txt'
    get_clean_data(test_file_path, new_path_test, relation_path)

