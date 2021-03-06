# -*- coding: utf-8 -*-

"""
Given tokenized text and corresponding document ids, output a file which lists, for each token in the text,
the document id it came from. Retain only the longest articles. This will serve as a proxy for sentence topic.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
import sys
from collections import defaultdict
import operator
import random
import os

parser = argparse.ArgumentParser(description='Output the document ids')
parser.add_argument('--tokens_in', type=str, default='../../data/wiki_polyglot/all.txt')
parser.add_argument('--dir_out', type=str, default='data/document_id/')
parser.add_argument('--ids_in', type=str, default='../../data/wiki_polyglot/doc_ids.txt')
parser.add_argument('--pos_in', type=str, default='../../data/wiki_polyglot/pos.txt')
parser.add_argument('--number_documents', type=int, default=1000)
args = parser.parse_args()

topics_train_file = open(os.path.join(args.dir_out, 'train.idx'), 'w')
topics_dev_file = open(os.path.join(args.dir_out, 'dev.idx'), 'w')
topics_test_file = open(os.path.join(args.dir_out, 'test.idx'), 'w')
pos_train_file = open(os.path.join(args.dir_out, 'train.pos'), 'w')
pos_dev_file = open(os.path.join(args.dir_out, 'dev.pos'), 'w')
pos_test_file = open(os.path.join(args.dir_out, 'test.pos'), 'w')
tokens_train_file = open(os.path.join(args.dir_out, 'train.tok'), 'w')
tokens_dev_file = open(os.path.join(args.dir_out, 'dev.tok'), 'w')
tokens_test_file = open(os.path.join(args.dir_out, 'test.tok'), 'w')
tokens_in_file = open(args.tokens_in, 'r')
topics_in_file = open(args.ids_in, 'r')
pos_in_file = open(args.pos_in, 'r')

with open(args.ids_in, 'r') as f:
    document_length = defaultdict(int)
    for line in f:
        document = line.strip()
        document_length[document] += 1
sorted_documents = sorted(document_length.items(), key=operator.itemgetter(1))
minimum_document_length = sorted_documents[-args.number_documents][1]
documents_to_keep = set([x for x,y in sorted_documents[-args.number_documents:]])

print('minimum document length: ', minimum_document_length)

for line in tokens_in_file:
    tokens = line.strip().split()
    topic = topics_in_file.readline().strip()
    pos = pos_in_file.readline()
    if topic not in documents_to_keep:
        continue
    topics = [topic] * len(tokens)

    partition = random.randrange(10)
    if partition < 1:
        print(line, file=tokens_dev_file, end='')
        print(' '.join(topics), file=topics_dev_file)
        print(pos, file=pos_dev_file, end='')
    elif partition < 3:
        print(line, file=tokens_test_file, end='')
        print(' '.join(topics), file=topics_test_file)
        print(pos, file=pos_test_file, end='')
    else:
        print(line, file=tokens_train_file, end='')
        print(' '.join(topics), file=topics_train_file)
        print(pos, file=pos_train_file, end='')
