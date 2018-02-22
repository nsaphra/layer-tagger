import sys
import os
import numpy
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy import sparse
import pickle
import ast
import json

import data

parser = argparse.ArgumentParser(description='Evaluate language model semantic and syntactic features')

# Model parameters.
parser.add_argument('--checkpoint', type=str,
                    help='model checkpoint to use')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--vocab', type=str,
                    help='vocab file')
parser.add_argument('--train-file', type=str,
                    help='location of the valid and test data corpus')
parser.add_argument('--valid-file', type=str,
                    help='location of the valid data corpus')
parser.add_argument('--test-file', type=str,
                    help='location of the test data corpus')
parser.add_argument('--original-src', type=str,
                    help='location of the modules required for the model')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--batch-size', type=int, default=60)
parser.add_argument('--log-interval', type=int, default=200)
parser.add_argument('--test-length', type=int, default=None,
                    help='number of lines to test in the test corpus')

args = parser.parse_args()

sys.path.append(args.original_src)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(args.train_file+'.tok', args.valid_file+'.tok', args.test_file+'.tok', vocab_file=args.vocab, test_length=args.test_length)

def batchify(data, bsz, use_cuda=True):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda and use_cuda:
        data = data.cuda()
    return data

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=True)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

print('Batching eval text data.')
test_data = batchify(corpus.test, args.batch_size)

print('Batching eval pos data.')
# TODO use permutation
pos_corpus = data.Corpus(args.train_file+'.sem', args.valid_file+'.sem', args.test_file+'.sem', test_length=args.test_length)
test_pos_tags = batchify(pos_corpus.test, args.batch_size)

def get_pos_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    return source[i+1:i+1+seq_len].view(-1)

idx2word = corpus.dictionary.idx2word
word2idx = corpus.dictionary.word2idx
vocab_size = len(corpus.dictionary.idx2word)

criterion = nn.CrossEntropyLoss()

def evaluate(data_source, test_pos_tags):
    ntokens = len(corpus.dictionary)
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    num_batches = len(data_source) // args.bptt
    total_loss = 0
    cur_loss = 0
    for batch,i in enumerate(range(0, data_source.size(0) - 1, args.bptt)):
        data, targets = get_batch(data_source, i)
        target_pos_tags = get_pos_batch(test_pos_tags, i)

        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        loss = criterion(output_flat, targets)
        cur_loss += loss
        total_loss += loss

        hidden = repackage_hidden(hidden)

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = cur_loss.data[0]
            average_loss = cur_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| {:5d}/{:5d} batches | ms/batch {:5.2f} | {{:5.2f}} loss |'.format(
                    batch, num_batches, elapsed * 1000 / args.log_interval, average_loss))
            cur_loss = 0

            start_time = time.time()

    return {'loss': total_loss.data[0] / num_batches}

# Run on test data.
print("test data size ", test_data.size())

test_stats = evaluate(test_data, test_pos_tags)
print('=' * 89)
print('| End | test ppl {:8.2f} | '.format(
      math.exp(test_stats['loss'])),
      ' | '.join(['{} {:5.2f}'.format(k, v) for k,v in test_stats.items()]),
      ' |')
print('=' * 89)
