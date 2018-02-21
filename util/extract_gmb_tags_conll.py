from pathlib import Path
import argparse
import sys

def all_file_tags(fname):
    tokens = []
    semantic_tags = []
    for line in open(fname, encoding='utf-8'):
        line = line.strip()
        if line == '':
            yield (tokens, semantic_tags)
            tokens = []
            semantic_tags = []
            continue
        try:
            token, sem = line.split('\t')
        except ValueError: # token contains white space
            print('Skipping sentence containing white space token: ', line,
                  file=sys.stderr)
            tokens = []
            semantic_tags = []
            continue
        if not args.keep_mwe and '~' in token:
            split_tokens = token.split('~')
            split_semantic = [sem] * len(split_tokens)

            token = ' '.join(split_tokens)
            sem = ' '.join(split_semantic)
        tokens.append(token)
        semantic_tags.append(sem)
    assert(len(tokens) == len(semantic_tags))

parser = argparse.ArgumentParser(description='Process gmb conll files into token and tag files')
parser.add_argument('--data', type=str, default='data/stag',
                    help='location of the gmb data directory')
parser.add_argument('--keep_mwe', action='store_true')
args = parser.parse_args()

for filename in Path(args.data).glob('**/*.conll'):
    print('extracting tags from ', filename)
    tokens_out = filename.with_suffix('.tok').open('w', encoding='utf-8')
    semantic_out = filename.with_suffix('.sem').open('w', encoding='utf-8')
    for (tokens, semantic_tags) in all_file_tags(filename):
        print(' '.join(tokens), file=tokens_out)
        print(' '.join(semantic_tags), file=semantic_out)
    tokens_out.close()
    semantic_out.close()
