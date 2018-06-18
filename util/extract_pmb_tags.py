# -*- coding: utf-8 -*-

"""
Process xml files from [1] into separate token and tag files.

[1] Abzianidze, Lasha, et al. "The parallel meaning bank: Towards a multilingual corpus of translations annotated with compositional meaning representations." arXiv preprint arXiv:1702.03964 (2017).
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
import sys

parser = argparse.ArgumentParser(description='Process all pmb files into token and tag files')
parser.add_argument('--data', type=str, default='data/pmb-1.0.0/data',
                help='location of the pmb data directory')
parser.add_argument('--tokens_out', type=str, default='data/pmb-1.0.0/tokens.txt')
parser.add_argument('--semtags_out', type=str, default='data/pmb-1.0.0/semtags.txt')
parser.add_argument('--postags_out', type=str, default='data/pmb-1.0.0/postags.txt')
parser.add_argument('--keep_mwe', action='store_true')
args = parser.parse_args()

def all_file_tags(fname):
    tokens = []
    semantic_tags = []
    pos_tags = []
    tree = ET.parse(fname).getroot()
    for tags in tree.iter('tags'):
        token = None
        sem = None
        pos = None
        for tag in tags.findall('tag'):
            tag_type = tag.get('type')
            if tag_type == 'tok':
                token = tag.text
                if ' ' in token:
                    print('Skipping sentence containing white space token: ', tok,
                          file=sys.stderr)
                    return (None, None, None)
            elif tag_type == 'pos':
                pos = tag.text
            # elif tag_type == 'sem':
                sem = tag.text
        if token and pos and sem and token != 'ø':
            if not args.keep_mwe and '~' in token:
                assert(token.find(' ') < 0)
                split_tokens = token.split('~')
                split_semantic = [sem] * len(split_tokens)
                split_pos = [pos] * len(split_tokens)

                token = ' '.join(split_tokens)
                sem = ' '.join(split_semantic)
                pos = ' '.join(split_pos)
            tokens.append(token)
            semantic_tags.append(sem)
            pos_tags.append(pos)
    assert(len(tokens) == len(semantic_tags) and len(pos_tags) == len(tokens))
    return (tokens, semantic_tags, pos_tags)

# process all files in data directory
tokens_out = open(args.tokens_out, 'w', encoding='utf-8')
semantic_out = open(args.semtags_out, 'w', encoding='utf-8')
pos_out = open(args.postags_out, 'w', encoding='utf-8')
for filename in Path(args.data).glob('**/en.drs.xml'):
    print('extracting tags from ', filename)
    (tokens, semantic_tags, pos_tags) = all_file_tags(filename)
    if tokens == None:
        continue
    print(' '.join(tokens).lower(), file=tokens_out)
    print(' '.join(semantic_tags), file=semantic_out)
    print(' '.join(pos_tags), file=pos_out)
tokens_out.close()
semantic_out.close()
pos_out.close()
