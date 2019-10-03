'''
Largely based on get-data-nmt.sh
'''

import random
from collections import namedtuple
from enum import Enum
from pathlib import Path

from arglib import add_argument, g, parse_args
from devlib import initiate
from devlib.preprocess.action import set_action_constant
from devlib.preprocess.format_file import FormatFile
from devlib.preprocess.pipeline import Pipeline

MAIN_DIR = Path('/scratch/j_luo/eat-nmt/XLM/')
DATA_DIR = Path('/scratch/j_luo/data/multi30k/')
TOOLS_DIR = Path('/scratch/j_luo/eat-nmt/XLM/tools/')
MULTI30K_DOMAIN_URL = 'https://github.com/multi30k/dataset/blob/master/data/task1/raw/'
MOSES_DIR = TOOLS_DIR / 'mosesdecoder'
FASTBPE = TOOLS_DIR / 'fastBPE/fast'
NUM_THREADS = 8
EAT_DIR = Path('/scratch/j_luo/EAT/')


class TestSet(Enum):
    FLICKR_2016 = '2016_flickr'
    FLICKR_2017 = '2017_flickr'
    FLICKR_2018 = '2018_flickr'
    MSCOCO_2017 = '2017_mscoco'


test_sets = {
    'de': {
        TestSet.FLICKR_2016,
        TestSet.FLICKR_2017,
        TestSet.MSCOCO_2017
    },
    'fr': {
        TestSet.FLICKR_2016,
        TestSet.FLICKR_2017,
        TestSet.MSCOCO_2017
    },
    'en': {
        TestSet.FLICKR_2016,
        TestSet.FLICKR_2017,
        TestSet.FLICKR_2018,
        TestSet.MSCOCO_2017
    },
    'cs': {
        TestSet.FLICKR_2016
    }
}

Key = namedtuple('Key', ['main', 'lang'])

if __name__ == "__main__":
    initiate(logger=True, gpus=True)

    add_argument('langs', dtype=str, nargs=2)
    add_argument('codes', dtype=str)
    add_argument('seed', dtype=int, default=1234)
    add_argument('split_lines', dtype=int, nargs=2)
    add_argument('eat', dtype=bool, default=False)
    parse_args()

    # Use random seed to make sure train split is persistent.
    random.seed(g.seed)

    # Set constant for the actions.
    set_action_constant('MOSES_DIR', MOSES_DIR)
    set_action_constant('FASTBPE', FASTBPE)
    set_action_constant('MAIN_DIR', MAIN_DIR)
    set_action_constant('NUM_THREADS', NUM_THREADS)
    if g.eat:
        set_action_constant('EAT_DIR', EAT_DIR)

    # Get the language pair.
    lang1, lang2 = sorted(g.langs)
    pair = f'{lang1}-{lang2}'

    # Get the folders.
    out_dir = DATA_DIR / f'{pair}'
    raw_dir = out_dir / 'raw'
    processed_dir = out_dir / 'processed'

    # ---------------------------------------------------------------------------- #
    #                                   Main body                                  #
    # ---------------------------------------------------------------------------- #

    # For test set, we need to figure out what to download.
    test_to_download = [f'test_{dataset.value}' for dataset in test_sets[lang1] & test_sets[lang2]]

    # Get all urls.
    urls = dict()
    for lang in g.langs:
        urls[Key('train', lang)] = f'{MULTI30K_DOMAIN_URL}/train.{lang}.gz?raw=true'
        urls[Key('dev', lang)] = f'{MULTI30K_DOMAIN_URL}/val.{lang}.gz?raw=true'
        for name in test_to_download:
            urls[Key(name, lang)] = f'{MULTI30K_DOMAIN_URL}/{name}.{lang}.gz?raw=true'

    # Download everything.
    pipeline = Pipeline(urls)
    pipeline.download(folder=raw_dir, pair=pair, ext='gz')

    # Merge all test sets.
    for lang in g.langs:
        merge_to = FormatFile(raw_dir, 'test', lang, 'gz', pair=pair)
        merge_to_key = Key('test', lang)
        to_merge_keys = [Key(name, lang) for name in test_to_download]
        pipeline.merge(to_merge_keys, merge_to_key, merge_to)

    # Decompress everything.
    pipeline.decompress()

    # Take two non-overlapping subsets from training.
    num_lines = sum(g.split_lines)
    indices = list(range(num_lines))
    random.shuffle(indices)
    indices1 = indices[:g.split_lines[0]]
    indices2 = indices[g.split_lines[0]:num_lines]
    line_ids = [indices1, indices2]
    key1 = Key('train', lang1)
    key2 = Key('train', lang2)
    pipeline.split(key1, line_ids, 0)
    pipeline.split(key2, line_ids, 1)

    # Now training sets are not parallel.
    link1 = pipeline.sources[key1].remove_pair().remove_part()
    link2 = pipeline.sources[key2].remove_pair().remove_part()
    pipeline.link(key1, link1)
    pipeline.link(key2, link2)

    # Preprocess.
    pipeline.preprocess()

    # For eat, we need to get the eat files first.
    if g.eat:
        eat_dir = out_dir / 'processed-eat'
        pipeline.parse(folder=eat_dir)
        pipeline.collapse()
        pipeline.convert_eat()
        # After conversion, some texts are missing due to empty EAT sequence.
        for split in ['dev', 'test']:
            pipeline.align(Key(split, lang1), Key(split, lang2))

    # Apply BPE.
    folder = eat_dir if g.eat else processed_dir
    pipeline.apply_bpe(codes=g.codes, folder=folder)

    # Extract full vocab.
    pipeline.extract_joint_vocab(Key('train', lang1), Key('train', lang2))

    # Binarize everything.
    pipeline.binarize()

    # Link monolingual validation and test data to parallel data.
    for split in ['dev', 'test']:
        for lang in g.langs:
            src_key = Key(split, lang)
            link = pipeline.sources[src_key].remove_pair()
            pipeline.link(src_key, link)
