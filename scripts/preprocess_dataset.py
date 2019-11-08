'''
Largely based on get-data-nmt.sh
'''

from trainlib import set_random_seeds
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
    initiate(logger=True, gpus=True, commit_id=True, random_seed=True)

    add_argument('langs', dtype=str, nargs=2)
    add_argument('codes', dtype=str)
    add_argument('seed', dtype=int, default=1234)
    add_argument('split_lines', dtype=int, nargs=2)
    add_argument('eat', dtype=str, default='', choices=['', 'eat', 'neo'])
    add_argument('dataset', dtype=str, default='multi30k', choices=['multi30k', 'iwslt'])
    add_argument('linear', dtype=bool, default=False)
    add_argument('pair', dtype=str, default='')
    parse_args(show=True)

    # Use random seed to make sure train split is persistent.
    random.seed(g.seed)

    # Set data directory.
    data_dir = Path(f'/scratch/j_luo/data/{g.dataset}/')

    # Set constant for the actions.
    set_action_constant('MOSES_DIR', MOSES_DIR)
    set_action_constant('FASTBPE', FASTBPE)
    set_action_constant('MAIN_DIR', MAIN_DIR)
    set_action_constant('NUM_THREADS', NUM_THREADS)
    if g.eat:
        set_action_constant('EAT_DIR', EAT_DIR)

    # Get the language pair.
    lang1, lang2 = sorted(g.langs)
    sorted_pair = f'{lang1}-{lang2}'
    pair = sorted_pair if g.pair == '' else g.pair

    # Get the folders.
    out_dir = data_dir / f'{pair}'
    raw_dir = out_dir / 'raw'
    processed_dir = out_dir / 'processed'

    # ---------------------------------------------------------------------------- #
    #                                   Main body                                  #
    # ---------------------------------------------------------------------------- #

    if g.dataset == 'multi30k':
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

    else:
        dataset_names = {
            'dev': ['dev2010', 'tst2010', 'tst2011'],
            'test': ['tst2012', 'tst2013', 'tst2014']
        }
        # Get all sources.
        sources = dict()
        for lang in g.langs:
            key = Key('train', lang)
            source = FormatFile(raw_dir, f'train', lang, 'txt', pair=pair, compat_main=f'train.{pair}.{lang}')
            sources[key] = source
        for lang in g.langs:
            for split, names in dataset_names.items():
                for name in names:
                    key = Key(name, lang)
                    source = FormatFile(raw_dir, name, lang, 'txt', pair=pair,
                                        compat_main=f'IWSLT16.TED.{name}.{pair}.{lang}')
                    sources[key] = source
        pipeline = Pipeline(sources)

        # Merge all dev sets and test sets.
        for lang in g.langs:
            for split in ['dev', 'test']:
                merge_to = FormatFile(raw_dir, split, lang, 'txt', pair=pair)
                merge_to_key = Key(split, lang)
                to_merge_keys = [Key(name, lang) for name in dataset_names[split]]
                pipeline.merge(to_merge_keys, merge_to_key, merge_to)

        # Take two non-overlapping subsets from training.
        key1 = Key('train', lang1)
        key2 = Key('train', lang2)
        pipeline.split_n_parts([key1, key2], 2, [0, 1])

    # Now training sets are not parallel.
    link1 = pipeline.sources[key1].remove_pair().remove_part()
    link2 = pipeline.sources[key2].remove_pair().remove_part()
    pipeline.link(key1, link1)
    pipeline.link(key2, link2)

    # Preprocess.
    pipeline.preprocess()

    # For eat, we need to get the eat files first.
    if g.eat:
        # Save conll-related files here.
        pipeline.parse(folder=out_dir / 'conll')
        pipeline.collapse()
        subfolder = f'processed-{g.eat}'
        if g.linear:
            subfolder += '-linear'
        eat_dir = out_dir / subfolder
        if g.eat == 'eat':
            pipeline.convert_eat(folder=eat_dir)
        else:
            pipeline.convert_neo(linear=g.linear, folder=eat_dir)
        # After conversion, some texts are missing due to empty EAT sequence.
        for split in ['dev', 'test']:
            pipeline.align(Key(split, lang1), Key(split, lang2), op=g.eat)

    # Apply BPE.
    folder = eat_dir if g.eat else processed_dir
    pipeline.apply_bpe(codes=g.codes, folder=folder)

    # Extract full vocab.
    pipeline.extract_joint_vocab(Key('train', lang1), Key('train', lang2))

    # Binarize everything.
    pipeline.binarize()

    # Follow XLM's naming tradition and link to proper files.
    for split in ['train', 'dev', 'test']:
        for lang in g.langs:
            src_key = Key(split, lang)
            main = 'valid' if split == 'dev' else split
            mono_link = FormatFile(folder, main, lang, 'pth')
            pipeline.link(src_key, mono_link)

            if split != 'train':
                para_link = FormatFile(folder, main, lang, 'pth', pair=pair)
                pipeline.link(src_key, para_link)

                if g.pair != '' and g.pair != sorted_pair:
                    para_link = FormatFile(folder, main, lang, 'pth', pair=sorted_pair)
                    pipeline.link(src_key, para_link)
