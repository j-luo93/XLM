'''
Largely based on get-data-nmt.sh
'''

import logging
import random
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

from arglib import add_argument, g, parse_args
from trainlib import create_logger

MAIN_DIR = Path('/scratch/j_luo/eat-nmt/XLM/')
DATA_DIR = Path('/scratch/j_luo/data/multi30k/')
TOOLS_DIR = Path('/scratch/j_luo/eat-nmt/XLM/tools/')
MULTI30K_DOMAIN_URL = 'https://github.com/multi30k/dataset/blob/master/data/task1/raw/'
MOSES = TOOLS_DIR / 'mosesdecoder'
REPLACE_UNICODE_PUNCT = MOSES / 'scripts/tokenizer/replace-unicode-punctuation.perl'
NORM_PUNC = MOSES / 'scripts/tokenizer/normalize-punctuation.perl'
REM_NON_PRINT_CHAR = MOSES / 'scripts/tokenizer/remove-non-printing-char.perl'
TOKENIZER = MOSES / 'scripts/tokenizer/tokenizer.perl'
FASTBPE = TOOLS_DIR / 'fastBPE/fast'

NUM_THREADS = 8


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


@dataclass
class Dataset:
    pair: str
    lang: str
    name: str
    url: str

    def __post_init__(self):
        out_dir = DATA_DIR / self.pair
        self.gz_path = out_dir / 'raw' / f'{self.name}.{self.lang}.gz'
        self.raw_path = self.gz_path.with_suffix('')
        self.plain_path = self.raw_path.with_suffix(f'{self.raw_path.suffix}.tok')
        self.bpe_path = out_dir / 'processed' / f'{self.name}.{self.pair}.{self.lang}'
        self.bin_path = self.bpe_path.with_suffix(f'{self.bpe_path.suffix}.pth')

    def take_subset(self, indices, new_bpe_path):
        if not check_exists(new_bpe_path):
            indices = set(indices)
            with self.bpe_path.open('r', encoding='utf8') as fin, new_bpe_path.open('w', encoding='utf8') as fout:
                for i, line in enumerate(fin):
                    if i in indices:
                        fout.write(line)
        self.bpe_path = new_bpe_path
        self.bin_path = self.bpe_path.with_suffix(f'{self.bpe_path.suffix}.pth')


class Datasets:

    def __init__(self, pair):
        self.datasets = dict()
        self.pair = pair

    def add(self, lang, name, url):
        dataset = Dataset(self.pair, lang, name, url)
        key = (name, lang)
        assert key not in self.datasets
        self.datasets[key] = dataset

    def __getitem__(self, key):
        return self.datasets[key]

    def __iter__(self):
        yield from self.datasets.values()

    def merge(self, names: List[str], lang: str, merged_name: str):
        """
        Call this to merge multiple datasets into a new one. Should be called before decompression.
        """
        # Create a new one.
        keys = [(name, lang) for name in names]
        merged_dataset = Dataset(self.pair, lang, merged_name, '')

        # Cat every gz file together.
        all_gz_paths = ' '.join([str(self[key].gz_path) for key in keys])
        subprocess.call(f'cat {all_gz_paths} > {merged_dataset.gz_path}', shell=True)

        # Delete old datasets.
        for key in keys:
            del self.datasets[key]

        # Add the new one.
        self.datasets[(merged_name, lang)] = merged_dataset


def check_exists(path):
    if path.exists():
        logging.info(f'{path} already exists.')
        return True
    return False


if __name__ == "__main__":
    create_logger()

    add_argument('langs', dtype=str, nargs=2)
    add_argument('codes', dtype=str)
    add_argument('seed', dtype=int, default=1234)
    parse_args()

    # Use random seed to make sure train split is persistent.
    random.seed(g.seed)

    # ---------------------------- Initialize datasets ---------------------------- #

    # Use this automatically compute some paths.
    lang1, lang2 = g.langs
    if lang1 > lang2:
        lang1, lang2 = lang2, lang1
    pair = f'{lang1}-{lang2}'
    datasets = Datasets(pair)

    # mkdir everything.
    out_dir = DATA_DIR / f'{pair}'
    (out_dir / 'raw').mkdir(exist_ok=True, parents=True)
    (out_dir / 'processed').mkdir(exist_ok=True, parents=True)

    # --------------------------- Download files first. -------------------------- #

    # For test set, we need to figure out what to download.
    test_to_download = [f'test_{dataset.value}' for dataset in test_sets[lang1] & test_sets[lang2]]

    # Get all paths and urls.
    for lang in g.langs:
        # Training set.
        datasets.add(
            lang,
            'train',
            f'{MULTI30K_DOMAIN_URL}/train.{lang}.gz?raw=true',
        )
        # Dev set.
        datasets.add(
            lang,
            'valid',
            f'{MULTI30K_DOMAIN_URL}/val.{lang}.gz?raw=true'
        )
        # Test set.
        for dataset in test_to_download:
            datasets.add(
                lang,
                dataset,
                f'{MULTI30K_DOMAIN_URL}/{dataset}.{lang}.gz?raw=true'
            )

    # Now try to download everything.
    for dataset in datasets:
        if not check_exists(dataset.gz_path):
            subprocess.call(f'wget {dataset.url} -O {dataset.gz_path}', shell=True)

    # Merge all test sets.
    for lang in g.langs:
        datasets.merge(test_to_download, lang, 'test')

    # --------------------------- Decompress everything -------------------------- #

    for dataset in datasets:
        if not check_exists(dataset.raw_path):
            subprocess.call(f'gunzip -k {dataset.gz_path}', shell=True)
            logging.imp(f'Decompressed file saved in {dataset.raw_path}.')

    # -------------------------------- Preprocess -------------------------------- #

    for dataset in datasets:
        if not check_exists(dataset.plain_path):
            subprocess.call(
                f"cat {dataset.raw_path} | {REPLACE_UNICODE_PUNCT} | {NORM_PUNC} -l {dataset.lang} | {REM_NON_PRINT_CHAR} |{TOKENIZER} -l {dataset.lang} -no-escape -threads {NUM_THREADS} > {dataset.plain_path}", shell=True)
            logging.imp(f'Tokenized file saved in {dataset.plain_path}.')

    # ------------------------------ Apply BPE codes ----------------------------- #

    # First figure out how to deal with '<EMPTY>'.
    empty_out = subprocess.check_output(
        f'{FASTBPE} applybpe_stream {g.codes} < <(echo "<EMPTY>")', shell=True, executable='/bin/bash')  # NOTE Have to use bash for this since process substitution is a bash-only feature.
    empty_out = empty_out.decode('utf8').strip()

    # Now apply BPE to everything.
    for dataset in datasets:
        if not check_exists(dataset.bpe_path):
            subprocess.call(f'{FASTBPE} applybpe {dataset.bpe_path} {dataset.plain_path} {g.codes}', shell=True)
            subprocess.call(f"sed -i 's/{empty_out}/<EMPTY>/g' {dataset.bpe_path}", shell=True)
            logging.imp(f'BPE-segmented file saved in {dataset.bpe_path}.')

    # ---------------------------- Extract vocabulary ---------------------------- #

    # For train sets, we need to take a nonoverlapping subset for lang1 and lang2.
    train1 = datasets[('train', lang1)]
    train2 = datasets[('train', lang2)]
    new_bpe_path1 = DATA_DIR / pair / 'processed' / f'{train1.name}.{lang1}'
    new_bpe_path2 = DATA_DIR / pair / 'processed' / f'{train2.name}.{lang2}'
    len1 = int(subprocess.check_output(f'cat {train1.bpe_path} | wc -l', shell=True))
    len2 = int(subprocess.check_output(f'cat {train2.bpe_path} | wc -l', shell=True))
    if len1 != len2:
        raise RuntimeError(f'{len1} should be equal to {len2}.')
    indices = list(range(len1))
    random.shuffle(indices)
    indices1 = indices[:len1 // 2]
    indices2 = indices[len1 // 2:]
    train1.take_subset(indices1, new_bpe_path1)
    train2.take_subset(indices2, new_bpe_path2)

    # Extract source and target vocabularies.
    train_paths = dict()
    for dataset in [train1, train2]:
        vocab_path = dataset.bpe_path.parent / f'vocab.{dataset.lang}'
        train_paths[dataset.lang] = dataset.bpe_path
        if not check_exists(vocab_path):
            subprocess.call(f'{FASTBPE} getvocab {dataset.bpe_path} > {vocab_path}', shell=True)
            logging.imp(f'Training vocab for {dataset.lang} saved in {vocab_path}.')

    # Extract full vocab.
    full_vocab_path = out_dir / 'processed' / f'vocab.{pair}'
    if not check_exists(full_vocab_path):
        subprocess.call(f'{FASTBPE} getvocab {train_paths[lang1]} {train_paths[lang2]} > {full_vocab_path}', shell=True)
        logging.imp(f'Full vocab saved in {full_vocab_path}.')

    # ------------------------------- Binarize data ------------------------------ #

    for dataset in datasets:
        if not check_exists(dataset.bin_path):
            subprocess.call(f'{MAIN_DIR}/preprocess.py {full_vocab_path} {dataset.bpe_path}', shell=True)
            logging.imp(f'Binarized data saved in {dataset.bin_path}.')

    # -------- Link monolingual validation and test data to parallel data -------- #

    print(datasets.datasets.keys())
    for dataset in datasets:
        if dataset.name != 'train':
            link_path = DATA_DIR / pair / 'processed' / f'{dataset.name}.{dataset.lang}.pth'
            if not check_exists(link_path):
                link_path.symlink_to(dataset.bin_path)
                logging.imp(f'Binarized data linked in {link_path}')
