from dataclasses import dataclass
from typing import Tuple

from arglib import Registry

reg = Registry('train_cfg')


@dataclass
class ParamsFromXLM:
    # NOTE(j_luo) These are class variables.
    SRC_LANG = ''
    TGT_LANG = ''

    # NOTE(j_luo) These are type-annotated therefore will be picked up by arglib.
    attention_dropout: float = 0.1
    batch_size: int = 32
    bptt: int = 256
    dropout: float = 0.1
    dump_path: str = "./dumped/"
    emb_dim: int = 1024
    encoder_only: bool = False
    epoch_size: int = 200000
    eval_bleu: bool = True
    gelu_activation: bool = True
    lambda_ae: str = '0:1,100000:0.1,300000:0'
    n_heads: int = 8
    n_layers: int = 6
    optimizer: str = "adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001"
    tokens_per_batch: int = 2000
    word_blank: float = 0.1
    word_dropout: float = 0.1
    word_shuffle: int = 3

    def __post_init__(self):
        cls = type(self)
        src = cls.SRC_LANG
        tgt = cls.TGT_LANG
        self.ae_steps = f'{src},{tgt}'
        self.bt_steps = f'{src}-{tgt}-{src},{tgt}-{src}-{tgt}'
        self.lgs = f'{src}-{tgt}'
        self.stopping_criterion = f'valid_{src}-{tgt}_mt_bleu,10'
        self.validation_metrics = f'valid_{src}-{tgt}_mt_bleu'
        self.data_path = f"./data/processed/{src}-{tgt}/"
        self.exp_name = f"unsupMT_{src}{tgt}"


@dataclass
class SingleGpuParams(ParamsFromXLM):
    tokens_per_batch: int = 1000


@reg
class EnFrBase(SingleGpuParams):
    SRC_LANG = 'en'
    TGT_LANG = 'fr'
    reload_model: str = f'save/mlm_enfr_1024.pth,save/mlm_enfr_1024.pth'


@reg
class DeEnBase(SingleGpuParams):
    SRC_LANG = 'de'
    TGT_LANG = 'en'
    reload_model: str = f'save/mlm_ende_1024.pth,save/mlm_ende_1024.pth'


@reg
class DeEnMulti30KBaseline(DeEnBase):

    eval_interval: int = 100
    old_data_paths: Tuple[str] = ('data/processed/de-en/valid.de.pth', 'data/processed/de-en/valid.en.pth')

    def __post_init__(self):
        super().__post_init__()
        cls = type(self)
        src = cls.SRC_LANG
        tgt = cls.TGT_LANG
        self.data_path = f'./data/multi30k/{src}-{tgt}/processed'
        self.exp_name = f'unsupMT_multi30K_{src}{tgt}'


@reg
class DeEnMulti30KBaselineNoBt(DeEnMulti30KBaseline):

    def __post_init__(self):
        super().__post_init__()
        cls = type(self)
        src = cls.SRC_LANG
        tgt = cls.TGT_LANG
        self.bt_steps = ''
        self.eval_mt_steps = f'{src}-{tgt},{tgt}-{src}'


@reg
class DeEnMulti30KEatNoBt(DeEnMulti30KBaselineNoBt):

    def __post_init__(self):
        super().__post_init__()
        cls = type(self)
        src = cls.SRC_LANG
        tgt = cls.TGT_LANG
        self.data_path = f"./data/multi30k/{src}-{tgt}/processed-eat"
        self.exp_name = f"unsupMT_multi30K_EAT_{src}{tgt}"
