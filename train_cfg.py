from dataclasses import dataclass

from arglib import Registry

reg = Registry('train')


@dataclass
class ParamsFromXLM:
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
        src = cls.src_lang
        tgt = cls.tgt_lang
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
    src_lang = 'en'
    tgt_lang = 'fr'
    reload_model = f'save/mlm_enfr_1024.pth,save/mlm_enfr_1024.pth'


@reg
class DeEnBase(SingleGpuParams):
    src_lang = 'de'
    tgt_lang = 'en'
    reload_model = f'save/mlm_ende_1024.pth,save/mlm_ende_1024.pth'
