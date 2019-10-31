# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import random

from arglib import add_argument, add_registry, parse_args
from devlib.named_tensor import patch_named_tensors
from src.data.loader import check_data_params, load_data
from src.evaluation.evaluator import EncDecEvaluator, SingleEvaluator
from src.model import build_model, check_model_params
from src.slurm import init_distributed_mode, init_signal_handler
from src.trainer import EncDecTrainer, SingleTrainer
from src.utils import initialize_exp, set_sampling_probs, shuf_order
from train_cfg import reg


def add_main_arguments():
    """
    Generate a parameters parser.
    """
    # main parameters
    add_argument("dump_path", dtype=str, default="./dumped/",
                 msg="Experiment dump path")
    add_argument("exp_name", dtype=str, default="",
                 msg="Experiment name")
    add_argument("save_periodic_epoch", dtype=int, default=0,
                 msg="Save the model periodically every few epochs (0 to disable)")
    add_argument("save_periodic_step", dtype=int, default=0,
                 msg="Save the model periodically every few steps (0 to disable)")
    add_argument("eval_interval", dtype=int, default=0,
                 msg="evaluate the model every few steps (0 to disable)")
    add_argument("exp_id", dtype=str, default="",
                 msg="Experiment ID")
    add_argument("log_level", dtype=str, default="INFO",
                 msg="log level")

    # float16 / AMP API
    add_argument("fp16", dtype=bool, default=False,
                 msg="Run model with float16")
    add_argument("amp", dtype=int, default=-1,
                 msg="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.")

    # only use an encoder (use a specific decoder for machine translation)
    add_argument("encoder_only", dtype=bool, default=True,
                 msg="Only use an encoder")

    # model parameters
    add_argument("emb_dim", dtype=int, default=512,
                 msg="Embedding layer size")
    add_argument("n_layers", dtype=int, default=4,
                 msg="Number of Transformer layers")
    add_argument("n_heads", dtype=int, default=8,
                 msg="Number of Transformer heads")
    add_argument("dropout", dtype=float, default=0,
                 msg="Dropout")
    add_argument("attention_dropout", dtype=float, default=0,
                 msg="Dropout in the attention layer")
    add_argument("gelu_activation", dtype=bool, default=False,
                 msg="Use a GELU activation instead of ReLU")
    add_argument("share_inout_emb", dtype=bool, default=True,
                 msg="Share input and output embeddings")
    add_argument("sinusoidal_embeddings", dtype=bool, default=False,
                 msg="Use sinusoidal embeddings")
    add_argument("use_lang_emb", dtype=bool, default=True,
                 msg="Use language embedding")
    add_argument("use_graph", dtype=bool, default=False,
                 msg="Use a graph formulation on top of transformer encoder")

    # adaptive softmax
    add_argument("asm", dtype=bool, default=False,
                 msg="Use adaptive softmax")
    add_argument("asm_cutoffs", dtype=str, default="8000,20000",
                 msg="Adaptive softmax cutoffs")
    add_argument("asm_div_value", dtype=float, default=4,
                 msg="Adaptive softmax cluster sizes ratio")

    # causal language modeling task parameters
    add_argument("context_size", dtype=int, default=0,
                 msg="Context size (0 means that the first elements in sequences won't have any context)")

    # masked language modeling task parameters
    add_argument("word_pred", dtype=float, default=0.15,
                 msg="Fraction of words for which we need to make a prediction")
    add_argument("sample_alpha", dtype=float, default=0,
                 msg="Exponent for transforming word counts to probabilities (~word2vec sampling)")
    add_argument("word_mask_keep_rand", dtype=str, default="0.8,0.1,0.1",
                 msg="Fraction of words to mask out / keep / randomize, among the words to predict")

    # input sentence noise
    add_argument("word_shuffle", dtype=float, default=0,
                 msg="Randomly shuffle input words (0 to disable)")
    add_argument("word_dropout", dtype=float, default=0,
                 msg="Randomly dropout input words (0 to disable)")
    add_argument("word_blank", dtype=float, default=0,
                 msg="Randomly blank input words (0 to disable)")

    # data
    add_argument("data_path", dtype='path', default="",
                 msg="Data path")
    add_argument("lgs", dtype=str, default="",
                 msg="Languages (lg1-lg2-lg3 .. ex: en-fr-es-de)")
    add_argument("max_vocab", dtype=int, default=-1,
                 msg="Maximum vocabulary size (-1 to disable)")
    add_argument("min_count", dtype=int, default=0,
                 msg="Minimum vocabulary count")
    add_argument("lg_sampling_factor", dtype=float, default=-1,
                 msg="Language sampling factor")

    # batch parameters
    add_argument("bptt", dtype=int, default=256,
                 msg="Sequence length")
    add_argument("max_len", dtype=int, default=100,
                 msg="Maximum length of sentences (after BPE)")
    add_argument("group_by_size", dtype=bool, default=True,
                 msg="Sort sentences by size during the training")
    add_argument("batch_size", dtype=int, default=32,
                 msg="Number of sentences per batch")
    add_argument("max_batch_size", dtype=int, default=0,
                 msg="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
    add_argument("tokens_per_batch", dtype=int, default=-1,
                 msg="Number of tokens per batch")

    # training parameters
    add_argument("split_data", dtype=bool, default=False,
                 msg="Split data across workers of a same node")
    add_argument("optimizer", dtype=str, default="adam,lr=0.0001",
                 msg="Optimizer (SGD / RMSprop / Adam, etc.)")
    add_argument("clip_grad_norm", dtype=float, default=5,
                 msg="Clip gradients norm (0 to disable)")
    add_argument("epoch_size", dtype=int, default=100000,
                 msg="Epoch size / evaluation frequency (-1 for parallel data size)")
    add_argument("max_epoch", dtype=int, default=100000,
                 msg="Maximum epoch size")
    add_argument("stopping_criterion", dtype=str, default="",
                 msg="Stopping criterion, and number of non-increase before stopping the experiment")
    add_argument("validation_metrics", dtype=str, default="",
                 msg="Validation metrics")
    add_argument("accumulate_gradients", dtype=int, default=1,
                 msg="Accumulate model gradients over N iterations (N times larger batch sizes)")

    # training coefficients
    add_argument("lambda_mlm", dtype=str, default="1",
                 msg="Prediction coefficient (MLM)")
    add_argument("lambda_clm", dtype=str, default="1",
                 msg="Causal coefficient (LM)")
    add_argument("lambda_pc", dtype=str, default="1",
                 msg="PC coefficient")
    add_argument("lambda_ae", dtype=str, default="1",
                 msg="AE coefficient")
    add_argument("lambda_mt", dtype=str, default="1",
                 msg="MT coefficient")
    add_argument("lambda_bt", dtype=str, default="1",
                 msg="BT coefficient")
    add_argument("lambda_ep", dtype=str, default="1",
                 msg="EP coefficient")

    # training steps
    add_argument("clm_steps", dtype=str, default="",
                 msg="Causal prediction steps (CLM)")
    add_argument("mlm_steps", dtype=str, default="",
                 msg="Masked prediction steps (MLM / TLM)")
    add_argument("mt_steps", dtype=str, default="",
                 msg="Machine translation steps")
    add_argument("ae_steps", dtype=str, default="",
                 msg="Denoising auto-encoder steps")
    add_argument("bt_steps", dtype=str, default="",
                 msg="Back-translation steps")
    add_argument("ep_steps", dtype=str, default="",
                 msg="EAT-plain reconstruction steps")
    add_argument("pc_steps", dtype=str, default="",
                 msg="Parallel classification steps")

    # reload pretrained embeddings / pretrained model / checkpoint
    add_argument("reload_emb", dtype=str, default="",
                 msg="Reload pretrained word embeddings")
    add_argument("reload_model", dtype=str, default="",
                 msg="Reload a pretrained model")
    add_argument("reload_checkpoint", dtype=str, default="",
                 msg="Reload a checkpoint")

    # beam search (for MT only)
    add_argument("beam_size", dtype=int, default=1,
                 msg="Beam size, default = 1 (greedy decoding)")
    add_argument("length_penalty", dtype=float, default=1,
                 msg="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")
    add_argument("early_stopping", dtype=bool, default=False,
                 msg="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.")

    # evaluation
    add_argument("eval_bleu", dtype=bool, default=False,
                 msg="Evaluate BLEU score during MT training")
    add_argument("eval_only", dtype=bool, default=False,
                 msg="Only run evaluations")

    # debug
    add_argument("debug_train", dtype=bool, default=False,
                 msg="Use valid sets for train sets (faster loading)")
    add_argument("debug_slurm", dtype=bool, default=False,
                 msg="Debug multi-GPU / multi-node within a SLURM job")
    add_argument("debug", msg="Enable all debug flags",
                 dtype=bool, default=False)

    # multi-gpu / multi-node
    add_argument("local_rank", dtype=int, default=-1,
                 msg="Multi-GPU - Local rank")
    add_argument("master_port", dtype=int, default=-1,
                 msg="Master port (for multi-node SLURM jobs)")

    # Add registry.
    add_registry(reg)


def main(params):

    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)

    # initialize the experiment
    logger = initialize_exp(params)

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()

    # load data
    data = load_data(params)

    # build model
    if params.encoder_only:
        model = build_model(params, data['dico'])
    else:
        encoder, decoder = build_model(params, data['dico'])

    # build trainer, reload potential checkpoints / build evaluator
    if params.encoder_only:
        trainer = SingleTrainer(model, data, params)
        evaluator = SingleEvaluator(trainer, data, params)
    else:
        trainer = EncDecTrainer(encoder, decoder, data, params)
        evaluator = EncDecEvaluator(trainer, data, params)

    # evaluation
    if params.eval_only:
        scores = evaluator.run_all_evals(trainer)
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        exit()

    # set sampling probabilities for training
    set_sampling_probs(data, params)

    # language model training
    for _ in range(params.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % trainer.tracker.epoch)

        trainer.n_sentences = 0

        while trainer.n_sentences < trainer.epoch_size:

            # CLM steps
            for lang1, lang2 in shuf_order(params.clm_steps, params):
                trainer.clm_step(lang1, lang2, params.lambda_clm)

            # MLM steps (also includes TLM if lang2 is not None)
            for lang1, lang2 in shuf_order(params.mlm_steps, params):
                trainer.mlm_step(lang1, lang2, params.lambda_mlm)

            # parallel classification steps
            for lang1, lang2 in shuf_order(params.pc_steps, params):
                trainer.pc_step(lang1, lang2, params.lambda_pc)

            # denoising auto-encoder steps
            for lang in shuf_order(params.ae_steps):
                trainer.denoise_mt_step(lang, lang, params.lambda_ae)

            # eat-plain steps, with optional added noise
            # NOTE(j_luo) This is not used.
            for lang in shuf_order(params.ep_steps):
                trainer.ep_step(lang, params.lambda_ep)

            # machine translation steps
            for lang1, lang2 in shuf_order(params.mt_steps, params):
                trainer.mt_step(lang1, lang2, params.lambda_mt)

            # back-translation steps
            for lang1, lang2, lang3 in shuf_order(params.bt_steps):
                trainer.bt_step(lang1, lang2, lang3, params.lambda_bt)

            trainer.iter()

            if params.eval_interval > 0 and trainer.tracker.n_total_iter % params.eval_interval == 0:
                # evaluate perplexity
                scores = evaluator.run_all_evals(trainer)
                # end of an evaluation interval.
                trainer.save_best_model(scores)
                trainer.save_periodic()
                trainer.end_interval(scores)

        logger.info("============ End of epoch %i ============" % trainer.tracker.epoch)

        # print / JSON log
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        if params.is_master:
            logger.info("__log__:%s" % json.dumps(scores))

        trainer.tracker.update('epoch')


if __name__ == '__main__':
    patch_named_tensors()

    # generate parser / parse parameters
    add_main_arguments()
    params = parse_args().as_namespace()

    # debug mode
    if params.debug:
        params.exp_name = 'debug'
        params.exp_id = 'debug_%08i' % random.randint(0, 100000000)
        params.debug_slurm = True
        params.debug_train = True

    # check parameters
    check_data_params(params)
    check_model_params(params)

    # run experiment
    main(params)
