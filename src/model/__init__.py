# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger

import torch

from arglib import add_argument, g

from .memory import HashingMemory
from .pretrain import load_embeddings
from .transformer import (DECODER_ONLY_PARAMS,  # , TRANSFORMER_LAYER_PARAMS
                          TransformerModel)

logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    # masked language modeling task parameters
    assert params.bptt >= 1
    assert 0 <= params.word_pred < 1
    assert 0 <= params.sample_alpha < 1
    s = params.word_mask_keep_rand.split(',')
    assert len(s) == 3
    s = [float(x) for x in s]
    assert all([0 <= x <= 1 for x in s]) and sum(s) == 1
    params.word_mask = s[0]
    params.word_keep = s[1]
    params.word_rand = s[2]

    # input sentence noise for DAE
    if len(params.ae_steps) == 0:
        assert params.word_shuffle == 0
        assert params.word_dropout == 0
        assert params.word_blank == 0
    else:
        assert params.word_shuffle == 0 or params.word_shuffle > 1
        assert 0 <= params.word_dropout < 1
        assert 0 <= params.word_blank < 1

    # model dimensions
    assert params.emb_dim % params.n_heads == 0

    # share input and output embeddings
    assert params.share_inout_emb is False or params.asm is False

    # adaptive softmax
    if params.asm:
        assert params.asm_div_value > 1
        s = params.asm_cutoffs.split(',')
        assert all([x.isdigit() for x in s])
        params.asm_cutoffs = [int(x) for x in s]
        assert params.max_vocab == -1 or params.asm_cutoffs[-1] < params.max_vocab

    # memory
    if params.use_memory:
        HashingMemory.check_params(params)
        s_enc = [x for x in params.mem_enc_positions.split(',') if x != '']
        s_dec = [x for x in params.mem_dec_positions.split(',') if x != '']
        assert len(s_enc) == len(set(s_enc))
        assert len(s_dec) == len(set(s_dec))
        assert all(x.isdigit() or x[-1] == '+' and x[:-1].isdigit() for x in s_enc)
        assert all(x.isdigit() or x[-1] == '+' and x[:-1].isdigit() for x in s_dec)
        params.mem_enc_positions = [(int(x[:-1]), 'after') if x[-1] == '+' else (int(x), 'in') for x in s_enc]
        params.mem_dec_positions = [(int(x[:-1]), 'after') if x[-1] == '+' else (int(x), 'in') for x in s_dec]
        assert len(params.mem_enc_positions) + len(params.mem_dec_positions) > 0
        assert len(params.mem_enc_positions) == 0 or 0 <= min([x[0] for x in params.mem_enc_positions]) <= max(
            [x[0] for x in params.mem_enc_positions]) <= params.n_layers - 1
        assert len(params.mem_dec_positions) == 0 or 0 <= min([x[0] for x in params.mem_dec_positions]) <= max(
            [x[0] for x in params.mem_dec_positions]) <= params.n_layers - 1

    # reload pretrained word embeddings
    if params.reload_emb != '':
        assert os.path.isfile(params.reload_emb)

    # reload a pretrained model
    if params.reload_model != '':
        if params.encoder_only:
            assert os.path.isfile(params.reload_model)
        else:
            s = params.reload_model.split(',')
            assert len(s) == 2
            assert all([x == '' or os.path.isfile(x) for x in s])


def set_pretrain_emb(model, dico, word2id, embeddings):
    """
    Pretrain word embeddings.
    """
    n_found = 0
    with torch.no_grad():
        for i in range(len(dico)):
            idx = word2id.get(dico[i], None)
            if idx is None:
                continue
            n_found += 1
            model.embeddings.weight[i] = embeddings[idx].cuda()
            model.pred_layer.proj.weight[i] = embeddings[idx].cuda()
    logger.info("Pretrained %i/%i words (%.3f%%)."
                % (n_found, len(dico), 100. * n_found / len(dico)))


add_argument('old_data_paths', nargs=2, dtype=str,
             msg='paths (first src, second tgt) to the old data to extract the vocabularies.')
add_argument('hack_reload', dtype=bool, default=False)


def build_model(params, dico):
    """
    Build model.
    """

    if params.encoder_only:
        # build
        model = TransformerModel(params, dico, is_encoder=True, with_output=True)

        # reload pretrained word embeddings
        if params.reload_emb != '':
            word2id, embeddings = load_embeddings(params.reload_emb, params)
            set_pretrain_emb(model, dico, word2id, embeddings)

        # reload a pretrained model
        if params.reload_model != '':
            logger.info("Reloading model from %s ..." % params.reload_model)
            reloaded = torch.load(params.reload_model, map_location=lambda storage,
                                  loc: storage.cuda(params.local_rank))['model']

            if all([k.startswith('module.') for k in reloaded.keys()]):
                reloaded = {k[len('module.'):]: v for k, v in reloaded.items()}

            # # HACK to reload models with less layers
            # for i in range(12, 24):
            #     for k in TRANSFORMER_LAYER_PARAMS:
            #         k = k % i
            #         if k in model.state_dict() and k not in reloaded:
            #             logger.warning("Parameter %s not found. Ignoring ..." % k)
            #             reloaded[k] = model.state_dict()[k]

            model.load_state_dict(reloaded)

        logger.info("Model: {}".format(model))
        logger.info("Number of parameters (model): %i" %
                    sum([p.numel() for p in model.parameters() if p.requires_grad]))

        return model.cuda()

    else:
        # build
        # TODO: only output when necessary - len(params.clm_steps + params.mlm_steps) > 0
        # NOTE(j_luo) set with_output to False, see https://github.com/facebookresearch/XLM/issues/109
        encoder = TransformerModel(params, dico, is_encoder=True, with_output=False)
        decoder = TransformerModel(params, dico, is_encoder=False, with_output=True)

        # reload pretrained word embeddings
        if params.reload_emb != '':
            word2id, embeddings = load_embeddings(params.reload_emb, params)
            set_pretrain_emb(encoder, dico, word2id, embeddings)
            set_pretrain_emb(decoder, dico, word2id, embeddings)

        # reload a pretrained model
        if params.reload_model != '':
            enc_path, dec_path = params.reload_model.split(',')
            assert not (enc_path == '' and dec_path == '')

            if g.old_data_paths:
                evp, dvp = g.old_data_paths
                ev = torch.load(evp)['dico']
                dv = torch.load(dvp)['dico']
                nevp = params.mono_dataset[params.langs[0]]['valid']
                ndvp = params.mono_dataset[params.langs[1]]['valid']
                nev = torch.load(nevp)['dico']
                ndv = torch.load(ndvp)['dico']

            def remap_vocab_ids(old, new, old_state, name, new_w):
                old_w = old_state[name]
                assert len(old) == len(old.word2id) == len(old.id2word) == len(old_w)
                assert len(new) == len(new.word2id) == len(new.id2word) == len(new_w)
                idx = list()
                for new_i, w in new.id2word.items():
                    if w == '<EMPTY>':
                        old_i = old.unk_index
                    else:
                        old_i = old.word2id[w]
                    idx.append(old_i)
                old_state[name] = old_w[idx]

            # reload encoder
            if enc_path != '':
                logger.info("Reloading encoder from %s ..." % enc_path)
                if g.hack_reload:
                    enc_reload = torch.load(enc_path)
                else:
                    enc_reload = torch.load(enc_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                enc_reload = enc_reload['model' if 'model' in enc_reload else 'encoder']
                if all([k.startswith('module.') for k in enc_reload.keys()]):
                    enc_reload = {k[len('module.'):]: v for k, v in enc_reload.items()}
                # NOTE(j_luo) set strict to False, see https://github.com/facebookresearch/XLM/issues/109
                if g.old_data_paths:
                    remap_vocab_ids(ev, nev, enc_reload, 'embeddings.weight', encoder.embeddings.weight)
                encoder.load_state_dict(enc_reload, strict=False)

            # reload decoder
            if dec_path != '':
                logger.info("Reloading decoder from %s ..." % dec_path)
                if g.hack_reload:
                    dec_reload = torch.load(dec_path)
                else:
                    dec_reload = torch.load(dec_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                dec_reload = dec_reload['model' if 'model' in dec_reload else 'decoder']
                if all([k.startswith('module.') for k in dec_reload.keys()]):
                    dec_reload = {k[len('module.'):]: v for k, v in dec_reload.items()}
                for i in range(params.n_layers):
                    for name in DECODER_ONLY_PARAMS:
                        if name % i not in dec_reload:
                            logger.warning("Parameter %s not found." % (name % i))
                            dec_reload[name % i] = decoder.state_dict()[name % i]
                if g.old_data_paths:
                    remap_vocab_ids(dv, ndv, dec_reload, 'embeddings.weight', decoder.embeddings.weight)
                    remap_vocab_ids(dv, ndv, dec_reload, 'pred_layer.proj.weight', decoder.pred_layer.proj.weight)
                    remap_vocab_ids(dv, ndv, dec_reload, 'pred_layer.proj.bias', decoder.pred_layer.proj.bias)
                decoder.load_state_dict(dec_reload, strict=False)

        logger.debug("Encoder: {}".format(encoder))
        logger.debug("Decoder: {}".format(decoder))
        logger.info("Number of parameters (encoder): %i" %
                    sum([p.numel() for p in encoder.parameters() if p.requires_grad]))
        logger.info("Number of parameters (decoder): %i" %
                    sum([p.numel() for p in decoder.parameters() if p.requires_grad]))

        return encoder.cuda(), decoder.cuda()
