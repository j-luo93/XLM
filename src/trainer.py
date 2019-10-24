# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import os
import time
from collections import OrderedDict
from logging import getLogger

import apex
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from arglib import add_argument, init_g_attr
from trainlib import Metric, Metrics, Tracker, log_this

from .data.verifier import Verifier
from .model.memory import HashingMemory
from .model.transformer import TransformerFFN
from .optim import get_optimizer
from .utils import (concat_batches, find_modules, parse_lambda_config, to_cuda,
                    update_lambdas)

logger = getLogger()


@init_g_attr(default="none")
class Trainer:

    add_argument('freeze_emb', dtype=bool, default=False, msg='Freeze embeddings.')
    add_argument('ae_add_noise', dtype=bool, default=True, msg='add noise to ae')
    add_argument('ep_add_noise', default=False, dtype=bool)
    add_argument('supervised_graph', dtype=str, default='', msg='supervise the process of graph induction')
    add_argument('lambda_graph', dtype=float, default=0.0, msg='hyperparameter for graph induction loss')

    def __init__(self, data, params, use_graph: 'p'):
        """
        Initialize trainer.
        """
        # epoch / iteration size
        self.epoch_size = params.epoch_size
        if self.epoch_size == -1:
            self.epoch_size = self.data
            assert self.epoch_size > 0

        # data iterators
        self.iterators = {}

        # list memory components
        self.memory_list = []
        self.ffn_list = []
        for name in self.MODEL_NAMES:
            find_modules(getattr(self, name), f'self.{name}', HashingMemory, self.memory_list)
            find_modules(getattr(self, name), f'self.{name}', TransformerFFN, self.ffn_list)
        logger.info("Found %i memories." % len(self.memory_list))
        logger.info("Found %i FFN." % len(self.ffn_list))

        # set parameters
        self.set_parameters()

        # float16 / distributed (no AMP)
        assert params.amp >= 1 or not params.fp16
        assert params.amp >= 0 or params.accumulate_gradients == 1
        if params.multi_gpu and params.amp == -1:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            for name in self.MODEL_NAMES:
                setattr(self, name, nn.parallel.DistributedDataParallel(getattr(self, name), device_ids=[
                        params.local_rank], output_device=params.local_rank, broadcast_buffers=True))

        # set optimizers
        self.set_optimizers()

        # float16 / distributed (AMP)
        if params.amp >= 0:
            self.init_amp()
            if params.multi_gpu:
                logger.info("Using apex.parallel.DistributedDataParallel ...")
                for name in self.MODEL_NAMES:
                    setattr(self, name, apex.parallel.DistributedDataParallel(
                        getattr(self, name), delay_allreduce=True))

        # stopping criterion used for early stopping
        if params.stopping_criterion != '':
            split = params.stopping_criterion.split(',')
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == '_':
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # probability of masking out / randomize / not modify words to predict
        params.pred_probs = torch.FloatTensor([params.word_mask, params.word_keep, params.word_rand])

        # probabilty to predict a word
        counts = np.array(list(self.data['dico'].counts.values()))
        params.mask_scores = np.maximum(counts, 1) ** -params.sample_alpha
        params.mask_scores[params.pad_index] = 0  # do not predict <PAD> index
        params.mask_scores[counts == 0] = 0       # do not predict special tokens

        # validation metrics
        self.eval_metrics = []
        metrics = [m for m in params.validation_metrics.split(',') if m != '']
        for m in metrics:
            m = (m[1:], False) if m[0] == '_' else (m, True)
            self.eval_metrics.append(m)
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.eval_metrics}

        # training statistics
        self.tracker = Tracker()
        self.tracker.add_track('epoch')
        self.tracker.add_track('n_iter')
        self.tracker.add_track('n_total_iter')
        self.tracker.add_track('n_sentences')
        self.tracker.add_update_fn('epoch', 'add')
        self.tracker.add_update_fn('n_iter', 'add')
        self.tracker.add_update_fn('n_total_iter', 'add')
        self.tracker.add_update_fn('n_sentences', 'addx')
        self.metrics = Metrics()
        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()

        # initialize lambda coefficients and their configurations
        parse_lambda_config(params)

        # Get a verifier if use_graph.
        if params.use_graph:
            self.verifier = Verifier()

        # Whether to add noise to AE.
        self.ae_add_noise = params.ae_add_noise

        self.supervised_graph = set()
        if params.supervised_graph:
            self.supervised_graph.update(params.supervised_graph.split(','))
            if not self.supervised_graph <= set(params.lgs.split('-')):
                raise ValueError('You can only supervise the graph inductive for the given languages.')
        if self.supervised_graph and not self.use_graph:
            raise ValueError(f'Must use graph in order to use supervision for graph.')

    def set_parameters(self):
        """
        Set parameters.
        """
        params = self.params
        self.parameters = {}
        named_params = []

        # Freeze embeddings if necessary.
        if params.freeze_emb:
            for name in self.MODEL_NAMES:
                p = getattr(self, name).embeddings.weight
                p.requires_grad = False

        # Obtain trainable parameters.
        for name in self.MODEL_NAMES:
            named_params.extend([(k, p) for k, p in getattr(self, name).named_parameters() if p.requires_grad])

        # model (excluding memory values)
        self.parameters['model'] = [p for k, p in named_params if not k.endswith(HashingMemory.MEM_VALUES_PARAMS)]

        # memory values
        if params.use_memory:
            self.parameters['memory'] = [p for k, p in named_params if k.endswith(HashingMemory.MEM_VALUES_PARAMS)]
            assert len(self.parameters['memory']) == len(params.mem_enc_positions) + len(params.mem_dec_positions)

        # log
        for k, v in self.parameters.items():
            total = sum([p.nelement() for p in v])
            logger.info("Found %i trainable parameters in %s, total size %d." % (len(v), k, total))
            assert len(v) >= 1

    def set_optimizers(self):
        """
        Set optimizers.
        """
        params = self.params
        self.optimizers = {}

        # model optimizer (excluding memory values)
        self.optimizers['model'] = get_optimizer(self.parameters['model'], params.optimizer)

        # memory values optimizer
        if params.use_memory:
            self.optimizers['memory'] = get_optimizer(self.parameters['memory'], params.mem_values_optimizer)

        # log
        logger.info("Optimizers: %s" % ", ".join(self.optimizers.keys()))

    def init_amp(self):
        """
        Initialize AMP optimizer.
        """
        params = self.params
        assert params.amp == 0 and params.fp16 is False or params.amp in [1, 2, 3] and params.fp16 is True
        opt_names = self.optimizers.keys()
        models = [getattr(self, name) for name in self.MODEL_NAMES]
        models, optimizers = apex.amp.initialize(
            models,
            [self.optimizers[k] for k in opt_names],
            opt_level=('O%i' % params.amp)
        )
        for name, model in zip(self.MODEL_NAMES, models):
            setattr(self, name, model)
        self.optimizers = {
            opt_name: optimizer
            for opt_name, optimizer in zip(opt_names, optimizers)
        }

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")
            # exit()

        params = self.params

        # optimizers
        names = self.optimizers.keys()
        optimizers = [self.optimizers[k] for k in names]

        # regular optimization
        if params.amp == -1:
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            if params.clip_grad_norm > 0:
                for name in names:
                    # norm_check_a = (sum([p.grad.norm(p=2).item() ** 2 for p in self.parameters[name]])) ** 0.5
                    clip_grad_norm_(self.parameters[name], params.clip_grad_norm)
                    # norm_check_b = (sum([p.grad.norm(p=2).item() ** 2 for p in self.parameters[name]])) ** 0.5
                    # print(name, norm_check_a, norm_check_b)
            for optimizer in optimizers:
                optimizer.step()

        # AMP optimization
        else:
            if self.tracker.n_iter % params.accumulate_gradients == 0:
                with apex.amp.scale_loss(loss, optimizers) as scaled_loss:
                    scaled_loss.backward()
                if params.clip_grad_norm > 0:
                    for name in names:
                        # norm_check_a = (sum([p.grad.norm(p=2).item() ** 2 for p in apex.amp.master_params(self.optimizers[name])])) ** 0.5
                        clip_grad_norm_(apex.amp.master_params(self.optimizers[name]), params.clip_grad_norm)
                        # norm_check_b = (sum([p.grad.norm(p=2).item() ** 2 for p in apex.amp.master_params(self.optimizers[name])])) ** 0.5
                        # print(name, norm_check_a, norm_check_b)
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                with apex.amp.scale_loss(loss, optimizers, delay_unscale=True) as scaled_loss:
                    scaled_loss.backward()

    def iter(self):
        """
        End of iteration.
        """
        self.tracker.update('n_iter')
        self.tracker.update('n_total_iter')
        update_lambdas(self.params, self.tracker.n_total_iter)
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.tracker.n_total_iter % 5 != 0:
            return

        s_iter = "%7i - " % self.tracker.n_total_iter
        s_stat = ' || '.join([
            '{}: {:7.4f}'.format(name, metric.mean) for name, metric in self.metrics.items()
            if metric.report_mean and metric.mean > 0
        ])

        # learning rates
        s_lr = " - "
        for k, v in self.optimizers.items():
            s_lr = s_lr + (" - %s LR: " % k) + " / ".join("{:.4e}".format(group['lr']) for group in v.param_groups)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} sent/s - {:8.2f} words/s - ".format(
            self.metrics.processed_s.total * 1.0 / diff,
            self.metrics.processed_w.total * 1.0 / diff
        )
        self.last_time = new_time
        self.metrics.clear()

        # log speed + metrics + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)

    def get_iterator(self, iter_name, lang1, lang2, stream, return_indices=False):
        """
        Create a new iterator for a dataset.
        """
        logger.info("Creating new training data iterator (%s) ..." %
                    ','.join([str(x) for x in [iter_name, lang1, lang2] if x is not None]))
        assert stream or not self.params.use_memory or not self.params.mem_query_batchnorm
        if lang2 is None:
            if stream:
                iterator = self.data['mono_stream'][lang1]['train'].get_iterator(
                    shuffle=True, return_indices=return_indices)
            else:
                iterator = self.data['mono'][lang1]['train'].get_iterator(
                    shuffle=True,
                    group_by_size=self.params.group_by_size,
                    n_sentences=-1,
                    return_indices=return_indices
                )
        else:
            assert stream is False
            _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            iterator = self.data['para'][(_lang1, _lang2)]['train'].get_iterator(
                shuffle=True,
                group_by_size=self.params.group_by_size,
                n_sentences=-1,
                return_indices=return_indices
            )

        self.iterators[(iter_name, lang1, lang2)] = iterator
        return iterator

    def get_batch(self, iter_name, lang1, lang2=None, stream=False, input_format='plain', return_indices=False):
        """
        Return a batch of sentences from a dataset.
        """
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        assert stream is False or lang2 is None
        iterator = self.iterators.get((iter_name, lang1, lang2), None)
        if iterator is None:
            iterator = self.get_iterator(iter_name, lang1, lang2, stream, return_indices=return_indices)
        try:
            x = next(iterator)
        except StopIteration:
            iterator = self.get_iterator(iter_name, lang1, lang2, stream, return_indices=return_indices)
            x = next(iterator)
        if lang2 is None or lang1 < lang2:
            if return_indices:
                (data, lengths), indices = x
            else:
                data, lengths = x
            if input_format == 'eat':
                assert len(data.shape) == 2
                sl, bs = data.shape
                assert sl % 9 == 0
                unpacked = data[sl // 9, 9, bs]
                data = unpacked[:, :3].view(-1, bs)
                lengths = lengths // 3
            return (data, lengths, indices) if return_indices else (data, lengths)
        else:
            return x[::-1]  # NOTE Not sure why this is reversed but I'm keeping it as it is.

    def word_shuffle(self, x, l):
        """
        Randomly shuffle input words.
        """
        if self.params.word_shuffle == 0:
            return x, l, None

        # define noise word scores
        noise = np.random.uniform(0, self.params.word_shuffle, size=(x.size(0) - 1, x.size(1)))
        noise[0] = -1  # do not move start sentence symbol

        assert self.params.word_shuffle > 1
        x2 = x.clone()
        permutations = list()
        for i in range(l.size(0)):
            # generate a random permutation
            scores = np.arange(l[i] - 1) + noise[:l[i] - 1, i]
            permutation = scores.argsort()
            permutations.append(permutation)
            # shuffle words
            x2[:l[i] - 1, i].copy_(x2[:l[i] - 1, i][torch.from_numpy(permutation)])
        return x2, l, permutations

    def word_dropout(self, x, l):
        """
        Randomly drop input words.
        """
        if self.params.word_dropout == 0:
            return x, l, None
        assert 0 < self.params.word_dropout < 1

        # define words to drop
        eos = self.params.eos_index
        assert (x[0] == eos).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_dropout
        keep[0] = 1  # do not drop the start sentence symbol

        sentences = []
        lengths = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == eos
            words = x[:l[i] - 1, i].tolist()
            # randomly drop words from the input
            new_s = [w for j, w in enumerate(words) if keep[j, i]]
            # we need to have at least one word in the sentence (more than the start / end sentence symbols)
            if len(new_s) == 1:
                to_add = np.random.randint(1, len(words))
                assert not keep[to_add, i]
                keep[to_add, i] = 1
                new_s.append(words[to_add])
            new_s.append(eos)
            assert len(new_s) >= 3 and new_s[0] == eos and new_s[-1] == eos
            sentences.append(new_s)
            lengths.append(len(new_s))
        # re-construct input
        l2 = torch.LongTensor(lengths)
        x2 = torch.LongTensor(l2.max(), l2.size(0)).fill_(self.params.pad_index)
        for i in range(l2.size(0)):
            x2[:l2[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l2, keep

    def word_blank(self, x, l):
        """
        Randomly blank input words.
        """
        if self.params.word_blank == 0:
            return x, l
        assert 0 < self.params.word_blank < 1

        # define words to blank
        eos = self.params.eos_index
        assert (x[0] == eos).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_blank
        keep[0] = 1  # do not blank the start sentence symbol

        sentences = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == eos
            words = x[:l[i] - 1, i].tolist()
            # randomly blank words from the input
            new_s = [w if keep[j, i] else self.params.mask_index for j, w in enumerate(words)]
            new_s.append(eos)
            assert len(new_s) == l[i] and new_s[0] == eos and new_s[-1] == eos
            sentences.append(new_s)
        # re-construct input
        x2 = torch.LongTensor(l.max(), l.size(0)).fill_(self.params.pad_index)
        for i in range(l.size(0)):
            x2[:l[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l

    def add_noise(self, words, lengths):
        """
        Add noise to the encoder input.
        """
        permutations, keep = None, None
        if self.ae_add_noise:
            words, lengths, permutations = self.word_shuffle(words, lengths)
            words, lengths, keep = self.word_dropout(words, lengths)
            # NOTE(j_luo) Masking words shouldn't affet graph_target.
            words, lengths = self.word_blank(words, lengths)
        return words, lengths, permutations, keep

    def mask_out(self, x, lengths):
        """
        Decide of random words to mask out, and what target they get assigned.
        """
        params = self.params
        slen, bs = x.size()

        # define target words to predict
        if params.sample_alpha == 0:
            pred_mask = np.random.rand(slen, bs) <= params.word_pred
            pred_mask = torch.from_numpy(pred_mask.astype(np.uint8))
        else:
            x_prob = params.mask_scores[x.flatten()]
            n_tgt = math.ceil(params.word_pred * slen * bs)
            tgt_ids = np.random.choice(len(x_prob), n_tgt, replace=False, p=x_prob / x_prob.sum())
            pred_mask = torch.zeros(slen * bs, dtype=torch.uint8)
            pred_mask[tgt_ids] = 1
            pred_mask = pred_mask.view(slen, bs)

        # do not predict padding
        pred_mask[x == params.pad_index] = 0
        pred_mask[0] = 0  # TODO: remove

        # mask a number of words == 0 [8] (faster with fp16)
        if params.fp16:
            pred_mask = pred_mask.view(-1)
            n1 = pred_mask.sum().item()
            n2 = max(n1 % 8, 8 * (n1 // 8))
            if n2 != n1:
                pred_mask[torch.nonzero(pred_mask).view(-1)[:n1 - n2]] = 0
            pred_mask = pred_mask.view(slen, bs)
            assert pred_mask.sum().item() % 8 == 0

        # generate possible targets / update x input
        _x_real = x[pred_mask]
        _x_rand = _x_real.clone().random_(params.n_words)
        _x_mask = _x_real.clone().fill_(params.mask_index)
        probs = torch.multinomial(params.pred_probs, len(_x_real), replacement=True)
        _x = _x_mask * (probs == 0).long() + _x_real * (probs == 1).long() + _x_rand * (probs == 2).long()
        x = x.masked_scatter(pred_mask, _x)

        assert 0 <= x.min() <= x.max() < params.n_words
        assert x.size() == (slen, bs)
        assert pred_mask.size() == (slen, bs)

        return x, _x_real, pred_mask

    def generate_batch(self, lang1, lang2, name):
        """
        Prepare a batch (for causal or non-causal mode).
        """
        params = self.params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None

        if lang2 is None:
            x, lengths = self.get_batch(name, lang1, stream=True)
            positions = None
            langs = x.clone().fill_(lang1_id) if params.n_langs > 1 else None
        elif lang1 == lang2:
            (x1, len1) = self.get_batch(name, lang1)
            (x2, len2) = (x1, len1)
            (x1, len1) = self.add_noise(x1, len1)
            x, lengths, positions, langs = concat_batches(
                x1, len1, lang1_id, x2, len2, lang2_id, params.pad_index, params.eos_index, reset_positions=False)
        else:
            (x1, len1), (x2, len2) = self.get_batch(name, lang1, lang2)
            x, lengths, positions, langs = concat_batches(
                x1, len1, lang1_id, x2, len2, lang2_id, params.pad_index, params.eos_index, reset_positions=True)

        return x, lengths, positions, langs, (None, None) if lang2 is None else (len1, len2)

    def save_checkpoint(self, name, include_optimizers=True):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return

        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            'epoch': self.tracker.epoch,
            'n_total_iter': self.tracker.n_total_iter,
            'best_metrics': self.best_metrics,
            'best_stopping_criterion': self.best_stopping_criterion,
        }

        for name in self.MODEL_NAMES:
            logger.warning(f"Saving {name} parameters ...")
            data[name] = getattr(self, name).state_dict()

        if include_optimizers:
            for name in self.optimizers.keys():
                logger.warning(f"Saving {name} optimizer ...")
                data[f'{name}_optimizer'] = self.optimizers[name].state_dict()

        data['dico_id2word'] = self.data['dico'].id2word
        data['dico_word2id'] = self.data['dico'].word2id
        data['dico_counts'] = self.data['dico'].counts
        data['params'] = {k: v for k, v in self.params.__dict__.items()}

        torch.save(data, path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        if not os.path.isfile(checkpoint_path):
            if self.params.reload_checkpoint == '':
                return
            else:
                checkpoint_path = self.params.reload_checkpoint
                assert os.path.isfile(checkpoint_path)
        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        data = torch.load(checkpoint_path, map_location='cpu')

        # reload model parameters
        for name in self.MODEL_NAMES:
            try:
                getattr(self, name).load_state_dict(data[name])
            except RuntimeError as e:
                logger.error(str(e))

        # reload optimizers
        for name in self.optimizers.keys():
            if False:  # AMP checkpoint reloading is buggy, we cannot do that - TODO: fix - https://github.com/NVIDIA/apex/issues/250
                logger.warning(f"Reloading checkpoint optimizer {name} ...")
                self.optimizers[name].load_state_dict(data[f'{name}_optimizer'])
            else:  # instead, we only reload current iterations / learning rates
                logger.warning(f"Not reloading checkpoint optimizer {name}.")
                for group_id, param_group in enumerate(self.optimizers[name].param_groups):
                    if 'num_updates' not in param_group:
                        logger.warning(f"No 'num_updates' for optimizer {name}.")
                        continue
                    logger.warning(f"Reloading 'num_updates' and 'lr' for optimizer {name}.")
                    param_group['num_updates'] = data[f'{name}_optimizer']['param_groups'][group_id]['num_updates']
                    param_group['lr'] = self.optimizers[name].get_lr_for_step(param_group['num_updates'])

        # reload main metrics
        self.tracker.load('epoch', data['epoch'] + 1)
        self.tracker.load('n_total_iter', data['n_total_iter'])
        self.best_metrics = data['best_metrics']
        self.best_stopping_criterion = data['best_stopping_criterion']
        logger.warning(
            f"Checkpoint reloaded. Resuming at epoch {self.tracker.epoch} / iteration {self.tracker.n_total_iter} ...")

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        # NOTE(j_luo) if save_periodic_step is specified, use this to override save_periodic_epoch.
        if self.params.save_periodic_step > 0 and self.tracker.n_total_iter % self.params.save_periodic_step == 0:
            self.save_checkpoint('periodic-%s' % self.tracker.now, include_optimizers=False)
        elif self.params.save_periodic_epoch > 0 and self.tracker.epoch % self.params.save_periodic_epoch == 0:
            self.save_checkpoint('periodic-%s' % self.tracker.now, include_optimizers=False)

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.eval_metrics:
            if metric not in scores:
                logger.warning("Metric \"%s\" not found in scores!" % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_checkpoint('best-%s' % metric, include_optimizers=False)

    def end_interval(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None and (self.params.is_master or not self.stopping_criterion[0].endswith('_mt_bleu')):
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info("New best validation score: %f" % self.best_stopping_criterion)
                self.decrease_counts = 0
            else:
                logger.info("Not a better validation score (%i / %i)."
                            % (self.decrease_counts, self.decrease_counts_max))
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info("Stopping criterion has been below its best value for more "
                            "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                if self.params.multi_gpu and 'SLURM_JOB_ID' in os.environ:
                    os.system('scancel ' + os.environ['SLURM_JOB_ID'])
                exit()
        self.save_checkpoint('checkpoint', include_optimizers=True)

    def round_batch(self, x, lengths, positions, langs):
        """
        For float16 only.
        Sub-sample sentences in a batch, and add padding,
        so that each dimension is a multiple of 8.
        """
        params = self.params
        if not params.fp16 or len(lengths) < 8:
            return x, lengths, positions, langs, None

        # number of sentences == 0 [8]
        bs1 = len(lengths)
        bs2 = 8 * (bs1 // 8)
        assert bs2 > 0 and bs2 % 8 == 0
        if bs1 != bs2:
            idx = torch.randperm(bs1)[:bs2]
            lengths = lengths[idx]
            slen = lengths.max().item()
            x = x[:slen, idx]
            positions = None if positions is None else positions[:slen, idx]
            langs = None if langs is None else langs[:slen, idx]
        else:
            idx = None

        # sequence length == 0 [8]
        ml1 = x.size(0)
        if ml1 % 8 != 0:
            pad = 8 - (ml1 % 8)
            ml2 = ml1 + pad
            x = torch.cat([x, torch.LongTensor(pad, bs2).fill_(params.pad_index)], 0)
            if positions is not None:
                positions = torch.cat([positions, torch.arange(pad)[:, None] + positions[-1][None] + 1], 0)
            if langs is not None:
                langs = torch.cat([langs, langs[-1][None].expand(pad, bs2)], 0)
            assert x.size() == (ml2, bs2)

        assert x.size(0) % 8 == 0
        assert x.size(1) % 8 == 0
        return x, lengths, positions, langs, idx

    def _check_use_graph(self, mode):
        assert not self.use_graph, f'Mode {mode} not supported with graph right now.'

    @log_this(arg_list=['lang1', 'lang2'])
    def clm_step(self, lang1, lang2, lambda_coeff):
        """
        Next word prediction step (causal prediction).
        CLM objective.
        """
        self._check_use_graph('clm')
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        name = 'model' if params.encoder_only else 'decoder'
        model = getattr(self, name)
        model.train()

        # generate batch / select words to predict
        x, lengths, positions, langs, _ = self.generate_batch(lang1, lang2, 'causal')
        x, lengths, positions, langs, _ = self.round_batch(x, lengths, positions, langs)
        alen = torch.arange(lengths.max(), dtype=torch.long, device=lengths.device)
        pred_mask = alen[:, None] < lengths[None] - 1
        if params.context_size > 0:  # do not predict without context
            pred_mask[:params.context_size] = 0
        y = x[1:].masked_select(pred_mask[:-1])
        assert pred_mask.sum().item() == y.size(0)

        # cuda
        x, lengths, langs, pred_mask, y = to_cuda(x, lengths, langs, pred_mask, y)

        # forward / loss
        tensor = model('fwd', x=x, lengths=lengths, langs=langs, causal=True)
        _, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=False)
        name = ('CLM-%s' % lang1) if lang2 is None else ('CLM-%s-%s' % (lang1, lang2))
        weight = pred_mask.sum()
        # NOTE(j_luo) loss is already reduced to mean.
        loss = Metric(name, loss * weight, weight)
        self.metrics += loss
        loss = lambda_coeff * loss.mean

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.tracker.update('n_sentences', params.batch_size)
        self.metrics += Metric('processed_s', lengths.size(0), None, report_mean=False)
        self.metrics += Metric('processed_w', pred_mask.sum(), None, report_mean=False)

    @log_this(arg_list=['lang1', 'lang2'])
    def mlm_step(self, lang1, lang2, lambda_coeff):
        """
        Masked word prediction step.
        MLM objective is lang2 is None, TLM objective otherwise.
        """
        self._check_use_graph('mlm')
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        name = 'model' if params.encoder_only else 'encoder'
        model = getattr(self, name)
        model.train()

        # generate batch / select words to predict
        x, lengths, positions, langs, _ = self.generate_batch(lang1, lang2, 'pred')
        x, lengths, positions, langs, _ = self.round_batch(x, lengths, positions, langs)
        x, y, pred_mask = self.mask_out(x, lengths)

        # cuda
        x, y, pred_mask, lengths, positions, langs = to_cuda(x, y, pred_mask, lengths, positions, langs)

        # forward / loss
        tensor = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)
        _, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=False)
        name = ('MLM-%s' % lang1) if lang2 is None else ('MLM-%s-%s' % (lang1, lang2))
        weight = pred_mask.sum()
        loss = Metric(name, loss * weight, weight)
        self.metrics += loss
        loss = lambda_coeff * loss.mean

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.tracker.update('n_sentences', params.batch_size)
        self.metrics += Metric('processed_s', lengths.size(0), 0.0, report_mean=False)
        self.metrics += Metric('processed_w', pred_mask.sum(), 0.0, report_mean=False)

    @log_this(arg_list=['lang1', 'lang2'])
    def pc_step(self, lang1, lang2, lambda_coeff):
        """
        Parallel classification step. Predict if pairs of sentences are mutual translations of each other.
        """
        self._check_use_graph('pc')
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        name = 'model' if params.encoder_only else 'encoder'
        model = getattr(self, name)
        model.train()

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        # sample parallel sentences
        (x1, len1), (x2, len2) = self.get_batch('align', lang1, lang2)
        bs = len1.size(0)
        if bs == 1:  # can happen (although very rarely), which makes the negative loss fail
            self.tracker.update('n_sentences', params.batch_size)
            return

        # associate lang1 sentences with their translations, and random lang2 sentences
        y = torch.LongTensor(bs).random_(2)
        idx_pos = torch.arange(bs)
        idx_neg = ((idx_pos + torch.LongTensor(bs).random_(1, bs)) % bs)
        idx = (y == 1).long() * idx_pos + (y == 0).long() * idx_neg
        x2, len2 = x2[:, idx], len2[idx]

        # generate batch / cuda
        x, lengths, positions, langs = concat_batches(
            x1, len1, lang1_id, x2, len2, lang2_id, params.pad_index, params.eos_index, reset_positions=False)
        x, lengths, positions, langs, new_idx = self.round_batch(x, lengths, positions, langs)
        if new_idx is not None:
            y = y[new_idx]
        x, lengths, positions, langs = to_cuda(x, lengths, positions, langs)

        # get sentence embeddings
        h = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)[0]

        # parallel classification loss
        CLF_ID1, CLF_ID2 = 8, 9  # very hacky, use embeddings to make weights for the classifier
        emb = (model.module if params.multi_gpu else model).embeddings.weight
        pred = F.linear(h, emb[CLF_ID1].unsqueeze(0), emb[CLF_ID2, 0])
        loss = F.binary_cross_entropy_with_logits(pred.view(-1), y.to(pred.device).type_as(pred))
        raise NotImplementedError('stats not taken care of here.')
        self.stats['PC-%s-%s' % (lang1, lang2)].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.tracker.update('n_sentences', params.batch_size)
        self.stats['processed_s'] += bs
        self.stats['processed_w'] += lengths.sum().item()


class SingleTrainer(Trainer):

    def __init__(self, model, data, params):

        self.MODEL_NAMES = ['model']

        # model / data / params
        self.model = model
        self.data = data
        self.params = params

        super().__init__(data, params)


class EncDecTrainer(Trainer):

    def __init__(self, encoder, decoder, data, params):

        self.MODEL_NAMES = ['encoder', 'decoder']

        # model / data / params
        self.encoder = encoder
        self.decoder = decoder
        self.data = data
        self.params = params

        super().__init__(data, params)

    @log_this(arg_list=['lang1', 'lang2'])
    def denoise_mt_step(self, lang1, lang2, lambda_coeff):
        if lang1 != lang2:
            raise ValueError(
                'denoise_mt_step should be called for the same language, but got "{lang1}" and "{lang2}". Did you mean to use mt_step?')
        return self._mt_step(lang1, lang2, lambda_coeff)

    @log_this(arg_list=['lang1', 'lang2'])
    def mt_step(self, lang1, lang2, lambda_coeff):
        self._check_use_graph('mt')
        if lang1 == lang2:
            raise ValueError(
                'mt_step should be called for different languages, but got "{lang1}" and "{lang2}". Did you mean to use denoise_mt_step?')
        return self._mt_step(lang1, lang2, lambda_coeff)

    @log_this(arg_list=['lang'])
    def ep_step(self, lang, lambda_coeff):
        self._check_use_graph('ep')
        return self._mt_step(lang, lang, lambda_coeff, ep=True)

    def _encode(self, encoder, mode, **kwargs):
        if 'graph_info' in kwargs and kwargs['graph_info'] is None:
            del kwargs['graph_info']
        if 'use_graph_loss' in kwargs:
            if kwargs['use_graph_loss']:
                kwargs['return_graph_data'] = True
            del kwargs['use_graph_loss']
        return encoder(mode, **kwargs)

    def _mt_step(self, lang1, lang2, lambda_coeff, ep=False):
        """
        Machine translation step.
        Can also be used for denoising auto-encoding.
        `ep` stands for eat-plain.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        self.encoder.train()
        self.decoder.train()

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        # generate batch
        use_graph_loss = (lang1 in self.supervised_graph) and (lang1 == lang2)
        if lang1 == lang2:
            input_format = 'eat' if ep else 'plain'
            x1, len1, *indices = self.get_batch('ae', lang1, input_format=input_format, return_indices=use_graph_loss)
            (x2, len2) = (x1, len1)
            if not ep or params.ep_add_noise:
                x1, len1, permutations, keep = self.add_noise(x1, len1)
        else:
            (x1, len1), (x2, len2) = self.get_batch('mt', lang1, lang2)
        langs1 = x1.clone().fill_(lang1_id)
        langs2 = x2.clone().fill_(lang2_id)

        # target words to predict
        alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
        pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
        y = x2[1:].masked_select(pred_mask[:-1])
        assert len(y) == (len2 - 1).sum().item()

        # cuda
        x1, len1, langs1, x2, len2, langs2, y = to_cuda(x1, len1, langs1, x2, len2, langs2, y)

        # Compute graph_info if needed
        graph_info = None
        if self.use_graph:
            if lang1 != lang2:
                raise RuntimeError('use_graph not activated for BT')
            else:
                graph_info = self.verifier.get_graph_info(x1)

        # encode source sentence
        kwargs = {'x': x1, 'lengths': len1, 'langs': langs1, 'causal': False,
                  'graph_info': graph_info, 'use_graph_loss': use_graph_loss}
        if use_graph_loss:  # NOTE(j_luo) Return graph data for supervising graph induction.
            assert len(indices) == 1
            indices = indices[0]
        graph_data = None
        if use_graph_loss:
            enc1, graph_data = self._encode(self.encoder, 'fwd', **kwargs)
        else:
            enc1 = self._encode(self.encoder, 'fwd', **kwargs)
        enc1 = enc1.transpose(0, 1)

        # Modify src_len if use_graph. It has been changed to represent words instead of BPEs.
        if graph_info is not None:
            len1 = graph_info.word_lengths

        # decode target sentence
        dec2 = self.decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1)

        # loss
        _, loss = self.decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=False)
        name = ('AE-%s' % lang1) if lang1 == lang2 else ('MT-%s-%s' % (lang1, lang2))
        weight = pred_mask.sum()
        loss = Metric(name, loss * weight, weight)
        self.metrics += loss
        loss = lambda_coeff * loss.mean

        # Add the graph induction loss if needed.
        final_loss = loss
        if use_graph_loss:
            graph_target = self.verifier.get_graph_target(x1, lang1, max(graph_info.word_lengths), indices, permutations=permutations, keep=keep)
            loss_edge_type, loss_edge_norm = self.verifier.get_graph_loss(graph_data, graph_target, lang1)
            final_loss = loss + params.lambda_graph * (loss_edge_type.mean + loss_edge_norm.mean)
            self.metrics += Metrics(loss_edge_type, loss_edge_norm)

        # optimize
        self.optimize(final_loss)

        # number of processed sentences / words
        self.tracker.update('n_sentences', params.batch_size)
        self.metrics += Metric('processed_s', len2.size(0), 0.0, report_mean=False)
        self.metrics += Metric('processed_w', (len2 - 1).sum().item(), 0.0, report_mean=False)

    @log_this(arg_list=['lang1', 'lang2', 'lang3'])
    def bt_step(self, lang1, lang2, lang3, lambda_coeff):
        """
        Back-translation step for machine translation.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        assert lang1 == lang3 and lang1 != lang2 and lang2 is not None
        params = self.params
        _encoder = self.encoder.module if params.multi_gpu else self.encoder
        _decoder = self.decoder.module if params.multi_gpu else self.decoder

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        # generate source batch
        x1, len1 = self.get_batch('bt', lang1)
        langs1 = x1.clone().fill_(lang1_id)

        # cuda
        x1, len1, langs1 = to_cuda(x1, len1, langs1)

        # Get graph_info if using graph.
        graph_info1 = None
        if self.use_graph:
            graph_info1 = self.verifier.get_graph_info(x1)  # generate a translation
        kwargs = {
            'x': x1, 'lengths': len1, 'langs': langs1, 'causal': False, 'graph_info': graph_info1
        }
        with torch.no_grad():

            # evaluation mode
            self.encoder.eval()
            self.decoder.eval()

            # encode source sentence and translate it
            enc1 = self._encode(_encoder, 'fwd', **kwargs)
            enc1 = enc1.transpose(0, 1)
            if graph_info1 is not None:
                real_len1 = graph_info1.word_lengths
            else:
                real_len1 = len1
            x2, len2 = _decoder.generate(enc1, real_len1, lang2_id, max_len=int(1.3 * len1.max().item() + 5))
            langs2 = x2.clone().fill_(lang2_id)

            # free CUDA memory
            del enc1

            # training mode
            self.encoder.train()
            self.decoder.train()

        # encode generate sentence
        graph_info2 = None
        if self.use_graph:
            graph_info2 = self.verifier.get_graph_info(x2)
        kwargs = {
            'x': x2, 'lengths': len2, 'langs': langs2, 'causal': False, 'graph_info': graph_info2
        }
        enc2 = self._encode(self.encoder, 'fwd', **kwargs)
        # enc2 = self.encoder('fwd', x=x2, lengths=len2, langs=langs2, causal=False)  # , graph_info=graph_info)
        enc2 = enc2.transpose(0, 1)

        # words to predict
        alen = torch.arange(len1.max(), dtype=torch.long, device=len1.device)
        pred_mask = alen[:, None] < len1[None] - 1  # do not predict anything given the last target word
        y1 = x1[1:].masked_select(pred_mask[:-1])

        # decode original sentence
        if graph_info1 is not None:
            real_len2 = graph_info2.word_lengths
        else:
            real_len2 = len2
        dec3 = self.decoder('fwd', x=x1, lengths=len1, langs=langs1, causal=True, src_enc=enc2, src_len=real_len2)

        # loss
        _, loss = self.decoder('predict', tensor=dec3, pred_mask=pred_mask, y=y1, get_scores=False)
        name = ('BT-%s-%s-%s' % (lang1, lang2, lang3))
        weight = pred_mask.sum()
        loss = Metric(name, loss * weight, weight)
        self.metrics += loss
        loss = loss.mean  # NOTE(j_luo) Didn't see lambda get used here.

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.tracker.update('n_sentences', params.batch_size)
        self.metrics += Metric('processed_s', len1.size(0), 0.0, report_mean=False)
        self.metrics += Metric('processed_w', (len1 - 1).sum().item(), 0.0, report_mean=False)
