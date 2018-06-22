import os
import logging
import argparse
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.optim as op
import torch.utils.data as td

import utils
import model
import dataset
from . import embeds


parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

group = parser.add_argument_group("Logging Options")
utils.add_logging_arguments(group, "train")
group.add_argument("--argparse-filename",
                   type=str, default="train-argparse.yml")
group.add_argument("--show-progress", action="store_true", default=False)

group = parser.add_argument_group("Model Parameters")
model.add_arguments(group)

group = parser.add_argument_group("Data Options")
group.add_argument("--data-path", type=str, required=True)
group.add_argument("--vocab", type=str, default=None)
group.add_argument("--vocab-limit", type=int, default=None)
group.add_argument("--data-workers", type=int, default=8)
group.add_argument("--pin-memory", action="store_true", default=False)
group.add_argument("--shuffle", action="store_true", default=False)
group.add_argument("--seed", type=int, default=None)
group.add_argument("--unk", type=str, default="<unk>")
group.add_argument("--eos", type=str, default="<eos>")
group.add_argument("--bos", type=str, default="<bos>")

group = parser.add_argument_group("Training Options")
group.add_argument("--save-dir", type=str, required=True)
group.add_argument("--save-period", type=int, default=1)
group.add_argument("--batch-size", type=int, default=32)
group.add_argument("--epochs", type=int, default=12)
group.add_argument("--kld-annealing", type=float, default=None)
group.add_argument("--optimizer", type=str, default="adam",
                   choices=["adam", "adamax", "adagrad", "adadelta"])
group.add_argument("--learning-rate", type=float, default=None)
group.add_argument("--samples", type=int, default=1)
group.add_argument("--gpu", type=int, action="append", default=[])

group = parser.add_argument_group("Word Embeddings Options")
embeds.add_embed_arguments(group)


def create_dataloader(args):
    vocab = None
    if args.vocab is not None:
        vocab = utils.load_pkl(args.vocab)
    dset = dataset.TextSequenceDataset(
        path=args.data_path,
        feats=["string", "tensor"],
        vocab=vocab,
        vocab_limit=args.vocab_limit,
        pad_eos=args.eos,
        pad_bos=args.bos,
        unk=args.unk,
    )
    if vocab is None:
        vocab = dset.vocab
    collator = dataset.TextSequenceBatchCollator(pad_idx=len(vocab))
    return td.DataLoader(
        dataset=dset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.data_workers,
        collate_fn=collator,
        pin_memory=args.pin_memory
    )


def prepare_model(args, dataloader):
    mdl = model.build_model(args, dataloader.dataset.vocab)
    mdl.reset_parameters()
    embeds.load_embeddings(
        args=args,
        vocab=dataloader.dataset.vocab,
        modules=[mdl.input_embed, mdl.output_embed]
    )
    return mdl


def get_optimizer_cls(args):
    kwargs = dict()
    if args.learning_rate is not None:
        kwargs["lr"] = args.learning_rate
    return utils.map_val(args.optimizer, {
        "adam": lambda p: op.Adam(p, **kwargs),
        "adamax": lambda p: op.Adamax(p, **kwargs),
        "adagrad": lambda p: op.Adagrad(p, **kwargs),
        "adadelta": lambda p: op.Adadelta(p, **kwargs)
    }, "optimizer")


class Trainer(object):
    def __init__(self, model, device, vocab, epochs, save_dir, save_period,
                 dynamic_rnn, optimizer_cls=op.Adam, show_progress=True,
                 kld_annealing=None, tensor_key="tensor", samples=1):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.vocab = vocab
        self.save_dir = save_dir
        self.save_period = save_period
        self.dynamic_rnn = dynamic_rnn
        self.optimizer_cls = optimizer_cls
        self.tensor_key = tensor_key
        self.kld_annealing = kld_annealing
        self.samples = samples
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=len(vocab))
        self.show_progress = show_progress
        self.progress = None
        self.unk = "<unk>"

    @property
    def module(self):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        else:
            return self.model

    def trainable_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                yield p

    def prepare_batch(self, batch):
        x, lens = batch[self.tensor_key]
        if self.dynamic_rnn:
            lens, idx = torch.sort(lens, 0, True)
            x = x[idx]
            string = batch["string"]
            string = [string[i] for i in idx.tolist()]
            batch["string"] = string
        batch_size = x.size(0)
        x, lens = x.to(self.device), lens.to(self.device)
        targets = x[:, 1:].contiguous()
        return batch_size, x, lens, targets

    def calculate_celoss(self, logits, targets):
        logits = logits.view(-1, len(self.vocab) + 1)
        targets = targets.view(-1)
        return self.cross_entropy(logits, targets)

    def snapshot(self, eidx):
        state_dict = self.module.state_dict()
        path = os.path.join(self.save_dir, f"checkpoint-e{eidx:02d}")
        torch.save(state_dict, path)
        logging.info(f"checkpoint saved to '{path}'.")

    def report_stats(self, stats):
        stats = {k: f"{v:.4f}" for k, v in stats.items()}
        desc = utils.join_dict(stats, ", ", "=")
        return desc

    def report_samples(self, sents, preds, lens):
        def to_sent(vec):
            return " ".join(self.vocab.i2f.get(w, self.unk) for w in vec)
        idx = torch.randperm(len(sents))[:self.samples]
        sents = [sents[i] for i in idx]
        preds, lens = preds[idx].cpu().tolist(), lens[idx].cpu().tolist()
        preds = [to_sent(pred[:l]) for pred, l in zip(preds, lens)]
        for i, (sent, pred) in enumerate(zip(sents, preds)):
            logging.info(f"Sample #{i + 1:02d}:")
            logging.info(f"Target:    {sent}")
            logging.info(f"Predicted: {pred}")

    def train(self, dataloader):
        self.global_step = 0
        self.model.train(True)
        optimizer = self.optimizer_cls(list(self.trainable_params()))
        self.progress = utils.tqdm(
            total=len(dataloader.dataset),
            disable=not self.show_progress
        )
        if self.kld_annealing is not None:
            kld_scale = 0.0
        else:
            kld_scale = 1.0

        for eidx in range(1, self.epochs + 1):
            self.local_step = 0
            stats_cum = collections.defaultdict(float)
            for batch in dataloader:
                optimizer.zero_grad()
                batch_size, x, lens, targets = self.prepare_batch(batch)
                self.global_step += batch_size
                self.local_step += batch_size
                self.progress.update(batch_size)
                ret = self.model(x, lens)
                logits, loss_kld = ret.get("pass"), ret.get("loss")
                loss = self.calculate_celoss(logits, targets)
                if loss_kld is not None:
                    loss += kld_scale * loss_kld.mean()
                loss.backward()
                optimizer.step()

                stats = {"loss": loss.item()}
                if loss_kld is not None:
                    stats["loss-kld"] = kld_scale * loss_kld.mean().item()
                    stats["kld-anneal"] = kld_scale
                for k, v in stats.items():
                    stats_cum[f"{k}-cum"] += v * batch_size
                desc = self.report_stats(stats)
                self.progress.set_description(desc)
                self.report_samples(batch.get("string"), logits.max(2)[1], lens)
            stats_cum = {k: v / self.local_step for k, v in stats_cum.items()}
            desc = self.report_stats(stats_cum)
            logging.info(f"[{eidx}] {desc}")
            if self.kld_annealing is not None:
                kld_scale += self.kld_annealing
                kld_scale = min(1.0, kld_scale)

            if eidx % self.save_period == 0:
                self.snapshot(eidx)


def report_model(trainer):
    params = sum(np.prod(p.size()) for p in trainer.trainable_params())
    logging.info(f"Number of parameters: {params:,}")


def train(args):
    devices = utils.get_devices(args.gpu)
    if args.seed is not None:
        utils.manual_seed(args.seed)

    logging.info("Loading data...")
    dataloader = create_dataloader(args)
    vocab = dataloader.dataset.vocab
    utils.save_pkl(vocab, os.path.join(args.save_dir, "vocab.pkl"))

    logging.info("Initializing training environment...")
    mdl = prepare_model(args, dataloader)
    optimizer_cls = get_optimizer_cls(args)
    trainer = Trainer(
        model=utils.to_device(mdl, devices),
        device=devices[0],
        vocab=vocab,
        epochs=args.epochs,
        save_dir=args.save_dir,
        save_period=args.save_period,
        optimizer_cls=optimizer_cls,
        tensor_key="tensor",
        samples=args.samples,
        show_progress=args.show_progress,
        kld_annealing=args.kld_annealing,
        dynamic_rnn=mdl.encoder.rnn.dynamic or mdl.decoder.rnn.dynamic
    )
    report_model(trainer)

    logging.info("Commecing training...")
    trainer.train(dataloader)

    logging.info("Done!")


if __name__ == "__main__":
    train(utils.initialize_script(parser))