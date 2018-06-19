import os
import logging
import argparse

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

group = parser.add_argument_group("Training Options")
group.add_argument("--save-dir", type=str, required=True)
group.add_argument("--save-period", type=int, default=1)
group.add_argument("--batch-size", type=int, default=32)
group.add_argument("--epochs", type=int, default=12)
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
        pad_eos="<eos>",
        unk=args.unk,
    )
    if vocab is None:
        vocab = dset.vocab
    collator = dataset.TextSequenceBatchCollator(
        pad_idx=len(vocab)
    )
    return td.DataLoader(
        dataset=dset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.data_workers,
        collate_fn=collator,
        pin_memory=args.pin_memory
    )


def prepare_model(args, dataloader):
    mdl = model.build_model(args, len(dataloader.dataset.vocab))
    mdl.reset_parameters()
    embeds.load_embeddings(
        args=args,
        vocab=dataloader.dataset.vocab,
        params=mdl.embedding.weight.detach()
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
                 dynamic_rnn, optimizer_cls=op.Adam,
                 tensor_key="tensor", samples=1):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.vocab = vocab
        self.dynamic_rnn = dynamic_rnn
        self.save_dir = save_dir
        self.save_period = save_period
        self.optimizer_cls = optimizer_cls
        self.tensor_key = tensor_key
        self.samples = samples
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=len(vocab))
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
        self.progress.set_description(desc)

    def report_samples(self, sents, preds, lens):
        def to_sent(vec):
            return " ".join(self.vocab.i2f.get(w, self.unk) for w in vec)
        idx = torch.randperm(len(sents))[:self.samples]
        sents = [sents[i] for i in idx]
        preds, lens = preds[idx].cpu().tolist(), lens[idx].cpu().tolist()
        preds = [to_sent(pred[:l]) for pred, l in zip(preds, lens)]
        for i, (sent, pred) in enumerate(zip(sents, preds)):
            logging.info(f"Sample #{i + 1:02d}:")
            logging.info(f"Target:\t{sent}")
            logging.info(f"Predicted:\t{pred}")

    def train(self, dataloader):
        self.global_step = 0
        self.model.train(True)
        optimizer = self.optimizer_cls(list(self.trainable_params()))
        self.progress = utils.tqdm(total=len(dataloader.dataset))

        for eidx in range(1, self.epochs + 1):
            self.local_step = 0
            for batch in dataloader:
                optimizer.zero_grad()
                batch_size, x, lens, targets = self.prepare_batch(batch)
                self.global_step += batch_size
                self.local_step += batch_size
                self.progress.update(batch_size)
                ret = self.model(x, lens)
                logits, loss_kld = ret.get("pass"), ret.get("loss").mean()
                loss = self.calculate_celoss(logits, targets) + loss_kld
                loss.backward()
                optimizer.step()

                stats = {
                    "loss": loss.item(),
                    "loss-kld": loss_kld.item()
                }
                self.report_stats(stats)
                self.report_samples(batch.get("string"), logits.max(2)[1], lens)

            if eidx % self.save_period == 0:
                self.snapshot(eidx)


def report_model(trainer):
    params = sum(np.prod(p.size()) for p in trainer.trainable_params())
    logging.info(f"Number of parameters: {params:,}")


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    utils.save_args(parser, args,
                    os.path.join(args.save_dir, args.argparse_filename))
    utils.config_basic_logger(args)
    devices = utils.get_devices(args.gpu)
    if args.seed is not None:
        utils.manual_seed(args.seed)

    logging.info("Loading data...")
    dataloader = create_dataloader(args)
    vocab = dataloader.dataset.vocab
    utils.save_pkl(vocab, os.path.join(args.save_dir, "vocab.pkl"))

    logging.info("Initializing training environment...")
    mdl = prepare_model(args, dataloader)
    mdl = utils.to_device(mdl, devices)
    optimizer_cls = get_optimizer_cls(args)
    trainer = Trainer(
        model=mdl,
        device=devices[0],
        vocab=vocab,
        epochs=args.epochs,
        save_dir=args.save_dir,
        save_period=args.save_period,
        optimizer_cls=optimizer_cls,
        tensor_key="tensor",
        samples=args.samples,
        dynamic_rnn=args.vae_encoder_cell_dynamic or args.vae_decoder_cell_dynamic
    )
    report_model(trainer)

    logging.info("Commecing training...")
    trainer.train(dataloader)

    logging.info("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    train(args)