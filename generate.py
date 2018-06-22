import os
import logging
import argparse
import collections

import torch
import torch.nn as nn
import torch.utils.data as td

import utils
import model
import dataset


parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

group = parser.add_argument_group("Logging Options")
utils.add_logging_arguments(group, "generate")
group.add_argument("--argparse-filename",
                   type=str, default="generate-argparse.yml")
group.add_argument("--samples-filename", type=str, default="samples.txt")
group.add_argument("--neighbors-filename", type=str, default="neighbors.txt")
group.add_argument("--show-progress", action="store_true", default=False)

group = parser.add_argument_group("Data Options")
group.add_argument("--data-path", type=str, default=None)
group.add_argument("--vocab", type=str, required=True)
group.add_argument("--data-workers", type=int, default=8)
group.add_argument("--seed", type=int, default=None)
group.add_argument("--unk", type=str, default="<unk>")
group.add_argument("--eos", type=str, default="<eos>")
group.add_argument("--bos", type=str, default="<bos>")

group = parser.add_argument_group("Generation Options")
group.add_argument("--ckpt-path", type=str, required=True)
group.add_argument("--decoder-key", type=str, default="decoder",
                   help="decoder key name in state dict loaded from checkpoint")
group.add_argument("--batch-size", type=int, default=32)
group.add_argument("--save-dir", type=str, required=True)
group.add_argument("--z-samples", type=int, default=100)
group.add_argument("--embed-freeze", type=int,)
group.add_argument("--nearest-neighbors", type=int, default=None)
group.add_argument("--max-length", type=int, default=30)
group.add_argument("--gpu", type=int, action="append", default=[])

group = parser.add_argument_group("Model Options")
model.add_arguments(group)


def prepare_dataset(args, vocab):
    dset = dataset.TextSequenceDataset(
        path=args.data_path,
        feats=["string", "tensor"],
        vocab=vocab,
        pad_eos=args.eos,
        pad_bos=args.bos,
        unk=args.unk
    )
    return dset
    # collator = dataset.TextSequenceBatchCollator(
    #     pad_idx=len(vocab)
    # )
    # return td.DataLoader(
    #     dataset=dset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.data_workers,
    #     collate_fn=collator,
    #     pin_memory=False
    # )


def filter_ckpt(ckpt, key):
    return {k[len(key):].lstrip("."): v for k, v in ckpt.items()
            if k.startswith(key)}


def load_decoder_ckpt(args):
    return filter_ckpt(torch.load(args.ckpt_path), args.decoder_key)


def prepare_model(args, vocab):
    mdl = model.build_model(args, vocab)
    ckpt = torch.load(args.ckpt_path)
    mdl.load_state_dict(ckpt)
    return mdl


class Generator(object):
    def __init__(self, model, device, batch_size, vocab, bos, eos, unk,
                 max_len):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.vocab = vocab
        self.bos = bos
        self.eos = eos
        self.unk = unk
        self.bos_idx = vocab.f2i.get(bos)
        self.eos_idx = vocab.f2i.get(eos)
        self.unk_idx = vocab.f2i.get(unk)
        self.max_len = max_len

    @property
    def module(self):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        else:
            return self.model

    def sample_z(self, num_samples):
        return torch.randn(num_samples, self.module.z_dim).to(self.device)

    def to_sent(self, idx):
        return " ".join(self.vocab.i2f.get(w, self.unk) for w in idx)

    def generate(self, num_samples):
        self.model.train(False)
        z = self.sample_z(num_samples)
        samples = []
        progress = utils.tqdm(total=num_samples, desc="generating")

        for i in range(0, num_samples, self.batch_size):
            z_batch = z[i:i + self.batch_size]
            progress.update(z_batch.size(0))
            x = z.new(z_batch.size(0), 1).fill_(self.bos_idx).long()
            x, lens = self.model.decode(z_batch, x,
                eos_idx=self.eos_idx,
                max_len=self.max_len
            )
            x, lens = x.cpu().tolist(), lens.cpu().tolist()
            for sent, l in zip(x, lens):
                samples.append(self.to_sent(sent[:l]))

        return samples


def create_bag_of_words(string):
    return utils.normalize(collections.Counter(string.split()))


def bow_cosine_similarity(bow1, bow2):
    common_keys = set(bow1) & set(bow2)
    return sum(bow1[k] * bow2[k] for k in common_keys)


def nearest_neighbors(args, samples, dataset):
    dataset_bow = [create_bag_of_words(dataset[i]["string"])
                   for i in range(len(dataset))]
    samples_bow = [create_bag_of_words(sample) for sample in samples]
    knn = []
    progress = utils.tqdm(total=len(samples), desc="finding knn")
    for sample, sample_bow in zip(samples, samples_bow):
        progress.update(1)
        sims = [(bow_cosine_similarity(sample_bow, dataset_bow[i]),
                 dataset[i]["string"]) for i in range(len(dataset))]
        sims.sort(key=lambda x: x[0], reverse=True)
        knn.append([x[1] for x in sims[:args.nearest_neighbors]])
    return knn


def save(args, samples, neighbors=None):
    samples_path = os.path.join(args.save_dir, args.samples_filename)
    with open(samples_path, "w") as f:
        for sample in samples:
            f.write(f"{sample}\n")
    if neighbors is not None:
        neighbors_path = os.path.join(args.save_dir, args.neighbors_filename)
        with open(neighbors_path, "w") as f:
            for neighbor in neighbors:
                delim = "\t"
                f.write(f"{delim.join(neighbor)}\n")


def generate(args):
    devices = utils.get_devices(args.gpu)
    if args.seed is not None:
        utils.manual_seed(args.seed)

    logging.info("Loading data...")
    vocab = utils.load_pkl(args.vocab)

    logging.info("Initializing generation environment...")
    model = prepare_model(args, vocab)
    model = utils.to_device(model, devices)
    generator = Generator(
        model=model,
        device=devices[0],
        batch_size=args.batch_size,
        vocab=vocab,
        bos=args.bos,
        eos=args.eos,
        unk=args.unk,
        max_len=args.max_length
    )

    logging.info("Commencing generation...")
    samples = generator.generate(args.z_samples)
    if args.nearest_neighbors is not None:
        dataset = prepare_dataset(args, vocab)
        neighbors = nearest_neighbors(args, samples, dataset)
    else:
        neighbors = None
    save(args, samples, neighbors)

    logging.info("Done!")


if __name__ == "__main__":
    args = utils.initialize_script(parser)
    generate(args)