"""Reproducible writer-grouped OnHW CNN-BiLSTM-CTC training and evaluation."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .model import CNNBiLSTMCTC
from .onhw import Record, load_words500, split_training_writers
from .text import Vocabulary, metrics


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_normalizer(records: list[Record]):
    if not records: raise ValueError("No training samples")
    total = sum(len(r.x) for r in records)
    mean = sum((r.x.astype(np.float64).sum(axis=0) for r in records),
               np.zeros(6, dtype=np.float64)) / total
    var = sum((((r.x.astype(np.float64) - mean)**2).sum(axis=0) for r in records),
              np.zeros(6, dtype=np.float64)) / total
    return mean.astype(np.float32), np.maximum(np.sqrt(var), 1e-6).astype(np.float32)


class SequenceDataset(Dataset):
    def __init__(self, records, vocab, mean, std):
        self.records, self.vocab, self.mean, self.std = records, vocab, mean, std
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        rec = self.records[index]
        x = torch.from_numpy(((rec.x - self.mean)/self.std).astype(np.float32))
        y = torch.as_tensor(self.vocab.encode(rec.text), dtype=torch.long)
        return x, y, rec.text


def collate(batch):
    xs, ys, texts = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    target_lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    return padded, lengths, torch.cat(ys), target_lengths, list(texts)


def dataloader(records, vocab, mean, std, batch_size, shuffle):
    return DataLoader(SequenceDataset(records, vocab, mean, std),
                      batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, num_workers=0)


def evaluate(model, loader, device, *, criterion=None):
    model.eval()
    all_refs, all_preds, losses = [], [], []
    with torch.no_grad():
        for xs, lengths, targets, target_lengths, refs in loader:
            logp, out_lengths = model(xs.to(device), lengths)
            if criterion is not None:
                loss = criterion(logp.float(), targets, out_lengths, target_lengths)
                losses.append(float(loss))
            best = logp.argmax(dim=-1).cpu().transpose(0, 1)
            for ids, end in zip(best, out_lengths):
                all_preds.append(loader.dataset.vocab.greedy(ids[:int(end)].tolist()))
            all_refs.extend(refs)
    out = metrics(all_preds, all_refs)
    out["ctc_loss"] = float(np.mean(losses)) if losses else None
    out["examples"] = list(zip(all_refs, all_preds))[:15]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Train 6-axis CNN-BiLSTM-CTC on OnHW-Words500")
    p.add_argument("--onhw", type=Path, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--trust-pickle", action="store_true",
                   help="Only unpickle verified authentic OnHW archives")
    p.add_argument("--out", type=Path, default=Path("runs/m4-onhw"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train", type=int, help="Debug subset; NOT a benchmark")
    p.add_argument("--max-official-val", type=int, help="Debug subset; NOT a benchmark")
    p.add_argument("--val-writer-fraction", type=float, default=0.15)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    p.add_argument("--skip-official-test", action="store_true")
    args = p.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.val_writer_fraction < 1:
        p.error("Epochs/batch size must be positive; val writer fraction between 0 and 1")
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    fold = load_words500(args.onhw, fold=args.fold, trust_pickle=args.trust_pickle,
                         max_train=args.max_train, max_val=args.max_official_val,
                         seed=args.seed)
    fit, inner_val = split_training_writers(fold.train, fraction=args.val_writer_fraction,
                                           seed=args.seed)
    fit_ids = {r.writer for r in fit}
    inner_ids = {r.writer for r in inner_val}
    assert fit_ids.isdisjoint(inner_ids)
    official_ids = {r.writer for r in fold.official_val}
    independent = fit_ids.isdisjoint(official_ids) and inner_ids.isdisjoint(official_ids)
    vocab = Vocabulary()
    mean, std = fit_normalizer(fit)
    train_loader = dataloader(fit, vocab, mean, std, args.batch_size, True)
    val_loader = dataloader(inner_val, vocab, mean, std, args.batch_size, False)
    model = CNNBiLSTMCTC(vocab.size, channels=args.channels,
                          hidden_size=args.hidden, layers=args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CTCLoss(blank=0, zero_infinity=False, reduction="mean")
    best = float("inf")
    best_path = args.out / "best.pt"
    history = []
    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        for xs, lengths, targets, target_lengths, _ in train_loader:
            log_probs, out_lengths = model(xs.to(device), lengths)
            loss = criterion(log_probs.float(), targets, out_lengths, target_lengths)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite CTC loss; check sequence/target lengths")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val = evaluate(model, val_loader, device, criterion=criterion)
        entry = {"epoch":epoch,"train_ctc_loss":float(np.mean(losses)),
                 "inner_val_cer":val["cer"],"inner_val_wer":val["wer"],
                 "inner_val_loss":val["ctc_loss"]}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if val["cer"] < best:
            best = val["cer"]
            torch.save({"state_dict":model.state_dict(),"config":model.config,
                        "vocabulary":vocab.chars,"normalizer_mean":mean.tolist(),
                        "normalizer_std":std.tolist(),"seed":args.seed,
                        "fold":args.fold,"selection_epoch":epoch},best_path)
    official = None
    if not args.skip_official_test:
        saved = torch.load(best_path,map_location=device,weights_only=True)
        model.load_state_dict(saved["state_dict"])
        official_loader = dataloader(fold.official_val,vocab,mean,std,args.batch_size,False)
        official = evaluate(model,official_loader,device,criterion=criterion)
    result = {
        "status":"completed",
        "protocol":"onhw-words500-official-fold-with-inner-writer-selection",
        "dataset_root":str(args.onhw),"fold":args.fold,"seed":args.seed,
        "training_writers":sorted(fit_ids),"inner_validation_writers":sorted(inner_ids),
        "official_partition_writers":sorted(official_ids),
        "official_partition_writer_disjoint":independent,
        "fit_samples":len(fit),"inner_val_samples":len(inner_val),
        "official_val_samples":len(fold.official_val),
        "rejected_short_invalid":{"train":fold.dropped_train,
                                  "official_val":fold.dropped_val},
        "debug_subset_not_official_benchmark":bool(args.max_train or args.max_official_val),
        "model_parameters":sum(p.numel() for p in model.parameters()),
        "best_inner_val_cer":best,"history":history,
        "official_validation_test":official,"checkpoint":str(best_path)
    }
    (args.out/"results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items()
                      if k not in ("history","training_writers","inner_validation_writers","official_partition_writers")},indent=2),flush=True)
    return result

if __name__ == "__main__":
    main()
