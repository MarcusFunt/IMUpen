"""Offline M4 tests using synthetic official-format OnHW archive fixtures."""
import json
import pickle
import numpy as np
import pytest
import torch

from smartpen.ml.model import CNNBiLSTMCTC, required_ctc_steps
from smartpen.ml.onhw import load_words500, split_training_writers
from smartpen.ml.text import Vocabulary, distance, metrics
from smartpen.ml.training import fit_normalizer, main
from smartpen.ml.predict import transcribe

def fixture_fold(root):
    d=root/"0"
    d.mkdir(parents=True)
    vocab=Vocabulary()
    rng=np.random.default_rng(123)
    words=["Hallo","Welt","Test","Hallo","Welt","Test","Hallo","Test","Welt","Hallo"]
    def data(n,ids):
        x=[rng.normal(size=(64+i,13)).astype("float32") for i in range(n)]
        y=[[vocab.chars.index(c) for c in words[i%len(words)]] for i in range(n)]
        y=[seq+[len(vocab.chars)]*(19-len(seq)) for seq in y]
        return x,y,ids
    tr=data(10,[100]*5+[101]*5)
    va=data(4,[200]*4)
    for name,obj in {"all_x_dat_train_imu.pkl":tr[0],
                     "all_train_gt.pkl":tr[1],"train_ids.pkl":tr[2],
                     "all_x_dat_val_imu.pkl":va[0],
                     "all_val_gt.pkl":va[1],"val_ids.pkl":va[2]}.items():
        with (d/name).open("wb") as f: pickle.dump(obj,f)
    return root

def test_vocab_and_metrics():
    v=Vocabulary("ab ")
    assert v.greedy([0,1,1,0,1,2,2,0])=="aab"
    assert v.encode("ab")==[1,2]
    assert required_ctc_steps(v.encode("aa"))==3
    assert distance("abcd","abX")==2
    assert metrics(["hello","a"],["hallo","a"])["cer"]==1/6

def test_length_padding_and_backward():
    torch.manual_seed(1)
    model=CNNBiLSTMCTC(8,channels=8,hidden_size=8,layers=2)
    a=torch.randn(1,41,6)
    b=torch.randn(1,56,6)
    model.eval()
    with torch.no_grad():
        one,n=model(a,torch.tensor([41]))
        joint=torch.zeros(2,56,6)
        joint[0,:41]=a[0];joint[1]=b[0]
        both,lens=model(joint,torch.tensor([41,56]))
    assert torch.allclose(one[:,0],both[:int(n[0]),0],atol=1e-4)
    assert lens.tolist()==[11,14]
    model.train()
    out,lens=model(torch.randn(2,64,6),torch.tensor([64,61]))
    loss=torch.nn.CTCLoss(blank=0)(out,torch.tensor([1,2,3,4,5]),
                                  lens,torch.tensor([3,2]))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.any(p.grad!=0) for p in model.parameters())

def test_onhw_loader_and_writer_separation(tmp_path):
    root=fixture_fold(tmp_path)
    with pytest.raises(PermissionError): load_words500(root)
    d=load_words500(root,trust_pickle=True)
    assert len(d.train)==10 and len(d.official_val)==4
    assert d.train[0].x.shape==(64,6)
    with (root/"0"/"all_x_dat_train_imu.pkl").open("rb") as f:
        source=pickle.load(f)
    assert d.train[0].x[0].tolist()==pytest.approx(source[0][0,3:9])
    assert d.train[0].text=="Hallo"
    assert {r.writer for r in d.official_val}=={"200"}
    a,b=split_training_writers(d.train)
    assert {r.writer for r in a}.isdisjoint({r.writer for r in b})
    mean,std=fit_normalizer(a)
    assert mean.shape==std.shape==(6,) and np.all(std>0)

def test_invalid_label_dropped(tmp_path):
    root=fixture_fold(tmp_path)
    file=root/"0"/"all_train_gt.pkl"
    with file.open("rb") as f: labels=pickle.load(f)
    labels[0][1]=999
    with file.open("wb") as f: pickle.dump(labels,f)
    d=load_words500(root,trust_pickle=True)
    assert len(d.train)==9 and d.dropped_train==1

def test_end_to_end_fixture(tmp_path):
    root=fixture_fold(tmp_path/"onhw")
    out=tmp_path/"run"
    result=main(["--onhw",str(root),"--trust-pickle","--out",str(out),
                 "--epochs","1","--batch-size","5","--channels","8",
                 "--hidden","8","--layers","1","--device","cpu"])
    assert result["official_validation_test"]["samples"]==4
    assert result["official_partition_writer_disjoint"]
    assert (out/"best.pt").exists()
    assert json.loads((out/"results.json").read_text())["fit_samples"]==5
    np.savez(tmp_path/"sample.npz",axes=np.zeros((80,6),dtype="float32"))
    assert isinstance(transcribe(tmp_path/"sample.npz",out/"best.pt"),str)
