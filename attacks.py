import os
import pickle

import torch
import librosa
import numpy as np
from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F

def attack_wav(dataloader, x,y, model, n_iter=10, eps=0.1, sign=True):
    model.eval()
    correct = 0
    total = 0

    print(f"running test on {x}")
    y = torch.LongTensor([y]).cuda()
    audio, sr = librosa.load(x, sr=None)
    audio = np.expand_dims(audio, 0)

    audio = torch.FloatTensor(audio)
    delta = torch.zeros_like(audio, requires_grad=True)

    for i in range(n_iter):
        spect, phase = dataloader.dataset.stft.transform(audio + delta)

        spect = spect.unsqueeze(0).cuda()
        yhat = model(spect)
        loss = F.nll_loss(yhat, y)
        loss.backward()
        # if sign:
        #     delta.data = delta + eps * torch.sign(delta.grad.data)
        # else:
        #     delta.data = delta + eps * delta.grad.data
        delta.data = delta + 2/255 * torch.sign(delta.grad.data)
        delta.data = torch.min(torch.max(delta,audio-eps),audio+eps)
        delta.data = torch.min(torch.max(delta,-audio-1),-audio+1)
        assert (audio+delta).min() >= -1
        assert (audio + delta).max() <= 1
        zero_gradients(delta)
        # adv_audio += delta
        # print(f"predicted class: {yhat.argmax()}, correct class: {y.item()}, loss: {loss.item()}")
    total += 1
    correct += int(yhat.argmax() == y.item())

    print(f"PGD accuracy {correct*100/total} ({correct}/{total})")
    return audio+delta.detach(), sr

def attack(dataloader, example_idx, model, n_iter=10, eps=0.01, sign=True, save = False):
    x, y = dataloader.dataset.spects[example_idx]
    adv_wav, sr = attack_wav(dataloader, x, y, model, n_iter, eps, sign)
    if save:
        wav_name = x.split("/")[-1].split(".")[0]
        librosa.output.write_wav(f"{wav_name}_adv.wav", adv_wav.numpy().transpose(), sr)


def get_pred(dataloader, w_path, model):
    audio, sr = librosa.load(w_path, sr=None)
    audio = np.expand_dims(audio, 0)
    audio = torch.FloatTensor(audio)
    spect, phase = dataloader.dataset.stft.transform(audio)
    spect = spect.unsqueeze(0).cuda()
    return model(spect)

def attack_no_loader(dataloader, model, n_iter=10, eps=0.01, sign=True):
    # path = "/data/ronic/kalman_wav"
    correct = 0

    wavs = [f for f in os.listdir("/data/ronic/kalman_wav") if ".wav" in f]
    total = 0
    non_skip_total = 0
    wav_2_class = {}
    ignore = []
    with open("idxs", "rb") as f:
        idxs = pickle.load(f)
    for i in idxs:
        name = dataloader.dataset.spects[i][0].split("/test/")[1].split("/")[1]
        if name in wav_2_class:
            ignore.append(name)
        wav_2_class[name] = (dataloader.dataset.spects[i][1],dataloader.dataset.spects[i][0])
    # wavs = [f"{path}/{f}" for f in os.listdir(path) if ".wav" in f and "sil" not in f]+["0c40e715_nohash_0_adv.wav"]
    for w in wavs:
        name = w.split("_Q")[0]+".wav"
        if name in ignore:
            continue
        if name not in wav_2_class.keys():
            continue
        label,w_path = wav_2_class[name]
        yhat = get_pred(dataloader, w_path, model)
        # print(yhat.argmax().item())
        # print(yhat.argmax().item() == label)
        if yhat.argmax().item() != label:
            non_skip_total+=1
            continue
        yhat = get_pred(dataloader, "/data/ronic/kalman_wav/"+w, model)
        total +=1
        non_skip_total+=1
        correct += int(yhat.argmax().item() == label)
        # attack_wav(dataloader, w, 0, model,n_iter, eps,sign)
    print(f"correct {correct*100/total} ({correct})/({total})")