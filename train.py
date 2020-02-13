from __future__ import print_function
import torch.nn.functional as F
from torch.autograd import Variable
import librosa
import torch
import numpy as np
from torch.autograd.gradcheck import zero_gradients


def train(loader, model, optimizer, epoch, cuda, log_interval, verbose=True):
    model.train()
    global_epoch_loss = 0
    for batch_idx, (data, target) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        global_epoch_loss += loss.item()
        if verbose:
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(loader.dataset), 100.
                    * batch_idx / len(loader), loss.item()))
    return global_epoch_loss / len(loader.dataset)


def test(loader, model, cuda, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(loader.dataset)
    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
    return test_loss

def attack(dataloader, example_idx, model, n_iter=10, eps=0.1, sign=True):
    model.eval()
    x, y = dataloader.dataset.spects[example_idx]
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
        if sign:
            delta.data = delta + eps * torch.sign(delta.grad.data)
        else:
            delta.data = delta + eps * delta.grad.data
        zero_gradients(delta)
        print(f"predicted class: {yhat.argmax()}, correct class: {y.item()}, loss: {loss.item()}")

