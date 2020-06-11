import pdb
import sys
from os import path

import numpy as np
import torch
from scipy.io import wavfile

from utils import vocoder
from utils.SqueezeWave.TacotronSTFT import TacotronSTFT


########################################################################################################################

def get_mel(audio):
    stft = TacotronSTFT(filter_length=1024,
                        hop_length=256,
                        win_length=1024,
                        sampling_rate=22050,
                        mel_fmin=0.0, mel_fmax=8000.0)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec = stft.mel_spectrogram(audio)
    return melspec


########################################################################################################################

def synthesize_one_audio(x, info, itarget, filename,
                         target_spk, blow, path_out,
                         sw_model,
                         print_mel=False,
                         convert=True, device='cuda', sr=22050,
                         normalize=True, xmax=0.98,
                         target_emb=None
                         ):
    # import pdb
    # pdb.set_trace()
    x = get_mel(x)
    isource = info[:, 3]

    # Convert
    if convert:
        # Forward & reverse
        x = x.to(device)
        isource = isource.to(device)
        itarget = itarget.to(device)
        z = blow.forward(x, isource)[0]
        if target_emb != None:
            original_emb = blow.embedding.weight.data[0]
            blow.embedding.weight.data[0] = target_emb
        x = blow.reverse(z, itarget)
        if target_emb != None:
            blow.embedding.weight.data[0] = original_emb
        x = x.cpu()

    if print_mel:
        # Hacking to print mel_syn here
        _, mel_fname = path.split(filename)  # p285/p285_04452
        if convert:
            mel_fname += '_to_' + target_spk
        mel_fname = path.join(path_out,
                              "{}_mel.pt".format(mel_fname))
        torch.save(x[0], mel_fname)
        # print("Saved mel to {}".format(mel_fname))
        ##

    # Vocoder Inference
    x = vocoder.infer(mel=x,
                      squeezewave=sw_model)
    x = x.cpu()

    # Filename
    _, filename = path.split(filename)
    if convert:
        filename += '_to_' + target_spk
    filename = path.join(path_out, filename + '.wav')

    # Synthesize
    # Refer to audioutils.synthesize
    x = x.squeeze().numpy().astype(np.float32)
    # Normalize
    if normalize:
        neginf_ps = np.isneginf(x)
        posinf_ps = np.isposinf(x)
        x[neginf_ps] = np.nan
        x[posinf_ps] = np.nan
        x -= np.nanmean(x)
        mx = np.nanmax(np.abs(x))
        if mx > 0:
            x *= xmax / mx
        x[neginf_ps] = -xmax
        x[posinf_ps] = xmax
        x[np.isnan(x)] = 0
    else:
        x = np.clip(x, -xmax, xmax)
        x[np.isnan(x)] = xmax
    # To 16 bit & save
    wavfile.write(filename, sr, np.array(x * 32767, dtype=np.int16))
    # return x as float
    return x


########################################################################################################################

def timer(start, end):
    days, rem = divmod(end - start, 3600 * 24)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    # return '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(minutes),seconds)
    return '{:0>2}:{:0>2}:{:0>2}:{:0>2}'.format(int(days), int(hours),
                                                int(minutes), int(seconds))


########################################################################################################################

class FIFOFixed(object):
    def __init__(self, l):
        self.data = l[:]
        return

    def push(self, v):
        self.data.append(v)
        return self.data.pop(0)

    def upperbound(self, factor=2):
        return np.mean(self.data) + factor * np.std(self.data)


########################################################################################################################

def print_arguments(args):
    print('=' * 100)
    print('Arguments =')
    aux = vars(args)
    tmp = list(aux.keys())
    tmp.sort()
    for arg in tmp:
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)
    return


def print_model_report(model, verbose=3):
    if verbose > 1:
        print(model)
    if verbose > 2:
        print('Dimensions =', end=' ')
    count = 0
    for p in model.parameters():
        if verbose > 2:
            print(p.size(), end=' ')
        count += np.prod(p.size())
    if verbose > 2:
        print()
    if verbose > 0:
        print('Num parameters = %s' % (human_format(count)))
    return count


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


########################################################################################################################

def repackage_hidden(h):
    if h is None:
        return None
    if isinstance(h, list):
        return list(repackage_hidden(v) for v in h)
    elif isinstance(h, tuple):
        return tuple(repackage_hidden(v) for v in h)
    return h.detach()


########################################################################################################################

TWOPI = 2 * np.pi


def get_timecode(dim, t, tframe, size=None, maxlen=10000, collapse=False):
    if size is None: size = tframe
    n = t.float().view(-1, 1, 1) + torch.linspace(0, tframe - 1,
                                                  steps=size).view(1, 1,
                                                                   -1).to(
        t.device)
    f = (10 ** (torch.arange(1, dim + 1).float() / dim)).view(1, -1, 1).to(
        t.device)
    tc = torch.sin(TWOPI * f * n / maxlen)
    if collapse:
        tc = tc.mean(1).unsqueeze(1)
    return tc


"""
def get_timecode(dim,t,size,fmin=30,fmax=330,fs=16000):
    samples=t.float().view(-1,1,1)+torch.arange(size).to(t.device).float().view(1,1,-1)
    freqs=torch.logspace(np.log10(fmin).item(),np.log10(fmax).item(),steps=dim).to(t.device).view(1,-1,1)
    signal=torch.sin(TWOPI*freqs*samples/fs)
    return signal
#"""


########################################################################################################################

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return


########################################################################################################################

HALFLOGTWOPI = 0.5 * np.log(2 * np.pi).item()


def gaussian_log_p(x, mu=None, log_sigma=None):
    if mu is None or log_sigma is None:
        return -HALFLOGTWOPI - 0.5 * (x ** 2)
    return -HALFLOGTWOPI - log_sigma - 0.5 * ((x - mu) ** 2) / torch.exp(
        2 * log_sigma)


def gaussian_sample(x, mu=None, log_sigma=None):
    if mu is None or log_sigma is None:
        return x
    return mu + torch.exp(log_sigma) * x


def disclogistic_log_p(x, mu=0, sigma=1, eps=1e-12):
    xx = (x - mu) / sigma
    return torch.log(torch.sigmoid(xx + 0.5) - torch.sigmoid(xx - 0.5) + eps)


########################################################################################################################

def loss_flow_nll(z, log_det):
    # size of: z = sbatch * n_mel_channels * lchunk
    #          log_det = sbatch
    _, n_mel_channels, lchunk = z.size()
    size = n_mel_channels * lchunk

    log_p = gaussian_log_p(z).sum(2).sum(1)

    nll = -log_p - log_det

    log_det /= size
    log_p /= size
    nll /= size

    log_det = log_det.mean()
    log_p = log_p.mean()
    nll = nll.mean()

    """
    # Sanity check
    if torch.isnan(nll) or nll>1000:
        print('\n***** EXIT: Wrong value in loss (log_p={:.2f},log_det={:.2f}) *****'.format(log_p.item(),log_det.item()))
        sys.exit()
    #"""

    return nll, np.array([nll.item(), log_p.item(), log_det.item()],
                         dtype=np.float32)


########################################################################################################################

def save_stuff(basename, report=None, args=None, model=None, optim=None):
    # Report
    if report is not None:
        torch.save(report, basename + '.report.pt')
    if args is not None:
        torch.save(args, basename + '.args.pt')
    # Model & optim
    if model is not None:
        try:
            torch.save(model.module, basename + '.model.pt')
        except:
            torch.save(model, basename + '.model.pt')
    if optim is not None:
        torch.save(optim, basename + '.optim.pt')
    return


def load_stuff(basename, device='cpu'):
    try:
        report = torch.load(basename + '.report.pt', map_location=device)
    except:
        report = None
    try:
        args = torch.load(basename + '.args.pt', map_location=device)
    except:
        args = None
    try:
        model = torch.load(basename + '.model.pt', map_location=device)
    except:
        model = None
    try:
        optim = torch.load(basename + '.optim.pt', map_location=device)
    except:
        optim = None
    return report, args, model, optim


########################################################################################################################

def pairwise_distance_matrix(x, y=None, eps=1e-10):
    x_norm = x.pow(2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = y.pow(2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2 * torch.mm(x, y.t().contiguous())
    if y is None:
        dist -= torch.diag(dist.diag())
    return torch.clamp(dist, eps, np.inf)


########################################################################################################################

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        return grad

    @staticmethod
    def reverse(ctx, x):
        return x + torch.rand_like(x) - 0.5

########################################################################################################################
