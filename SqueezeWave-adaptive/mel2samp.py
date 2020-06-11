# We retain the copyright notice by NVIDIA from the original code. However, we
# we reserve our rights on the modifications based on the original code.
#
# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************\
import os
import time
import random
import argparse
import json
import torch
import torch.utils.data
import sys
from scipy.io.wavfile import read
import numpy as np

# We're using the audio processing from TacoTron2 to make sure it matches
from TacotronSTFT import TacotronSTFT

MAX_WAV_VALUE = 32768.0
EXTENSION = '.pt'


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate


class Mel2Samp(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, n_audio_channel, path_in, split, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax,
                 stride, temp_jitter=False, store_in_ram=False, seed=0,
                 split_utterances=True, pc_split_utterances=0.1,
                 split_speakers=False, pc_split_speakers=0.1,
                 frame_energy_thres=0.025, do_audio_load=True,
                 trim=None, select_speaker=None, select_file=None,
                 verbose=True):
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax,
                                 n_group=n_audio_channel)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate

        # Temp Jitter may be True when doing data aug in training
        self.path_in = path_in
        self.split = split
        self.lchunk = segment_length
        self.stride = stride
        self.temp_jitter = temp_jitter
        self.store_in_ram = store_in_ram
        if trim is None or trim <= 0:
            trim = np.inf

        # Get filenames in folder and subfolders
        self.filenames = []
        for dirpath, dirnames, filenames in os.walk(self.path_in):
            for fn in filenames:
                if not fn.endswith(EXTENSION): continue
                new_fn = os.path.join(dirpath, fn)
                new_fn = os.path.relpath(new_fn, self.path_in)
                self.filenames.append(new_fn)
        self.filenames.sort()
        random.seed(seed)
        random.shuffle(self.filenames)

        # Get speakers & utterances
        self.speakers = {}
        self.utterances = {}
        for fullfn in self.filenames:
            spk, ut = self.filename_split(fullfn)
            if spk not in self.speakers:
                self.speakers[spk] = len(self.speakers)
            if ut not in self.utterances:
                self.utterances[ut] = len(self.utterances)
        self.n_max_speakers = len(self.speakers)

        # Split
        lutterances = list(self.utterances.keys())
        lutterances.sort()
        random.shuffle(lutterances)
        lspeakers = list(self.speakers.keys())
        lspeakers.sort()
        random.shuffle(lspeakers)
        isplit_ut = int(len(lutterances) * pc_split_utterances)
        isplit_spk = int(len(lspeakers) * pc_split_speakers)
        if split == 'train':
            spk_del = lspeakers[-2 * isplit_spk:]
            ut_del = lutterances[-2 * isplit_ut:]
        elif split == 'valid':
            spk_del = lspeakers[:-2 * isplit_spk] + lspeakers[-isplit_spk:]
            ut_del = lutterances[:-2 * isplit_ut] + lutterances[-isplit_ut:]
        elif split == 'train+valid':
            spk_del = lspeakers[-isplit_spk:]
            ut_del = lutterances[-isplit_ut:]
        elif split == 'test':
            spk_del = lspeakers[:-isplit_spk]
            ut_del = lutterances[:-isplit_ut]
        else:
            print('Not implemented split', split)
            sys.exit()
        if split_speakers:
            for spk in spk_del:
                del self.speakers[spk]
        if split_utterances:
            for ut in ut_del:
                del self.utterances[ut]

        # Filter filenames by speaker and utterance
        filenames_new = []
        for filename in self.filenames:
            spk, ut = self.filename_split(filename)
            if spk in self.speakers and ut in self.utterances:
                filenames_new.append(filename)
        self.filenames = filenames_new

        # Select speaker
        if select_speaker is not None:
            select_speaker = select_speaker.split(',')
            filenames_new = []
            for filename in self.filenames:
                spk, ut = self.filename_split(filename)
                if spk in select_speaker and spk in self.speakers:
                    filenames_new.append(filename)
            if len(filenames_new) == 0:
                print('\nERROR: Selected an invalid speaker. Options are:',
                      list(self.speakers.keys()))
                sys.exit()
            self.filenames = filenames_new

        # Select specific file
        if select_file is not None:
            select_file = select_file.split(',')
            filenames_new = []
            for filename in self.filenames:
                _, file = os.path.split(filename[:-len(EXTENSION)])
                if file in select_file:
                    filenames_new.append(filename)
            if len(filenames_new) == 0:
                print('\nERROR: Selected an invalid file. Options are:',
                      self.filenames[:int(np.min([50, len(self.filenames)]))],
                      '... (without folder and without extension))')
                sys.exit()
            self.filenames = filenames_new

        # Indices!
        self.audios = [None] * len(self.filenames)
        self.indices = []
        duration = {}
        start = time.time()
        if do_audio_load:
            for i, filename in enumerate(self.filenames):
                if verbose:
                    if i % 1000 == 0:
                        print('Read {} out of {} files'.
                              format(i + 1, len(self.filenames)))
                    # print('\rRead audio {:5.1f}%'.format(
                    #     100 * (i + 1) / len(self.filenames)), end='')
                # Info
                spk, ut = self.filename_split(filename)
                ispk, iut = self.speakers[spk], self.utterances[ut]
                # Load
                if spk not in duration:
                    duration[spk] = 0
                if duration[spk] >= trim:
                    continue
                x = torch.load(os.path.join(self.path_in, filename))
                if self.store_in_ram:
                    self.audios[i] = x.clone()
                x = x.float()
                # Process
                for j in range(0, len(x), stride):
                    if j + self.lchunk >= len(x):
                        continue
                    else:
                        xx = x[j:j + self.lchunk]
                    if (xx.pow(2).sum() / self.lchunk).sqrt().item() >= frame_energy_thres:
                        info = [i, j, 0, ispk, iut]
                        self.indices.append(torch.LongTensor(info))
                    duration[spk] += stride / sampling_rate
                    if duration[spk] >= trim:
                        break
                self.indices[-1][2] = 1
            if verbose:
                print()
            self.indices = torch.stack(self.indices)
        print("Time elapsed: {}".format(time.time() - start))

        # Print
        if verbose:
            totalduration = 0
            for key in duration.keys():
                totalduration += duration[key]
            print(
                'Loaded {:s}: {:.1f} h, {:d} spk, {:d} ut, {:d} files, {:d} frames (fet={:.1e},'.format(
                    split, totalduration / 3600, len(self.speakers),
                    len(self.utterances), len(self.filenames),
                    len(self.indices), frame_energy_thres), end='')
            if trim is None or trim > 1e12:
                print(' no trim)')
            else:
                print(' trim={:.1f}s)'.format(trim))
            if select_speaker is not None:
                print('Selected speaker(s):', select_speaker)
            if select_file is not None:
                print('Selected file(s):', select_file)

    def filename_split(self, fullfn):
        aux = os.path.split(fullfn)[-1][:-len(EXTENSION)].split('_')
        return aux[0], aux[1]

    def get_mel(self, audio):
        audio = audio.unsqueeze(0)
        audio = torch.autograd.Variable(audio, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def get_whole_audio(self, idx):
        # Load file
        if self.store_in_ram:
            x = self.audios[idx]
        else:
            x = torch.load(os.path.join(self.path_in, self.filenames[idx]))
        assert x is not None
        x = x.float()
        # Info
        spk, ut = self.filename_split(self.filenames[idx])
        ispk, iut = self.speakers[spk], self.utterances[ut]
        y = torch.LongTensor([ispk])
        ichap = torch.LongTensor([iut])
        last = torch.LongTensor([1])
        return x, y, ichap, last

    def __getitem__(self, index):
        if self.split == 'test':
            return self.get_whole_audio(index)
        i, j, last, ispk, ichap = self.indices[index, :]
        # Load file
        if self.store_in_ram:
            tmp = self.audios[i]
        else:
            tmp = torch.load(os.path.join(self.path_in, self.filenames[i]))
        # Temporal jitter
        if self.temp_jitter:
            j = j + np.random.randint(-self.stride // 2, self.stride // 2)
            if j < 0:
                j = 0
            elif j > len(tmp) - self.lchunk:
                j = np.max([0, len(tmp) - self.lchunk])
        # Get frame
        if j + self.lchunk > len(tmp):
            x = tmp[j:].float()
            x = torch.cat([x, torch.zeros(self.lchunk - len(x))])
        else:
            x = tmp[j:j + self.lchunk].float()
        # Get info
        y = torch.LongTensor([ispk])

        # mel = self.get_mel(x)

        return x, y, ichap, last

    def __len__(self):
        if self.split == 'test':
            return len(self.filenames)
        return self.indices.size(0)

# ===================================================================
# Takes directory of clean audio and makes directory of spectrograms
# Useful for making test sets
# ===================================================================
if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Output directory')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    data_config = config["data_config"]
    squeezewave_config = config["squeezewave_config"]
    mel2samp = Mel2Samp(squeezewave_config['n_audio_channel'], **data_config)

    filepaths = files_to_list(args.filelist_path)

    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    for filepath in filepaths:
        audio, sr = load_wav_to_torch(filepath)
        melspectrogram = mel2samp.get_mel(audio)
        filename = os.path.basename(filepath)
        new_filepath = args.output_dir + '/' + filename + '.pt'
        print(new_filepath)
        torch.save(melspectrogram, new_filepath)
