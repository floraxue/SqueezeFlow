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
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import os
import json
from copy import deepcopy
from scipy.io.wavfile import write
import numpy as np
import torch
from torch.utils.data import DataLoader

from mel2samp import Mel2Samp
from denoiser import Denoiser
from TacotronSTFT import TacotronSTFT


# Utility functions
def create_reverse_dict(inp):
    reverse = {}
    for k, v in inp.items():
        assert v not in reverse
        reverse[v] = k
    return reverse


def save_audio_chunks(frames, filename, stride, sr=22050, ymax=0.98,
                      normalize=True):
    # Generate stream
    y = torch.zeros((len(frames) - 1) * stride + len(frames[0]))
    for i, x in enumerate(frames):
        y[i * stride:i * stride + len(x)] += x
    # To numpy & deemph
    y = y.numpy().astype(np.float32)
    # if deemph>0:
    #     y=deemphasis(y,alpha=deemph)
    # Normalize
    if normalize:
        y -= np.mean(y)
        mx = np.max(np.abs(y))
        if mx > 0:
            y *= ymax / mx
    else:
        y = np.clip(y, -ymax, ymax)
    # To 16 bit & save
    write(filename, sr, np.array(y * 32767, dtype=np.int16))
    return y


def get_mel(audio):
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec = stft.mel_spectrogram(audio)
    return melspec


def main(squeezewave_path, sigma, output_dir, is_fp16,
         denoiser_strength):
    # mel_files = files_to_list(mel_files)
    squeezewave = torch.load(squeezewave_path)['model']
    squeezewave = squeezewave.remove_weightnorm(squeezewave)
    squeezewave.cuda().eval()
    if is_fp16:
        from apex import amp
        squeezewave, _ = amp.initialize(squeezewave, [], opt_level="O3")

    if denoiser_strength > 0:
        denoiser = Denoiser(squeezewave).cuda()

    n_audio_channel = squeezewave_config["n_audio_channel"]
    testset = Mel2Samp(n_audio_channel, frame_energy_thres=0.02, **data_config)
    # =====START: ADDED FOR DISTRIBUTED======
    # train_sampler = DistributedSampler(trainset) if num_gpus > 1 else None
    # =====END:   ADDED FOR DISTRIBUTED======
    test_loader = DataLoader(testset, num_workers=0, shuffle=False,
                             # sampler=train_sampler,
                             batch_size=1 if data_config['split'] == 'test' else 12,
                             pin_memory=False,
                             drop_last=True)

    speakers_to_sids = deepcopy(testset.speakers)
    sids_to_speakers = create_reverse_dict(speakers_to_sids)
    ut_to_uids = deepcopy(testset.utterances)
    uids_to_ut = create_reverse_dict(ut_to_uids)

    # sid_target = np.random.randint(len(speakers_to_sids))
    # speaker_target = sids_to_speakers[sid_target]
    # sid_target = torch.LongTensor([[sid_target] *
    #                                test_loader.batch_size]).view(
    #     test_loader.batch_size, 1).to('cuda')

    audios = []
    mels = []
    n_audios = 0
    for i, batch in enumerate(test_loader):
        audio_source, sid_source, uid_source, is_last = batch
        mel_source = get_mel(audio_source)
        mel_source = mel_source.to('cuda')
        import pdb
        pdb.set_trace()

        with torch.no_grad():
            predicted = squeezewave.infer(mel_source, sigma=sigma)
            if denoiser_strength > 0:
                predicted = denoiser(predicted, denoiser_strength)
                predicted = predicted.squeeze(1)
            # predicted = predicted * MAX_WAV_VALUE

        for j in range(len(predicted)):
            p = predicted[j].cpu()
            audios.append(p)
            mels.append(mel_source[j].cpu())
            speaker_source = sids_to_speakers[sid_source[j].data.item()]
            ut_source = uids_to_ut[uid_source[j].data.item()]
            last = is_last[j].data.item()
            if last:
                ## Hacking to print mel_source here
                fname = os.path.join(output_dir,
                                     "{}_{}_mel.pt".format(speaker_source, ut_source))
                pdb.set_trace()
                torch.save(mels, fname)
                print("Saved mel to {}".format(fname))
                ##

                # audio_path = os.path.join(
                #     output_dir,
                #     "{}_{}_to_{}_synthesis.wav".format(speaker_source,
                #                                        ut_source,
                #                                        speaker_target))
                audio_path = os.path.join(
                    output_dir,
                    "{}_{}_synthesis.wav".format(speaker_source,
                                                       ut_source))
                print("Synthesizing file No.{} at {}".format(n_audios,
                                                             audio_path))
                save_audio_chunks(audios, audio_path, data_config['stride'],
                                  data_config['sampling_rate'])

                audios = []
                mels = []
                n_audios += 1


    # for i, file_path in enumerate(mel_files):
    #     file_name = os.path.splitext(os.path.basename(file_path))[0]
    #     mel = torch.load(file_path)
    #     mel = torch.autograd.Variable(mel.cuda())
    #     mel = torch.unsqueeze(mel, 0)
    #     mel = mel.half() if is_fp16 else mel
    #     with torch.no_grad():
    #         audio = squeezewave.infer(mel, sigma=sigma).float()
    #         if denoiser_strength > 0:
    #             audio = denoiser(audio, denoiser_strength)
    #         audio = audio * MAX_WAV_VALUE
    #     audio = audio.squeeze()
    #     audio = audio.cpu().numpy()
    #     audio = audio.astype('int16')
    #     audio_path = os.path.join(
    #         output_dir, "{}_synthesis.wav".format(file_name))
    #     write(audio_path, sampling_rate, audio)
    #     print(audio_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='JSON file for configuration')
    parser.add_argument('-w', '--squeezewave_path', required=True,
                        help='Path to squeezewave decoder checkpoint with model')
    parser.add_argument('-o', "--output_dir", required=True)
    parser.add_argument("-s", "--sigma", default=1.0, type=float)
    # parser.add_argument("--sampling_rate", default=22050, type=int)
    parser.add_argument("--is_fp16", action="store_true")
    parser.add_argument("-d", "--denoiser_strength", default=0.0, type=float,
                        help='Removes model bias. Start with 0.1 and adjust')

    args = parser.parse_args()
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    global data_config
    data_config = config["data_config"]
    data_config['split'] = 'train'
    global squeezewave_config
    squeezewave_config = config['squeezewave_config']

    stft = TacotronSTFT(filter_length=data_config['filter_length'],
                        hop_length=data_config['hop_length'],
                        win_length=data_config['win_length'],
                        sampling_rate=data_config['sampling_rate'],
                        mel_fmin=data_config['mel_fmin'],
                        mel_fmax=data_config['mel_fmax'],
                        n_group=squeezewave_config['n_audio_channel'])

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    main(args.squeezewave_path, args.sigma, args.output_dir,
         args.is_fp16, args.denoiser_strength)
