import os
from copy import deepcopy

import torch
import numpy as np
from scipy.io import wavfile

from utils import datain, utils, vocoder


def main():
    args = torch.load('/rscratch/xuezhenruo/blow_vctk/blow_emb_0508_pc80/ckpt_20000.pt')
    del args['model_state_dict']
    del args['optimizer_state_dict']
    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    args = Namespace(**args)
    device = 'cuda'
    xmax = 0.98
    sr = 22050

    blow_model = torch.load('/rscratch/xuezhenruo/blow_vctk/blow_emb_0508_pc80/ckpt_20000.model.pt',
                            map_location=device)
    blow_model.precalc_matrices('on')
    blow_model.eval()

    dataset_test = datain.DataSet(args.path_data, args.lchunk, args.stride,
                                  split='test', sampling_rate=args.sr,
                                  trim=args.trim,
                                  frame_energy_thres=args.frame_energy_thres,
                                  temp_jitter=False,
                                  seed=args.seed)
    testloader = torch.utils.data.DataLoader(dataset_test,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=args.nworkers)

    sw_path = '/rscratch/xuezhenruo/blow-mel/res/L128_large_pretrain'
    squeezewave = torch.load(sw_path)['model']
    squeezewave = squeezewave.remove_weightnorm(squeezewave)
    squeezewave.cuda().eval()

    filenames = dataset_test.filenames[:50]
    speakers = deepcopy(dataset_test.speakers)
    male_spk = 'p232'
    female_spk = 'p234'
    male_ispk = speakers[male_spk]
    female_ispk = speakers[female_spk]
    coeffs = torch.load('/rscratch/xuezhenruo/blow_vctk/blow_emb_0508_pc80/coeffs.pt')

    # Synthesize
    path_out = os.path.join(args.base_fn_out, 'syn_manual', 'ckpt_{}_unseen'.format(20000))
    os.makedirs(path_out, exist_ok=True)
    print('Saving {} audios to {}'.format(len(filenames), path_out))
    for k, (x, info) in enumerate(testloader):
        if k % 10 == 0:
            print('Saving', k)
        if k >= len(filenames):
            break
        itarget = torch.LongTensor([0])  # represents unseen speaker, emb will be inserted at speaker id 0
        fn = filenames[k][:-3]

        x = utils.get_mel(x)
        isource = info[:, 3]

        # Forward
        x = x.to(device)
        isource = isource.to(device)
        itarget = itarget.to(device)
        z = blow_model.forward(x, isource)[0]

        for alpha in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            new_coeff = alpha * coeffs[male_ispk] + (1 - alpha) * coeffs[female_ispk]
            new_coeff = new_coeff.unsqueeze(0)
            target_spk = 'unseen_alpha{}'.format(alpha)

            # Reverse
            original_coeff = blow_model.coeff.data[0]
            blow_model.coeff.data[0] = new_coeff
            mel_t = blow_model.reverse(z, itarget)
            blow_model.coeff.data[0] = original_coeff

            # Vocoder Inference
            x = vocoder.infer(mel=mel_t, squeezewave=squeezewave)
            x = x.cpu()
            x = x.squeeze().numpy().astype(np.float32)

            x = np.clip(x, -xmax, xmax)
            x[np.isnan(x)] = xmax

            # Save wav
            _, fname = os.path.split(fn)  # p285/p285_04452
            fname += '_to_' + target_spk
            save_fname = os.path.join(path_out,
                                      "{}.wav".format(fname))
            mel_fname = os.path.join(path_out,
                                     "{}_mel.pt".format(fname))

            wavfile.write(save_fname, sr, np.array(x * 32767, dtype=np.int16))
            torch.save(mel_t[0], mel_fname)


if __name__ == '__main__':
    main()

