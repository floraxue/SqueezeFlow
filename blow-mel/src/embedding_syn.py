import torch
import numpy as np
import os

from utils import datain, utils


def main():
    args = torch.load('/rscratch/xuezhenruo/blow_vctk/blow_emb_0508_pc80/ckpt_20000.pt')
    del args['model_state_dict']
    del args['optimizer_state_dict']
    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    args = Namespace(**args)

    blow_model = torch.load('/rscratch/xuezhenruo/blow_vctk/blow_emb_0508_pc80/ckpt_20000.model.pt',
                            map_location='cuda')
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

    speakers = dataset_test.speakers
    # lspeakers = list(speakers.keys())
    lspeakers = ['p248'] * 20
    target_speakers = lspeakers
    filenames = dataset_test.filenames[:len(target_speakers)]

    # Synthesize
    path_out = os.path.join(args.base_fn_out, 'syn_manual', 'ckpt_{}_p248'.format(20000))
    os.makedirs(path_out, exist_ok=True)
    print('Saving {} audios to {}'.format(len(target_speakers), path_out))
    for k, (x, info) in enumerate(testloader):
        if k >= len(target_speakers):
            break
        target_spk = target_speakers[k]
        itarget = torch.LongTensor([speakers[target_spk]])
        fn = filenames[k][:-3]

        synthesized_x = utils.synthesize_one_audio(x, info, itarget,
                                                   filename=fn,
                                                   target_spk=target_spk,
                                                   blow=blow_model,
                                                   path_out=path_out,
                                                   sw_model=squeezewave,
                                                   convert=True,
                                                   sr=22050,
                                                   normalize=False,
                                                   print_mel=True)


if __name__ == '__main__':
    main()
