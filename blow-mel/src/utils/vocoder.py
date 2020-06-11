import torch
from utils.SqueezeWave.denoiser import Denoiser


def infer(mel, squeezewave,
          sigma=1.0, is_fp16=False, denoiser_strength=0.03):
    if is_fp16:
        from apex import amp
        squeezewave, _ = amp.initialize(squeezewave, [], opt_level="O3")

    if denoiser_strength > 0:
        denoiser = Denoiser(squeezewave).cuda()

    mel = torch.autograd.Variable(mel.cuda())
    mel = mel.half() if is_fp16 else mel
    with torch.no_grad():
        audio = squeezewave.infer(mel, sigma=sigma).float()
        if denoiser_strength > 0:
            audio = denoiser(audio, denoiser_strength)
            audio = audio.squeeze(1)
        # audio = audio * MAX_WAV_VALUE
    return audio


# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-f', "--filelist_path", required=True)
#     parser.add_argument('-w', '--squeezewave_path',
#                         help='Path to squeezewave decoder checkpoint with model')
#     parser.add_argument('-o', "--output_dir", required=True)
#     parser.add_argument("-s", "--sigma", default=1.0, type=float)
#     parser.add_argument("--sampling_rate", default=22050, type=int)
#     parser.add_argument("--is_fp16", action="store_true")
#     parser.add_argument("-d", "--denoiser_strength", default=0.0, type=float,
#                         help='Removes model bias. Start with 0.1 and adjust')
#
#     args = parser.parse_args()
#
#     main(args.filelist_path, args.squeezewave_path, args.sigma, args.output_dir,
#          args.sampling_rate, args.is_fp16, args.denoiser_strength)
