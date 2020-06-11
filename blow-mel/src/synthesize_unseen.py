import argparse
import os
import sys
import time
from copy import deepcopy
import pdb

import numpy as np
import torch
import torch.utils.data

from utils import audio as audioutils
from utils import datain
from utils import utils
from utils import vocoder

########################################################################################################################

# Arguments
parser = argparse.ArgumentParser(description='Audio synthesis script')
parser.add_argument('--seed_input', default=0, type=int, required=False,
                    help='(default=%(default)d)')
parser.add_argument('--seed', default=0, type=int, required=False,
                    help='(default=%(default)d)')
parser.add_argument('--device', default='cuda', type=str, required=False,
                    help='(default=%(default)s)')
# Data
parser.add_argument('--path_data_root', default='', type=str, required=False,
                    help='(default=%(default)s)')
parser.add_argument('--trim', default=-1, type=float, required=False,
                    help='(default=%(default)f)')
parser.add_argument('--adapted_base_fn_model', default='', type=str, required=True,
                    help='(default=%(default)s)')
parser.add_argument('--trained_base_fn_model', default='', type=str, required=True,
                    help='(default=%(default)s)')
parser.add_argument('--path_out', default='../res/', type=str, required=True,
                    help='(default=%(default)s)')
parser.add_argument('--split', default='test', type=str, required=False,
                    help='(default=%(default)s)')
parser.add_argument('--force_source_file', default='', type=str,
                    required=False, help='(default=%(default)s)')
parser.add_argument('--force_source_speaker', default='', type=str,
                    required=False, help='(default=%(default)s)')
parser.add_argument('--force_target_speaker', default='', type=str,
                    required=False, help='(default=%(default)s)')
# Conversion
parser.add_argument('--fn_list', default='', type=str, required=False,
                    help='(default=%(default)s)')
parser.add_argument('--sbatch', default=256, type=int, required=False,
                    help='(default=%(default)d)')
parser.add_argument('--convert', action='store_true')
parser.add_argument('--zavg', action='store_true', required=False,
                    help='(default=%(default)s)')
parser.add_argument('--alpha', default=3, type=float, required=False,
                    help='(default=%(default)f)')
# Synthesis
parser.add_argument('--lchunk', default=-1, type=int, required=False,
                    help='(default=%(default)d)')
parser.add_argument('--stride', default=-1, type=int, required=False,
                    help='(default=%(default)d)')
parser.add_argument('--synth_nonorm', action='store_true')
parser.add_argument('--maxfiles', default=10000000, type=int, required=False,
                    help='(default=%(default)d)')
parser.add_argument('--sw_path', default='/work/x/blow-mel/res/L128_large_pretrain',
                    type=str, required=False, help='(default=%(default)d)')

# Process arguments
args = parser.parse_args()
if args.trim <= 0:
    args.trim = None
if args.force_source_file == '':
    args.force_source_file = None
if args.force_source_speaker == '':
    args.force_source_speaker = None
if args.force_target_speaker == '':
    args.force_target_speaker = None
if args.fn_list == '':
    args.fn_list = 'list_seed' + str(
        args.seed_input) + '_' + args.split + '.tsv'

# Print arguments
utils.print_arguments(args)

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.device == 'cuda':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(args.seed)

########################################################################################################################

# Load model, pars, & check

print('Load stuff')
checkpoint = torch.load(args.trained_base_fn_model + '.pt')
_, _, adapted_blow_model, _ = utils.load_stuff(args.adapted_base_fn_model)
_, _, trained_blow_model, _ = utils.load_stuff(args.trained_base_fn_model)
pars = checkpoint.copy()
del pars['model_state_dict']
del pars['optimizer_state_dict']
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
pars = Namespace(**pars)
try:
    losses_train, losses_valid, losses_test = np.vstack(
        pars.losses['train']), np.vstack(pars.losses['valid']), \
                                              pars.losses['test']
    print('Best losses = ', np.min(losses_train, axis=0),
          np.min(losses_valid, axis=0), losses_test)
except:
    print('[Could not load losses]')
print('-' * 100)
adapted_blow_model = adapted_blow_model.to(args.device)
trained_blow_model = trained_blow_model.to(args.device)
utils.print_model_report(trained_blow_model, verbose=1)
print('-' * 100)

# Check lchunk & stride
if args.lchunk <= 0:
    args.lchunk = pars.lchunk
if args.stride <= 0:
    # args.stride = args.lchunk // 2
    args.stride = args.lchunk
# if pars.model != 'blow' and not pars.model.startswith('test'):
#     if not args.zavg:
#         print(
#             '[WARNING: You are not using Blow. Are you sure you do not want to use --zavg?]')
#     if args.lchunk != pars.lchunk:
#         args.lchunk = pars.lchunk
#         print(
#             '[WARNING: ' + pars.model + ' model can only operate with same frame size as training. It has been changed, but you may want to change the stride now]')
if args.stride == args.lchunk:
    print('[Synth with 0% overlap]')
    window = torch.ones(args.lchunk)
elif args.stride == args.lchunk // 2:
    print('[Synth with 50% overlap]')
    window = torch.hann_window(args.lchunk)
elif args.stride == args.lchunk // 4 * 3:
    print('[Synth with 25% overlap]')
    window = torch.hann_window(args.lchunk // 2)
    window = torch.cat(
        [window[:len(window) // 2], torch.ones(args.lchunk // 2),
         window[len(window) // 2:]])
else:
    print(
        '[WARNING: No specific overlap strategy. Forcing Hann window and normalize]')
    window = torch.hann_window(args.lchunk)
    args.synth_nonorm = False
window = window.view(1, -1)

print('-' * 100)

########################################################################################################################

# Data
print('Load VCTK_22kHz_adapt test split speakers')
adapt_data_path = os.path.join(args.path_data_root, 'VCTK_22kHz_10')
dataset = datain.DataSet(adapt_data_path, pars.lchunk, pars.stride,
                         sampling_rate=pars.sr, split='test',
                         seed=pars.seed, do_audio_load=False)
speakers = deepcopy(dataset.speakers)
lspeakers = list(speakers.keys())
print('Adapting to these speakers:')
print(lspeakers)

# Input data
print('Load VCTK_22kHz_train test split audio')
train_data_path = os.path.join(args.path_data_root, 'VCTK_22kHz_98')
dataset = datain.DataSet(train_data_path, args.lchunk, args.stride,
                         sampling_rate=pars.sr, split='test',
                         trim=args.trim,
                         select_speaker=args.force_source_speaker,
                         select_file=args.force_source_file,
                         seed=pars.seed)
loader = torch.utils.data.DataLoader(dataset,
                                     batch_size=1,
                                     shuffle=False, num_workers=0)
fnlist = dataset.filenames

########################################################################################################################

# Prepare model
try:
    # adapted_blow_model.precalc_matrices('on')
    trained_blow_model.precalc_matrices('on')
except:
    pass
adapted_blow_model.eval()
trained_blow_model.eval()
# embeddings_source = trained_blow_model.embedding.weight.data
embeddings_target = adapted_blow_model.embedding.weight.data
print('-' * 100)

# Synthesis loop
# Each loop synthesizes one full audio
squeezewave = torch.load(args.sw_path)['model']
squeezewave = squeezewave.remove_weightnorm(squeezewave)
squeezewave.cuda().eval()
print('Synth')
t_synth = 0
try:
    with torch.no_grad():
        for k, (x, info) in enumerate(loader):

            isource = info[:, 3]
            target_spk = np.random.choice(np.array(lspeakers), 1)[0]
            itarget = speakers[target_spk]
            target_emb = embeddings_target[itarget]
            itarget = torch.LongTensor([0])  # util func assumes target_emb at index 0

            # Track time
            tstart = time.time()

            _, model_fname = os.path.split(args.adapted_base_fn_model)
            path_out = os.path.join(args.path_out, 'syn_manual_unseen',model_fname)
            os.makedirs(path_out, exist_ok=True)

            synthesized_x = utils.synthesize_one_audio(x, info, itarget,
                                                       filename=fnlist[k].split('.')[0],
                                                       target_spk=target_spk,
                                                       blow=trained_blow_model,
                                                       path_out=path_out,
                                                       sw_model=squeezewave,
                                                       print_mel=True,
                                                       convert=args.convert,
                                                       sr=pars.sr,
                                                       normalize=False,
                                                       target_emb=target_emb)

            # Track time
            t_synth += time.time() - tstart
except KeyboardInterrupt:
    print()

########################################################################################################################

# Report times
print('-' * 100)
print('Time')
print('t_synth = {}'.format(t_synth))
print('-' * 100)

# Done
if args.convert:
    print('*** Conversions done ***')
else:
    print('*** Original audio. No conversions done ***')
