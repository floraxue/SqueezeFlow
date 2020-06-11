import argparse
import time
import os
from copy import deepcopy

import torch
import numpy as np
from sklearn.decomposition import PCA
from numpy.testing import assert_array_almost_equal

from utils import utils
from utils import datain
from utils import audio as audioutils

from models import blow

########################################################################################################################

batch_data_augmentation = audioutils.DataAugmentation('cpu')


def batch_loop(model, optim, args, e, eval, dataset, loader, logger, lr,
               loss_best, losses_track,
               msg_pre='',
               exit_at_first_fwd=False):
    # Prepare
    if eval:
        model.eval()
    else:
        model.train()

    # Loop
    cum_losses = 0
    cum_num = 0
    msg_post = ''
    for b, (x, info) in enumerate(loader):

        # Prepare data
        if not eval and args.augment > 0:
            if args.augment > 1:
                x = batch_data_augmentation.noiseg(x, 0.001)
            x = batch_data_augmentation.emphasis(x, 0.2)
            x = batch_data_augmentation.magnorm_flip(x, 1)
            if args.augment > 1:
                x = batch_data_augmentation.compress(x, 0.1)

        # get mel
        x = utils.get_mel(x)

        s = info[:, 3].to(args.device)
        x = x.to(args.device)
        # Forward
        z, log_det = model.forward(x, s)
        loss, losses = utils.loss_flow_nll(z, log_det)

        """
        # Test reverse
        if e==0 and b==5:
            with torch.no_grad():
                model.eval()
                z,_=model.forward(x,s)
                xhat=model.reverse(z,s)
            dif=(x-xhat).abs()
            print()
            print(z[0].view(-1).cpu())
            print(xhat[0].view(-1).cpu())
            print('AvgDif =',dif.mean().item(),' MaxDif =',dif.max().item())
            print(dif.view(-1).cpu())
            sys.exit()
        #"""

        # Exit?
        if exit_at_first_fwd:
            return loss, losses, msg_pre

        # Backward
        if not eval:
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Report/print
        if eval:
            logger.add_scalar('eval_loss', loss, b + len(loader) * e)
        else:
            logger.add_scalar('train_loss', loss, b + len(loader) * e)
        cum_losses += losses * len(x)
        cum_num += len(x)
        msg = '\r| T = ' + utils.timer(tstart, time.time()) + ' | '
        msg += 'Epoch = {:3d} ({:5.1f}%) | '.format(e + 1, 100 * (
                b * args.sbatch + len(info)) / len(dataset))
        if eval:
            msg_post = 'Eval loss = '
        else:
            msg_post = 'Train loss = '
        for i in range(len(cum_losses)):
            msg_post += '{:7.4f} '.format(cum_losses[i] / cum_num)
        msg_post += '| '
        print(msg + msg_pre + msg_post, end='')

        if not eval:
            if (b + len(loader) * e) % 2000 == 0:
                fpath = os.path.join(args.base_fn_out,
                                     'ckpt_{}'.format(b + len(loader) * e))
                to_save = vars(args)
                try:
                    model_state_dict = model.module.state_dict()
                except:
                    model_state_dict = model.state_dict()
                to_save.update({
                    'epoch': e,
                    'iter': b + len(loader) * e,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optim.state_dict(),
                    'lr': lr,  # shadowing the original lr
                    'loss_best': loss_best,
                    'losses': losses_track,
                })
                torch.save(to_save, fpath + '.pt')
                torch.save(model.module, args.base_fn_out +
                           '/ckpt_{}.model.pt'.format(b + len(loader) * e))
                torch.save(optim, args.base_fn_out +
                           '/ckpt_{}.optim.pt'.format(b + len(loader) * e))
                make_audio_evals(dataset_test, loader_test, fpath,
                                 b + len(loader) * e, args.sw_path)

    cum_losses /= cum_num
    return cum_losses[0], cum_losses, msg_pre + msg_post


########################################################################################################################


def make_audio_evals(test_dataset, test_loader, model_fpath, iter, sw_path,
                     device='cuda'):

    # Make a blow model for eval
    _, _, blow_model, _ = utils.load_stuff(model_fpath, device)
    try:
        blow_model.precalc_matrices('on')
    except:
        pass
    blow_model.eval()

    squeezewave = torch.load(sw_path)['model']
    squeezewave = squeezewave.remove_weightnorm(squeezewave)
    squeezewave.cuda().eval()

    # Synthesize
    path_out = os.path.join(args.base_fn_out, 'syn', 'ckpt_{}'.format(iter))
    os.makedirs(path_out, exist_ok=True)
    print('Saving {} audios to {}'.format(len(filenames), path_out))
    for k, (x, info) in enumerate(test_loader):
        if k >= len(itrafos):
            break
        isource, itarget = itrafos[k]
        fn = 'ckpt_{}_'.format(iter) + filenames[k][:-3]
        target_spk = target_speakers[k]
        synthesized_x = utils.synthesize_one_audio(x, info, itarget,
                                                   filename=fn,
                                                   target_spk=target_spk,
                                                   blow=blow_model,
                                                   path_out=path_out,
                                                   sw_model=squeezewave,
                                                   convert=True,
                                                   sr=22050,
                                                   normalize=False)
        # Add audio to TB. audio is a 1D array within [-1, 1]
        log_name = os.path.split(fn)[1] + '_to_' + target_spk
        log_name = 'test_audio_ckpt_{}/{}'.format(iter, log_name)
        logger.add_audio(log_name, synthesized_x, iter, 22050)


def load_best_model():
    # checkpoint = torch.load(args.base_fn_out + '.pt')
    # model = blow.Model(**checkpoint)
    # model.load_state_dict(checkpoint['model_state_dict'])
    _, _, model, _ = utils.load_stuff(args.base_fn_out)
    model = model.to(args.device)
    if args.multigpu:
        model = torch.nn.DataParallel(model)
    return model


def get_optimizer(model, lr):
    if args.optim == 'adam':
        args.clearmomentum = True
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif args.optim == 'sgd':
        args.clearmomentum = True
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif args.optim == 'sgdm':
        args.clearmomentum = False
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.85)
    elif args.optim == 'adabound':
        import adabound
        args.clearmomentum = False
        return adabound.AdaBound(model.parameters(), lr=lr)
    return None


########################################################################################################################


def parse_args():

    parser = argparse.ArgumentParser(description='Embedding Test')
    parser.add_argument('--seed', default=0, type=int, required=False,
                        help='(default=%(default)d)')
    parser.add_argument('--device', default='cuda', type=str, required=False,
                        help='(default=%(default)s)')
    parser.add_argument('--nworkers', default=0, type=int, required=False,
                        help='(default=%(default)d)')
    # --- Data
    parser.add_argument('--path_data', default='', type=str, required=True,
                        help='(default=%(default)s)')
    parser.add_argument('--sr', default=22050, type=int, required=False,
                        help='(default=%(default)d)')
    parser.add_argument('--trim', default=-1, type=float, required=False,
                        help='(default=%(default)f)')
    parser.add_argument('--lchunk', default=16384, type=int, required=False,
                        help='(default=%(default)d)')
    parser.add_argument('--stride', default=-1, type=int, required=False,
                        help='(default=%(default)d)')
    parser.add_argument('--frame_energy_thres', default=0.025, type=float,
                        required=False, help='(default=%(default)f)')
    parser.add_argument('--augment', default=1, type=int, required=False,
                        help='(default=%(default)d)')
    parser.add_argument('--train_split', default='train', type=str, required=False,
                        help='(default=%(default)s)')
    # --- Optimization & training
    parser.add_argument('--nepochs', default=999, type=int, required=False,
                        help='(default=%(default)d)')
    parser.add_argument('--sbatch', default=256, type=int, required=False,
                        help='(default=%(default)d)')
    parser.add_argument('--optim', default='adam', type=str, required=False,
                        help='(default=%(default)s',
                        choices=['adam', 'sgd', 'sgdm', 'adabound'])
    parser.add_argument('--lr', default=1e-4, type=float, required=False,
                        help='(default=%(default)f)')
    parser.add_argument('--lr_thres', default=1e-4, type=float, required=False,
                        help='(default=%(default)f)')
    parser.add_argument('--lr_patience', default=10, type=int, required=False,
                        help='(default=%(default)d)')
    parser.add_argument('--lr_factor', default=0.2, type=float, required=False,
                        help='(default=%(default)f)')
    parser.add_argument('--lr_restarts', default=2, type=int, required=False,
                        help='(default=%(default)d)')
    parser.add_argument('--multigpu', action='store_true')
    # --- Model
    parser.add_argument('--model', type=str, default='blow', required=False,
                        help='(default=%(default)s)',
                        choices=['realnvp', 'glow', 'glow_wn', 'blow', 'blow2',
                                 'test1', 'test2', 'test3'])
    parser.add_argument('--load_existing', default='', type=str, required=False,
                        help='(default=%(default)s)')
    parser.add_argument('--nsqueeze', default=2, type=int, required=False,
                        help='(default=%(default)d)')
    parser.add_argument('--nblocks', default=8, type=int, required=False,
                        help='(default=%(default)d)')
    parser.add_argument('--nflows', default=12, type=int, required=False,
                        help='(default=%(default)d)')
    parser.add_argument('--ncha', default=480, type=int, required=False,
                        help='(default=%(default)d)')
    # --- Results
    parser.add_argument('--base_fn_out', default='', type=str, required=True,
                        help='(default=%(default)s)')
    parser.add_argument('--sw_path', default='/work/xuezhenruo/blow-mel/res/L128_large_pretrain',
                        type=str, required=False, help='(default=%(default)d)')

    args = parser.parse_args()

    # Process arguments
    if args.trim <= 0:
        args.trim = None
    if args.stride <= 0:
        args.stride = args.lchunk
    if args.multigpu:
        args.ngpus = torch.cuda.device_count()
        args.sbatch *= args.ngpus
    else:
        args.ngpus = 1
    utils.print_arguments(args)
    return args


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(args.seed)


def load_data(args):
    # Data sets and data loaders
    print('Data')
    dataset_train = datain.DataSet(args.path_data, args.lchunk, args.stride,
                                   split=args.train_split, sampling_rate=args.sr,
                                   trim=args.trim,
                                   frame_energy_thres=args.frame_energy_thres,
                                   temp_jitter=args.augment > 0,
                                   seed=args.seed)
    dataset_valid = datain.DataSet(args.path_data, args.lchunk, args.stride,
                                   split='valid', sampling_rate=args.sr,
                                   trim=args.trim,
                                   frame_energy_thres=args.frame_energy_thres,
                                   temp_jitter=False,
                                   seed=args.seed)
    dataset_test = datain.DataSet(args.path_data, args.lchunk, args.stride,
                                  split='test', sampling_rate=args.sr,
                                  trim=args.trim,
                                  frame_energy_thres=args.frame_energy_thres,
                                  temp_jitter=False,
                                  seed=args.seed)
    loader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=args.sbatch,
                                               shuffle=True, drop_last=True,
                                               num_workers=args.nworkers)
    loader_valid = torch.utils.data.DataLoader(dataset_valid,
                                               batch_size=args.sbatch,
                                               shuffle=False,
                                               num_workers=args.nworkers)
    loader_test = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=args.nworkers)

    print('-' * 100)
    return dataset_train, dataset_valid, dataset_test, loader_train, loader_valid, loader_test


def load_experiment(args):
    # load previous experiment
    assert args.load_existing != ''
    print('Load previous experiment')
    checkpoint = torch.load(args.load_existing + '.pt')
    _, _, model, _ = utils.load_stuff(args.load_existing, device='cuda')
    optim = get_optimizer(model, args.lr)
    optim.load_state_dict(checkpoint['optimizer_state_dict'])

    print('[Loaded model]')
    utils.print_model_report(model, verbose=1)
    iter = checkpoint['iter']
    epoch_start = checkpoint['epoch']

    loss_best = checkpoint['loss_best']
    print(
        '[Loaded report at iter {} with {:d} epochs; '
        'best validation was {:.2f}]'.format(
            iter, epoch_start, loss_best))

    args.load_existing = checkpoint['load_existing']
    args.multigpu = checkpoint['multigpu']
    args.optim = checkpoint['optim']
    args.seed = checkpoint['seed']
    args.lr = checkpoint['lr']
    print('New arguments')
    utils.print_arguments(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda': torch.cuda.manual_seed(args.seed)

    return model


def init_experiment():
    print('New {:s} model'.format(args.model))
    model = blow.Model(args.nsqueeze, args.nblocks, args.nflows,
                       args.ncha, args.ntargets,
                       base_embs=base_embs, use_coeff=True,
                       pc_dim=args.pc_dim)
    utils.print_model_report(model, verbose=1)
    model.to(args.device)

    optim = get_optimizer(model, args.lr)

    epoch_start = 0
    losses_track = {'train': [], 'valid': [], 'test': np.inf}
    loss_best = np.inf
    # Dry run to init the model
    print('Forward init')
    with torch.no_grad():
        batch_loop(model, optim, args, -1, False, dataset_train, loader_train,
                   logger, args.lr, loss_best, losses_track,
                   exit_at_first_fwd=True)

    return model, optim, epoch_start, losses_track, loss_best


def get_transformations():
    # Hacking to get the top 20 of the transformation list with default seed
    # Refer to synthesize.py get transformation list
    speakers = dataset_test.speakers
    lspeakers = list(speakers.keys())
    filenames = dataset_test.filenames[:20]
    target_speakers = []
    for i in range(20):
        t_spk = lspeakers[np.random.randint(len(speakers))]
        target_speakers.append(t_spk)
    assert len(filenames) == len(target_speakers)
    # Make tranformation list used in batch_loop eval
    itrafos = []
    for fn, t_spk in zip(filenames, target_speakers):
        _, fn = os.path.split(fn)  # p285/p285_04452.pt
        fn = fn[:-3]
        s_spk, ut = dataset_test.filename_split(fn)
        isrc, itgt = speakers[s_spk], speakers[t_spk]
        isrc = torch.LongTensor([isrc])
        itgt = torch.LongTensor([itgt])
        itrafos.append([isrc, itgt])
    return filenames, target_speakers, itrafos


def train(model, optim, epoch_start, losses_track, loss_best):
    print('Train')
    lr = args.lr
    patience = args.lr_patience
    restarts = args.lr_restarts
    epoch_offset = max(0, epoch_start)
    try:
        for e in range(epoch_offset, args.nepochs):

            # Run
            model.train()
            loss, losses_train, msg = batch_loop(model, optim, args, e, False,
                                                 dataset_train, loader_train,
                                                 logger, lr,
                                                 loss_best, losses_track)
            losses_track['train'].append(losses_train)
            with torch.no_grad():
                model.eval()
                loss, losses_valid, _ = batch_loop(model, optim, args, e, True,
                                                   dataset_valid, loader_valid,
                                                   logger, lr,
                                                   loss_best, losses_track,
                                                   msg_pre=msg)
                losses_track['valid'].append(losses_valid)

            # Control stall
            if np.isnan(loss) or loss > 1000:
                patience = 0
                loss = np.inf
                model = load_best_model()

            # Best model?
            if loss < loss_best * (1 + args.lr_thres):
                print('*', end='')
                loss_best = loss
                patience = args.lr_patience
                to_save = vars(args)
                try:
                    model_state_dict = model.module.state_dict()
                except:
                    model_state_dict = model.state_dict()
                to_save.update({
                    'epoch': e + 1,
                    'iter': len(loader_train) * (e + 1),
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optim.state_dict(),
                    'lr': lr,  # shadowing the original lr
                    'loss_best': loss_best,
                    'losses': losses_track,
                })
                torch.save(to_save, args.base_fn_out + '.pt')
                torch.save(model.module, args.base_fn_out + '.model.pt')
                torch.save(optim, args.base_fn_out + '.optim.pt')
            else:
                # Learning rate annealing or exit
                patience -= 1
                if patience <= 0:
                    restarts -= 1
                    if restarts < 0:
                        print('End')
                        break
                    if lr < 1e-7:
                        print('lr={:.1e}'.format(lr), end='')
                        continue
                    lr *= args.lr_factor
                    print('lr={:.1e}'.format(lr), end='')
                    if args.clearmomentum:
                        optim = get_optimizer(model, lr)
                    else:
                        for pg in optim.param_groups:
                            pg['lr'] = lr
                    patience = args.lr_patience

            print()
    except KeyboardInterrupt:
        print()
    print('-' * 100)


if __name__ == '__main__':
    args = parse_args()
    set_seed(args)

    args.pc_dim = 82

    embeddings = torch.load('/rscratch/xuezhenruo/blow_vctk/blow_200331_test/embeddings.pt')

    ######### Get basis vectors
    embs = embeddings.cpu().numpy()
    embs_center = embs - np.mean(embs, axis=0)
    u, s, vt = np.linalg.svd(embs_center, full_matrices=False)

    # explained variance analysis
    variance_explained = [s[i]**2 / sum(s**2) for i in range(len(s))]
    print(variance_explained)
    variance_explained = np.array(variance_explained)
    print("explained vairance > 1%",
          len(variance_explained[variance_explained > 0.01]))
    print("explained vairance > 2%",
          len(variance_explained[variance_explained > 0.02]))
    print("explained vairance > 3%",
          len(variance_explained[variance_explained > 0.03]))

    # sklearn pca
    X_train = embs
    # for ncomp in range(20, 100, 5):
    #     pca = PCA(n_components=ncomp)
    #     pca.fit(X_train)
    #     X_train_pca = pca.transform(X_train)
    #     X_projected = pca.inverse_transform(X_train_pca)
    #     loss = ((X_train - X_projected) ** 2).mean()
    #     print("{} PC, recon loss is {}".format(ncomp, loss))
    pca = PCA(n_components=args.pc_dim)
    pca.fit(X_train)
    base_embs = pca.components_
    base_embs = torch.FloatTensor(base_embs).cuda()

    dataset_train, dataset_valid, dataset_test, loader_train, loader_valid, loader_test = load_data(args)
    args.ntargets = dataset_test.maxspeakers

    print('Init')
    tstart = time.time()

    # Init tensorboard
    from tensorboardX import SummaryWriter
    os.makedirs(args.base_fn_out, exist_ok=True)
    logger = SummaryWriter(os.path.join(args.base_fn_out, 'logs'))

    # # Loading trained model
    # model_fpath = args.load_existing
    # sw_path = args.sw_path
    # model = load_experiment(args)

    model, optim, epoch_start, losses_track, loss_best = init_experiment()

    # Multigpu
    if args.multigpu:
        print('[Using {:d} GPUs]'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    #
    # ######### Synthesize for unseen speakers
    # speakers = deepcopy(dataset_test.speakers)
    # male_spk = 'p232'
    # female_spk = 'p234'
    # male_ispk = speakers[male_spk]
    # female_ispk = speakers[female_spk]
    #
    # for alpha in np.arange(0, 1.2, 0.2):
    #     new_emb = alpha * embeddings[male_ispk] + (1 - alpha) * embeddings[female_ispk]
    #     new_emb = new_emb.unsqueeze(0)
    #     make_audio_evals_unseen(loader_test, model_fpath, sw_path,
    #                      new_emb, alpha)



    filenames, target_speakers, itrafos = get_transformations()
    train(model, optim, epoch_start, losses_track, loss_best)

    print("Done")
