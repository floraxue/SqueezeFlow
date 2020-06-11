import torch
import numpy as np
import argparse
import os

from utils import utils, datain
from utils import audio as audioutils


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(args.seed)


def load_data(args, pars):
    # Data sets and data loaders
    print('Data')
    dataset_train = datain.DataSet(args.path_data, pars.lchunk, pars.stride,
                                   split='train', sampling_rate=pars.sr,
                                   trim=pars.trim,
                                   frame_energy_thres=pars.frame_energy_thres,
                                   temp_jitter=pars.augment > 0,
                                   seed=args.seed)
    dataset_valid = datain.DataSet(args.path_data, pars.lchunk, pars.stride,
                                   split='valid', sampling_rate=pars.sr,
                                   trim=pars.trim,
                                   frame_energy_thres=pars.frame_energy_thres,
                                   temp_jitter=False,
                                   seed=args.seed)
    dataset_test = datain.DataSet(args.path_data, pars.lchunk, pars.stride,
                                  split='test', sampling_rate=pars.sr,
                                  trim=pars.trim,
                                  frame_energy_thres=pars.frame_energy_thres,
                                  temp_jitter=False,
                                  seed=args.seed)
    loader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=args.sbatch,
                                               shuffle=True, drop_last=True,
                                               num_workers=pars.nworkers)
    loader_valid = torch.utils.data.DataLoader(dataset_valid,
                                               batch_size=args.sbatch,
                                               shuffle=False,
                                               num_workers=pars.nworkers)
    loader_test = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=pars.nworkers)

    print('-' * 100)
    return dataset_train, loader_train, dataset_valid, loader_valid, dataset_test, loader_test


batch_data_augmentation = audioutils.DataAugmentation('cpu')


def batch_loop(blow_model, optimizer, loader, args, pars, epoch, logger,
               dataset_test, loader_test,
               is_eval=False, device='cuda'):
    # Prepare
    if is_eval:
        blow_model.eval()
    else:
        blow_model.train()

    running_loss = []
    for k, (audios, info) in enumerate(loader):
        # if is_eval:
        #     import pdb
        #     pdb.set_trace()
        # Prepare data
        if not is_eval and pars.augment > 0:
            if pars.augment > 1:
                audios = batch_data_augmentation.noiseg(audios, 0.001)
            audios = batch_data_augmentation.emphasis(audios, 0.2)
            audios = batch_data_augmentation.magnorm_flip(audios, 1)
            if pars.augment > 1:
                audios = batch_data_augmentation.compress(audios, 0.1)

        # get mel
        x = utils.get_mel(audios)
        isource = info[:, 3]

        # Forward
        x = x.to(device)
        isource = isource.to(device)
        z, log_det = blow_model.forward(x, isource)
        loss, losses = utils.loss_flow_nll(z, log_det)

        if not is_eval:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss.append(loss)

        # print('\r {} iter {} loss {}'.format('eval' if is_eval else 'train', k,
        #                                      loss.item()), end='')
        if k % 10 == 0:
            print('{} iter {} loss {} losses {}'.format(
                'eval' if is_eval else 'train',
                k, loss, losses))
            logger.add_scalar('val_loss' if is_eval else 'train_loss',
                              sum(running_loss)/ (k + 1),
                              k + len(loader) * epoch)
        if (k + len(loader) * epoch) % 1000 == 0 and not is_eval:
            ckpt_path = os.path.join(args.path_out,
                                     'iter_{}.pt'.format(k + len(loader) * epoch))
            blow_ckpt_path = os.path.join(args.path_out,
                                          'iter_{}.model.pt'.format(k + len(loader) * epoch))
            torch.save({'optim_state_dict': optimizer.state_dict(),
                        'iter': k + len(loader) * epoch,
                        'running_loss': sum(running_loss) / (k + 1)},
                       ckpt_path)
            if args.multigpu:
                torch.save(blow_model.module, blow_ckpt_path)
            else:
                torch.save(blow_model, blow_ckpt_path)
            # make_audio_evals(dataset_test, loader_test, blow_ckpt_path,
            #                  logger, epoch, args)

    # print('\r', end='')  # To clear the output

    return running_loss


def make_audio_evals(dataset_test, testloader, blow_ckpt_path,
                     logger, epoch, args, device='cuda'):
    # Hacking to get the top 20 of the transformation list with default seed
    filenames = dataset_test.filenames[:20]

    # Make a blow model for eval
    _, _, blow_model, _ = utils.load_stuff(blow_ckpt_path, device)
    blow_model.use_coeff = False   # Compatibility with embedding branch
    try:
        blow_model.precalc_matrices('on')
    except:
        pass
    blow_model.eval()

    squeezewave = torch.load(args.sw_path)['model']
    squeezewave = squeezewave.remove_weightnorm(squeezewave)
    squeezewave.cuda().eval()

    # Synthesize
    path_out = os.path.join(args.base_fn_out, 'syn_adapt')
    os.makedirs(path_out, exist_ok=True)
    print('Saving {} mels to {}'.format(len(filenames), path_out))
    for k, (audio, info) in enumerate(testloader):
        if k >= len(filenames):
            break
        itarget = torch.LongTensor([0])  # represents unseen speaker, emb will be inserted at speaker id 0
        fn = filenames[k][:-3]
        target_spk = 'unseen_alpha{}'.format(alpha)

        synth_audio = utils.synthesize_one_audio(audio, info, itarget, fn, target_spk,
                                                 blow_model, path_out, squeezewave,
                                                 print_mel=True,
                                                 convert=True,
                                                 target_emb=target_emb)
        # Add audio to TB. audio is a 1D array within [-1, 1]
        log_name = os.path.split(fn)[1] + '_to_' + target_spk
        log_name = 'test_audio_ckpt_{}/{}'.format(epoch + 1, log_name)
        logger.add_audio(log_name, synth_audio, epoch + 1, 22050)



def main():
    parser = argparse.ArgumentParser(description='Audio synthesis script')
    parser.add_argument('--path_data', default='', type=str, required=True,
                        help='(default=%(default)s)')
    parser.add_argument('--base_fn_model', default='', type=str, required=True,
                        help='(default=%(default)s)')
    parser.add_argument('--path_out', default='../res/', type=str, required=True,
                        help='(default=%(default)s)')
    parser.add_argument('--split', default='test', type=str, required=False,
                        help='(default=%(default)s)')
    parser.add_argument('--force_target_speaker', default='', type=str,
                        required=False, help='(default=%(default)s)')
    parser.add_argument('--sw_path', default='/work/x/blow-mel/res/L128_large_pretrain',
                        type=str, required=False, help='(default=%(default)d)')
    parser.add_argument('--sbatch', default=256, type=int, required=False,
                        help='(default=%(default)d)')
    parser.add_argument('--lr', default=1e-4, type=float, required=False,
                        help='(default=%(default)f)')
    parser.add_argument('--multigpu', action='store_true')
    args = parser.parse_args()
    args.seed = 0
    args.device = 'cuda'

    set_seed(args)

    if args.multigpu:
        args.ngpus = torch.cuda.device_count()
        args.sbatch *= args.ngpus
    else:
        args.ngpus = 1

    checkpoint = torch.load(args.base_fn_model + '.pt')
    pars = checkpoint.copy()
    del pars['model_state_dict']
    del pars['optimizer_state_dict']
    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    pars = Namespace(**pars)

    dataset_train, loader_train, dataset_valid, loader_val, \
        dataset_test, loader_test = load_data(args, pars)
    args.ntargets = dataset_train.maxspeakers  # should be 10
    args.emb_dim = 128

    _, _, blow_model, _ = utils.load_stuff(args.base_fn_model)
    blow_model.cuda()
    for p in blow_model.parameters():
        p.requires_grad = False
    new_embs = np.random.randn(args.ntargets, args.emb_dim).astype(np.float32)
    new_embs = torch.from_numpy(new_embs).cuda()
    blow_model.embedding.weight.data = new_embs
    blow_model.embedding.weight.requires_grad = True

    optimizer = torch.optim.Adam(blow_model.parameters(), lr=args.lr)

    # Init tensorboard
    from tensorboardX import SummaryWriter
    os.makedirs(args.path_out, exist_ok=True)
    logger = SummaryWriter(os.path.join(args.path_out, 'logs'))

    if args.multigpu:
        blow_model = torch.nn.DataParallel(blow_model)

    print("Training")
    loss_best = np.inf
    patience = 10
    restarts = 1000
    lr = args.lr
    for epoch in range(100000):

        running_loss = batch_loop(blow_model, optimizer, loader_train, args, pars,
                                  epoch, logger, dataset_test, loader_test)
        running_loss = sum(running_loss) / len(loader_train)

        # Eval
        with torch.no_grad():
            running_loss_eval = batch_loop(blow_model, optimizer, loader_val, args, pars,
                                           epoch, logger, dataset_test, loader_test,
                                           is_eval=True)
        running_loss_eval = sum(running_loss_eval) / len(loader_val)
        print('epoch [%d] train loss: %.3f val loss: %.3f' %
              (epoch + 1,
               running_loss,
               running_loss_eval))

        if running_loss_eval < loss_best * (1 + 1e-4):
            loss_best = running_loss_eval
            patience = 10
        else:
            # Learning rate annealing or exit
            patience -= 1
            if patience <= 0:
                restarts -= 1
                if restarts < 0:
                    print('End')
                    break
                if lr < 1e-6:
                    print('lr={:.1e}'.format(lr))
                    continue
                lr *= 0.5
                print('lr={:.1e}'.format(lr))
                optimizer = torch.optim.Adam(blow_model.parameters(), lr=lr)
                patience = 10




if __name__ == '__main__':
    main()
