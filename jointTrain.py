# code to train world coord regression from RGB Image
# models are saved in checkpoints-wc/

import os
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import matplotlib

matplotlib.use('Agg')

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from models import get_model
from loaders import get_loader
from utils import show_wc_tnsboard, get_lr, show_unwarp_tnsboard
import grad_loss
import recon_lossc


def write_log_file(log_file_name, losses, epoch, lrate, phase, state):
    with open(log_file_name, 'a') as f:
        if state == 'wc':
            f.write("\n{} LRate: {} Epoch: {} State: {} Loss: {} MSE: {} GradLoss: {}".format(phase, lrate, epoch,
                                                                            state,losses[0], losses[1], losses[2]))
        elif state == 'bm':
            f.write(
                "\n{} LRate: {} Epoch: {} State: {} Loss: {} MSE: {} UnwarpL2: {} UnwarpSSIMloss: {}".format(phase,
                                                    lrate, epoch, state, losses[0], losses[1], losses[2],losses[3]))
        elif state == 'total':
            f.write(
                "\n{} LRate: {} Epoch: {} State: {} Loss: {} MSE: {}".format(phase,
                                                    lrate, epoch, state, losses[0], losses[1]))



def train(args):
    # Setup Dataloader
    wc_data_loader = get_loader('doc3dwc')
    data_path = args.data_path
    wc_t_loader = wc_data_loader(data_path, is_transform=True, img_size=(args.wc_img_rows, args.wc_img_cols),
                                 augmentations=args.augmentation)
    wc_v_loader = wc_data_loader(data_path, is_transform=True, split='val',
                                 img_size=(args.wc_img_rows, args.wc_img_cols))

    wc_n_classes = wc_t_loader.n_classes
    wc_trainloader = data.DataLoader(wc_t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    wc_valloader = data.DataLoader(wc_v_loader, batch_size=args.batch_size, num_workers=8)

    # Setup Model
    model_wc = get_model('unetnc', wc_n_classes, in_channels=3)
    model_wc = torch.nn.DataParallel(model_wc, device_ids=range(torch.cuda.device_count()))
    model_wc.cuda()

    # Setup Dataloader
    bm_data_loader = get_loader('doc3dbmnic')
    bm_t_loader = bm_data_loader(data_path, is_transform=True, img_size=(args.bm_img_rows, args.bm_img_cols))
    bm_v_loader = bm_data_loader(data_path, is_transform=True, split='val',
                                 img_size=(args.bm_img_rows, args.bm_img_cols))

    bm_n_classes = bm_t_loader.n_classes
    bm_trainloader = data.DataLoader(bm_t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    bm_valloader = data.DataLoader(bm_v_loader, batch_size=args.batch_size, num_workers=8)

    # Setup Model
    model_bm = get_model('dnetccnl', bm_n_classes, in_channels=3)
    model_bm = torch.nn.DataParallel(model_bm, device_ids=range(torch.cuda.device_count()))
    model_bm.cuda()

    if os.path.isfile(args.shape_net_loc):
        print("Loading model_wc from checkpoint '{}'".format(args.shape_net_loc))
        checkpoint = torch.load(args.shape_net_loc)
        model_wc.load_state_dict(checkpoint['model_state'])
        print("Loaded checkpoint '{}' (epoch {})".format(args.shape_net_loc, checkpoint['epoch']))
    else:
        print("No model_wc checkpoint found at '{}'".format(args.shape_net_loc))
        exit(1)
    if os.path.isfile(args.texture_mapping_net_loc):
        print("Loading model_bm from checkpoint '{}'".format(args.texture_mapping_net_loc))
        checkpoint = torch.load(args.texture_mapping_net_loc)
        model_bm.load_state_dict(checkpoint['model_state'])
        print("Loaded checkpoint '{}' (epoch {})".format(args.texture_mapping_net_loc, checkpoint['epoch']))
    else:
        print("No model_bm checkpoint found at '{}'".format(args.texture_mapping_net_loc))
        exit(1)

    # Activation
    htan = nn.Hardtanh(0, 1.0)

    # Optimizer
    optimizer = torch.optim.Adam(list(model_wc.parameters()) + list(model_bm.parameters()), lr=args.l_rate,
                                 weight_decay=5e-4, amsgrad=True)

    # LR Scheduler 
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Losses
    MSE = nn.MSELoss()
    loss_fn = nn.L1Loss()
    gloss = grad_loss.Gradloss(window_size=5, padding=2)
    reconst_loss = recon_lossc.Unwarploss()

    epoch_start = 0

    # Log file:
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    experiment_name = 'joint train'
    log_file_name = os.path.join(args.logdir, experiment_name + '.txt')
    if os.path.isfile(log_file_name):
        log_file = open(log_file_name, 'a')
    else:
        log_file = open(log_file_name, 'w+')

    log_file.write('\n---------------  ' + experiment_name + '  ---------------\n')
    log_file.close()

    # Setup tensorboard for visualization
    if args.tboard:
        # save logs in runs/<experiment_name> 
        writer = SummaryWriter(comment=experiment_name)

    best_val_mse = 99999.0
    global_step = 0
    LClambda = 0.2
    bm_img_size = (128, 128)

    alpha = 0.5
    beta = 0.5
    for epoch in range(epoch_start, args.n_epoch):
        avg_loss = 0.0

        wc_avg_l1loss = 0.0
        wc_avg_gloss = 0.0
        wc_train_mse = 0.0

        bm_avgl1loss = 0.0
        bm_avgrloss = 0.0
        bm_avgssimloss = 0.0
        bm_train_mse = 0.0

        model_wc.train()
        model_bm.train()
        if epoch == 50 and LClambda < 1.0:
            LClambda += 0.2
        for (i, (wc_images, wc_labels)), (i, (bm_images, bm_labels)) in zip(enumerate(wc_trainloader),
                                                                            enumerate(bm_trainloader)):
            wc_images = Variable(wc_images.cuda())
            wc_labels = Variable(wc_labels.cuda())

            optimizer.zero_grad()
            wc_outputs = model_wc(wc_images)
            pred_wc = htan(wc_outputs)
            g_loss = gloss(pred_wc, wc_labels)
            wc_l1loss = loss_fn(pred_wc, wc_labels)
            loss = alpha * (wc_l1loss + LClambda * g_loss)

            bm_images = Variable(bm_images.cuda())
            bm_labels = Variable(bm_labels.cuda())
            bm_input = F.interpolate(pred_wc, bm_img_size)

            target = model_bm(bm_input)
            target_nhwc = target.transpose(1, 2).transpose(2, 3)
            bm_val_l1loss = loss_fn(target_nhwc, bm_labels)
            rloss, ssim, uworg, uwpred = reconst_loss(bm_images[:, :-1, :, :], target_nhwc, bm_labels)
            loss += beta * ((10.0 * bm_val_l1loss) + (0.5 * rloss))

            avg_loss += float(loss)

            wc_avg_l1loss += float(wc_l1loss)
            wc_avg_gloss += float(g_loss)
            wc_train_mse += float(MSE(pred_wc, wc_labels).item())

            bm_avgl1loss += float(bm_val_l1loss)
            bm_avgrloss += float(rloss)
            bm_avgssimloss += float(ssim)

            bm_train_mse += float(MSE(target_nhwc, bm_labels).item())

            loss.backward()
            optimizer.step()
            global_step += 1

            if (i + 1) % 50 == 0:
                print("Epoch[%d/%d] Batch [%d/%d] Loss: %.4f" % (
                    epoch + 1, args.n_epoch, i + 1, len(wc_trainloader), avg_loss / 50.0))
                avg_loss = 0.0

            if args.tboard and (i + 1) % 20 == 0:
                show_wc_tnsboard(global_step, writer, wc_images, wc_labels, pred_wc, 8, 'Train Inputs', 'Train WCs',
                                 'Train pred_wc. WCs')
                writer.add_scalar('WC: L1 Loss/train', wc_avg_l1loss / (i + 1), global_step)
                writer.add_scalar('WC: Grad Loss/train', wc_avg_gloss / (i + 1), global_step)
                show_unwarp_tnsboard(bm_images, global_step, writer, uwpred, uworg, 8, 'Train GT unwarp',
                                     'Train Pred Unwarp')
                writer.add_scalar('BM: L1 Loss/train', bm_avgl1loss / (i + 1), global_step)
                writer.add_scalar('CB: Recon Loss/train', bm_avgrloss / (i + 1), global_step)
                writer.add_scalar('CB: SSIM Loss/train', bm_avgssimloss / (i + 1), global_step)

        wc_train_mse = wc_train_mse / len(wc_trainloader)
        wc_avg_l1loss = wc_avg_l1loss / len(wc_trainloader)
        wc_avg_gloss = wc_avg_gloss / len(wc_trainloader)
        print("wc Training L1:%4f" % (wc_avg_l1loss))
        print("wc Training MSE:'{}'".format(wc_train_mse))
        wc_train_losses = [wc_avg_l1loss, wc_train_mse, wc_avg_gloss]

        lrate = get_lr(optimizer)

        write_log_file(log_file_name, wc_train_losses, epoch + 1, lrate, 'Train', 'wc')

        bm_avgssimloss = bm_avgssimloss / len(bm_trainloader)
        bm_avgrloss = bm_avgrloss / len(bm_trainloader)
        bm_avgl1loss = bm_avgl1loss / len(bm_trainloader)
        bm_train_mse = bm_train_mse / len(bm_trainloader)
        print("bm Training L1:%4f" % (bm_avgl1loss))
        print("bm Training MSE:'{}'".format(bm_train_mse))
        bm_train_losses = [bm_avgl1loss, bm_train_mse, bm_avgrloss, bm_avgssimloss]

        write_log_file(log_file_name, bm_train_losses, epoch + 1, lrate, 'Train', 'bm')

        model_wc.eval()
        model_bm.eval()

        val_mse = 0.0
        val_loss = 0.0

        wc_val_loss = 0.0
        wc_val_gloss = 0.0
        wc_val_mse = 0.0

        bm_val_l1loss = 0.0
        val_rloss = 0.0
        val_ssimloss = 0.0
        bm_val_mse = 0.0

        for (i_val, (wc_images_val, wc_labels_val)), (i_val, (bm_images_val, bm_labels_val)) in tqdm(
                zip(enumerate(wc_valloader), enumerate(bm_valloader))):
            with torch.no_grad():
                wc_images_val = Variable(wc_images_val.cuda())
                wc_labels_val = Variable(wc_labels_val.cuda())

                wc_outputs = model_wc(wc_images_val)
                pred_val = htan(wc_outputs)
                wc_g_loss = gloss(pred_val, wc_labels_val).cpu()
                pred_val = pred_val.cpu()
                wc_labels_val = wc_labels_val.cpu()
                wc_val_loss += loss_fn(pred_val, wc_labels_val)
                wc_val_mse += float(MSE(pred_val, wc_labels_val))
                wc_val_gloss += float(wc_g_loss)

                bm_images_val = Variable(bm_images_val.cuda())
                bm_labels_val = Variable(bm_labels_val.cuda())
                bm_input = F.interpolate(pred_val, bm_img_size)
                target = model_bm(bm_input)
                target_nhwc = target.transpose(1, 2).transpose(2, 3)
                pred = target_nhwc.data.cpu()
                gt = bm_labels_val.cpu()
                bm_val_l1loss += loss_fn(target_nhwc, bm_labels_val)
                rloss, ssim, uworg, uwpred = reconst_loss(bm_images_val[:, :-1, :, :], target_nhwc, bm_labels_val)
                val_rloss += float(rloss.cpu())
                val_ssimloss += float(ssim.cpu())
                bm_val_mse += float(MSE(pred, gt))
                val_loss += (alpha * wc_val_loss + beta * bm_val_l1loss)
                val_mse += (wc_val_mse + bm_val_mse)
            if args.tboard:
                show_unwarp_tnsboard(bm_images_val, epoch + 1, writer, uwpred, uworg, 8, 'Val GT unwarp',
                                     'Val Pred Unwarp')

        if args.tboard:
            show_wc_tnsboard(epoch + 1, writer, wc_images_val, wc_labels_val, pred_val, 8, 'Val Inputs', 'Val WCs',
                             'Val Pred. WCs')
            writer.add_scalar('WC: L1 Loss/val', wc_val_loss, epoch + 1)
            writer.add_scalar('WC: Grad Loss/val', wc_val_gloss, epoch + 1)

            writer.add_scalar('BM: L1 Loss/val', bm_val_l1loss, epoch + 1)
            writer.add_scalar('CB: Recon Loss/val', val_rloss, epoch + 1)
            writer.add_scalar('CB: SSIM Loss/val', val_ssimloss, epoch + 1)
            writer.add_scalar('total val loss', val_loss, epoch + 1)

        wc_val_loss = wc_val_loss / len(wc_valloader)
        wc_val_mse = wc_val_mse / len(wc_valloader)
        wc_val_gloss = wc_val_gloss / len(wc_valloader)
        print("wc val loss at epoch {}:: {}".format(epoch + 1, wc_val_loss))
        print("wc val MSE: {}".format(wc_val_mse))

        bm_val_l1loss = bm_val_l1loss / len(bm_valloader)
        bm_val_mse = bm_val_mse / len(bm_valloader)
        val_ssimloss = val_ssimloss / len(bm_valloader)
        val_rloss = val_rloss / len(bm_valloader)
        print("bm val loss at epoch {}:: {}".format(epoch + 1, bm_val_l1loss))
        print("bm val mse: {}".format(bm_val_mse))

        val_loss /= len(wc_valloader)
        val_mse /= len(wc_valloader)
        print("val loss at epoch {}:: {}".format(epoch + 1, val_loss))
        print("val mse: {}".format(val_mse))

        bm_val_losses = [bm_val_l1loss, bm_val_mse, val_rloss, val_ssimloss]
        wc_val_losses = [wc_val_loss, wc_val_mse, wc_val_gloss]
        total_val_losses = [val_loss, val_mse]
        write_log_file(log_file_name, wc_val_losses, epoch + 1, lrate, 'Val', 'wc')
        write_log_file(log_file_name, bm_val_losses, epoch + 1, lrate, 'Val', 'bm')
        write_log_file(log_file_name, total_val_losses, epoch + 1, lrate, 'Val', 'total')

        # reduce learning rate
        sched.step(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            state_wc = {'epoch': epoch + 1,
                     'model_state': model_wc.state_dict(),
                     'optimizer_state': optimizer.state_dict(), }
            torch.save(state_wc,
                       args.logdir + "{}_{}_{}_{}_{}_best_wc_model.pkl".format('unetnc', epoch + 1, wc_val_mse, wc_train_mse,
                                                                            experiment_name))
            state_bm = {'epoch': epoch + 1,
                     'model_state': model_bm.state_dict(),
                     'optimizer_state': optimizer.state_dict(), }
            torch.save(state_bm,
                       args.logdir + "{}_{}_{}_{}_{}_best_bm_model.pkl".format('dnetccnl', epoch + 1, bm_val_mse, bm_train_mse,
                                                                            experiment_name))

        if (epoch + 1) % 10 == 0 and epoch > 70:
            state_wc = {'epoch': epoch + 1,
                     'model_state': model_wc.state_dict(),
                     'optimizer_state': optimizer.state_dict(), }
            torch.save(state_wc, args.logdir + "{}_{}_{}_{}_{}_wc_model.pkl".format('unetnc', epoch + 1, wc_val_mse, wc_train_mse,
                                                                              experiment_name))
            state_bm = {'epoch': epoch + 1,
                     'model_state': model_bm.state_dict(),
                     'optimizer_state': optimizer.state_dict(), }
            torch.save(state_bm, args.logdir + "{}_{}_{}_{}_{}_bm_model.pkl".format('dnetccnl', epoch + 1, bm_val_mse, bm_train_mse,
                                                                              experiment_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--data_path', nargs='?', type=str, default='',
                        help='Data path to load data')
    parser.add_argument('--wc_img_rows', nargs='?', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--wc_img_cols', nargs='?', type=int, default=256,
                        help='Width of the input image')
    parser.add_argument('--bm_img_rows', nargs='?', type=int, default=128,
                        help='Height of the input image')
    parser.add_argument('--bm_img_cols', nargs='?', type=int, default=128,
                        help='Width of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5,
                        help='Learning Rate')
    parser.add_argument('--shape_net_loc', nargs='?', type=str, default=None,
                        help='Path to previous saved shape network model to restart from')
    parser.add_argument('--texture_mapping_net_loc', nargs='?', type=str, default=None,
                        help='Path to previous saved texture mapping network model to restart from')
    parser.add_argument('--logdir', nargs='?', type=str, default='./logdir',
                        help='Path to store the loss logs')
    parser.add_argument('--tboard', dest='tboard', action='store_true',
                        help='Enable visualization(s) on tensorboard | False by default')
    parser.add_argument('--augmentation', nargs='?', type=bool, default=False,
                        help='whether to augment training data')
    parser.set_defaults(tboard=False)

    args = parser.parse_args()
    train(args)

# CUDA_VISIBLE_DEVICES=1 python jointTrain.py --data_path ./data/DewarpNet/doc3d/ --batch_size 40 --tboard --shape_net_loc ./eval/models/unetnc_doc3d.pkl --texture_mapping_net_loc dnetccnl_doc3d.pkl
