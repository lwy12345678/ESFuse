import sys

sys.path.append("..")
import time
import visdom
import pathlib
import warnings
import logging.config
import argparse, os
import torch.backends.cudnn
import torch.utils.data
import torch.nn.functional
import datetime
from tqdm import tqdm
from dataloader.fuse_data_vsm import GetDataset_type2
# from model.model import FusionNet
from model.model_SEA_1 import Main_Interpreter, Feature_ReconB, edge_Interpreter
from model.model_SEA_1 import Feature_Recon, DIM
from loss.loss import *
from loss.loss_vif import fusion_loss_vif


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def hyper_args():
    """
    get hyper parameters from args
    """
    parser = argparse.ArgumentParser(description='RobF Net train process')

    # dataset
    parser.add_argument('--ir_reg', default='/public/home/wz/lwy_code/ESFuse/ir', type=str)
    parser.add_argument('--vi', default='/public/home/wz/lwy_code/ESFuse/vi', type=str)
    parser.add_argument('--edge_path', default='/public/home/wz/lwy_code/ESFuse/edge', type=str)

    # parser.add_argument('--data_len', default='221', type=int)
    # train loss weights
    parser.add_argument('--theta', default=5, type=float, help='edge')
    # implement details
    # parser.add_argument('--dim', default=128, type=int, help='AFuse feather dim')
    parser.add_argument('--batchsize', default=30, type=int, help='mini-batch size')  # 32
    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
    parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument('--nEpochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")
    parser.add_argument("--step", type=int, default=100, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument('--resume', default='', help='resume checkpoint')
    parser.add_argument('--interval', default=20, help='record interval')
    # checkpoint
    parser.add_argument("--load_model_fuse", default=None, help="path to pretrained model (default: none)")
    # parser.add_argument("--load_model_fuse", default='./cache/9.2/fus_0250.pth', help="path to pretrained model (default: none)")
    parser.add_argument('--ckpt', default='./cache/', help='checkpoint cache folder')

    args = parser.parse_args()
    return args

def main(args, visdom):

    cuda = args.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    log = logging.getLogger()

    epoch = args.nEpochs
    interval = args.interval

    print("===> Creating Save Path of Checkpoints")
    cache = pathlib.Path(args.ckpt)

    print("===> Loading datasets")
    # crop = torchvision.transforms.RandomResizedCrop(256)
    # folder_dataset_train_ir = glob.glob(args.ir_reg)
    # folder_dataset_train_vi = glob.glob(args.vi)
    # folder_dataset_train_ir_map = glob.glob(args.ir_map)
    # folder_dataset_train_vi_map = glob.glob(args.vi_map)
    data = GetDataset_type2('train', ir_path=args.ir_reg, vi_path=args.vi, edge_path=args.edge_path)
    training_data_loader = torch.utils.data.DataLoader(data, args.batchsize, True, pin_memory=True)

    print("===> Building models")
    main_model = nn.DataParallel(Main_Interpreter(nfeats=64).to(device))
    edge_model = nn.DataParallel(edge_Interpreter().to(device))
    decoder = nn.DataParallel(Feature_Recon().to(device))
    Dim = nn.DataParallel(DIM().to(device))

    print("===> Setting Optimizers")
    optimizer_1 = torch.optim.Adam(params=main_model.parameters(), lr=args.lr)
    optimizer_2 = torch.optim.Adam(params=edge_model.parameters(), lr=args.lr)
    optimizer_3 = torch.optim.Adam(params=decoder.parameters(), lr=args.lr)
    optimizer_4 = torch.optim.Adam(params=Dim.parameters(), lr=args.lr)


    # print("===> Building deformation")
    # affine  = AffineTransform(translate=0.01)
    # elastic = ElasticTransform(kernel_size=101, sigma=16)

    # TODO: optionally copy weights from a checkpoint
    if args.load_model_fuse is not None:
        print('Loading pre-trained FuseNet checkpoint %s' % args.load_model_fuse)
        log.info(f'Loading pre-trained checkpoint {str(args.load_model_fuse)}')
        state = torch.load(str(args.load_model_fuse))
        main_model.load_state_dict(state['main_model'])
        edge_model.load_state_dict(state['edge_model'])
        decoder.load_state_dict(state['decoder'])
        Dim.load_state_dict(state['DIM'])
    else:
        print("=> no model found at '{}'".format(args.load_model_fuse))

    print("===> Starting Training")
    for epoch in range(args.start_epoch, args.nEpochs + 1):
        tqdm_loader = training_data_loader
        train(args, tqdm_loader, optimizer_1, optimizer_2, optimizer_3, optimizer_4,
              main_model, edge_model, decoder, Dim, epoch, device)


def train(args, tqdm_loader, optimizer_1, optimizer_2, optimizer_3, optimizer_4,
           main_model, edge_model, decoder,  Dim, epoch, device):

    main_model.train()
    edge_model.train()
    decoder.train()
    Dim.train()
    # TODO: update learning rate of the optimizer
    lr_F = adjust_learning_rate(args, optimizer_1, epoch - 1)
    print("Epoch={}, lr_F={} ".format(epoch, lr_F))

    loss_total = []
    # prev_time = time.time()
    for i, (ir, vi, edge) in enumerate(tqdm_loader):
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        optimizer_3.zero_grad()
        optimizer_4.zero_grad()

        main_model.zero_grad()
        edge_model.zero_grad()
        decoder.zero_grad()
        Dim.zero_grad()
        # --------------------------------------------------------
        # device = torch.device('cuda:0')
        ir_reg, vi, edge = ir, vi, edge
        vi_ycbcr = RGB2YCrCb(vi)
        vi_y = vi_ycbcr[:, 0:1, :, :]
        ir_feat, vi_feat = main_model(ir_reg, vi_y)
        edge_out, R = edge_model(ir_feat)
        f, F_edge1 = Dim(R, edge_out)
        fuse_out = decoder(ir_feat, vi_feat, edge = F_edge1)
            # print('----------------------------------------')
            # ---------------------------------------------------------
        fusion_loss_ = fusion_loss_vif()
        fusion_loss = fusion_loss_(vi_y, ir_reg, fuse_out, device)
        edge_loss = torch.nn.functional.l1_loss(edge_out, edge.cuda()) * args.theta

            # --------------------------------------------------
        loss = fusion_loss + edge_loss
            # loss = fusion_loss


        loss.backward()
        nn.utils.clip_grad_norm_(
                main_model.parameters(), max_norm=0.01, norm_type=2)
        nn.utils.clip_grad_norm_(
                edge_model.parameters(), max_norm=0.01, norm_type=2)
        nn.utils.clip_grad_norm_(
                decoder.parameters(), max_norm=0.01, norm_type=2)
        optimizer_1.step()
        optimizer_2.step()
        optimizer_3.step()
        optimizer_4.step()

        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [fusion_loss: %f]"
            % (
                epoch,
                args.nEpochs,
                i,
                len(tqdm_loader),
                fusion_loss.item(),
                # edge_loss.item()
            )
        )

    checkpoint = {
        'main_model': main_model.state_dict(),
        'edge_model': edge_model.state_dict(),
        'decoder': decoder.state_dict(),
        'DIM': Dim.state_dict(),
    }

    # TODO: save checkpoint
    save_checkpoint(checkpoint, epoch, args.ckpt) if epoch % args.interval == 0 else None


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.step))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def save_checkpoint(net, epoch, cache):
    model_folder = cache
    model_out_path = model_folder + f'fus_{epoch:04d}.pth'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(net, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = hyper_args()
    visdom = visdom.Visdom(port=8097, env='Fusion')

    main(args, visdom)


