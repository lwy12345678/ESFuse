from PIL import Image
import PIL
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import cv2
import time
import imageio
import kornia.utils
import torchvision.transforms as transforms
from tqdm import tqdm
# from model_SEA_1 import lwy_Fusionnet
import torch.nn as nn
import sys
sys.path.append('/public/home/w__y/code/fir2')
from model import Main_Interpreter, Feature_Recon, edge_Interpreter, DIM
import glob
# from model_2 import FusionNet
import torchvision.utils as vutils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = lwy_Fusionnet(nfeats=64).to(device)
main_model = nn.DataParallel(Main_Interpreter(nfeats=64).to(device))
edge_model = nn.DataParallel(edge_Interpreter().to(device))
decoder = nn.DataParallel(Feature_Recon().to(device))
dim = nn.DataParallel(DIM().to(device))


model_path = "fus_0024.pth"
main_model.load_state_dict(torch.load(model_path)['main_model'])
edge_model.load_state_dict(torch.load(model_path)['edge_model'])
decoder.load_state_dict(torch.load(model_path)['decoder'])
dim.load_state_dict(torch.load(model_path)['dim'])

def fusion():
    # ---------------------------------TNO------------------------------------
    # ir_path = '/home/w_y/datasets/IR_VI/TNO_/ir'   # ir路径
    # vi_path = '/home/w_y/datasets/IR_VI/TNO_/vi'   # vi路径
    # model_type_as_color = False
    # vi_is_y = True

    # --------------------------------Road-------------------------------------
    ir_path = '/public/home/w__y/datasets/IR_VI/RoadScene/test/ir'   # ir路径
    vi_path = '/public/home/w__y/datasets/IR_VI/RoadScene/test/vi'   # vi路径
    model_type_as_color = False
    vi_is_y = False

    save = '/public/home/w__y/code/fir2/test/result/RoadScene/'
    os.makedirs(os.path.dirname(save), exist_ok=True)
    data = GetDataset_type2('val', ir_path=ir_path, vi_path=vi_path)
    test_data_loader = torch.utils.data.DataLoader(dataset=data,
                                                   batch_size=1,
                                                   shuffle=False
                                                   )
    i = 0
    time_list = []
    with torch.no_grad():
        for ir, vi, name in test_data_loader:
            main_model.eval()
            edge_model.eval()
            decoder.eval()
            dim.eval()

            i += 1


            img1_org = ir.cuda()
            img2_org = vi.cuda()
            img2_org_ycrcb = img2_org
            start = time.time()
            ir_feat, vi_feat = main_model(img1_org, img2_org_ycrcb)
            R, edge_out = edge_model(ir_feat)
            # vutils.save_image(edge_out[0], f'./000/edge/edge.png')
            f, f_edge_out = dim(R, edge_out)
            # for i in range(f_edge_out.size(1)):
            #     vutils.save_image(f_edge_out[0, i], f'./000/result/channel_{i}.png')
            fusion_image = decoder(ir_feat, vi_feat, f, f_edge_out)
            end = time.time()
            time_list.append(end - start)
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
                # print('1---------------------------')
            fused_image = fusion_image.cpu().numpy()
                # print('2---------------------------')
            fused_image = (fused_image - np.min(fused_image)) / (
                        np.max(fused_image) - np.min(fused_image)
                )
            fused_image = np.uint8(255.0 * fused_image)

                # print(fused_image.shape)
            image = fused_image[0, 0, :, :]
                # print(image.shape)
            image = Image.fromarray(image)
            # print(name)
            image.save(save + name[0])

            print('finish:', name[0])

        print('平均耗时：', np.mean(time_list[1:]), 's')
        print('std：', np.std(time_list[1:]))

class GetDataset(torch.utils.data.Dataset):

    def __init__(self, ir_path, vi_path, len):
        super(GetDataset, self).__init__()
        self.ir = ir_path
        self.vi = vi_path
        self.len = len


    def __getitem__(self, index):
        ir_path = self.ir + str(index + 1) + '.jpg'
        vi_path = self.vi + str(index + 1) + '.jpg'

        ir = _imread(ir_path)
        vi = _imread(vi_path)


        return (ir, vi)

    def __len__(self):
        return self.len


# def _imread(path):
#     im_cv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     assert im_cv is not None, f"Image {str(path)} is invalid."
#     im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
#     return im_ts


def _imread(path):
    im_cv = Image.open(path).convert('L')
    # im_cv = cv2.imread(str(path), flags)
    im_cv = im_cv.resize((600,400), Image.ANTIALIAS)
    assert im_cv is not None, f"Image {str(path)} is invalid."
    # im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
    tran = transforms.ToTensor()
    im_ts = tran(im_cv) / 255.
    return im_ts

def imsave(im_s, dst, im_name: str = ''):
    """
    save images to path
    :param im_s: image(s)
    :param dst: if one image: path; if multiple images: folder path
    :param im_name: name of image
    """

    im_s = im_s if type(im_s) == list else [im_s]
    dst = [dst / str(i + 1).zfill(3) / im_name for i in range(len(im_s))] if len(im_s) != 1 else [dst / im_name]
    for im_ts, p in zip(im_s, dst):
        im_ts = im_ts.squeeze().cpu()
        p.parent.mkdir(parents=True, exist_ok=True)
        im_cv = kornia.utils.tensor_to_image(im_ts) * 255.
        cv2.imwrite(str(p), im_cv)

class GetDataset_type2(torch.utils.data.Dataset):
    def __init__(self, split, ir_path=None, vi_path=None):
        super(GetDataset_type2, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        if split == 'val':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split=='val':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            image_vis = cv2.imread(vis_path, 0)

            image_inf = cv2.imread(ir_path, 0)
            # print(image_vis.shape)
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_vis = np.asarray(Image.fromarray(image_vis), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            image_vis = np.expand_dims(image_vis, axis=0)
            name = self.filenames_vis[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                name,
            )

    def __len__(self):
        # print(self.length)
        return self.length


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames

class Fusion_dataset(torch.utils.data.Dataset):
    def __init__(self, split, vi_is_y, ir_path=None, vi_path=None):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        self.vi_is_y = vi_is_y

        if split == 'val':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split=='val':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
                # image_vis = np.array(Image.open(vis_path))
            image_vis = cv2.imread(vis_path, 0)

            image_inf = cv2.imread(ir_path, 0)
                # print(image_vis.shape)
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_vis = np.asarray(Image.fromarray(image_vis), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            image_vis = np.expand_dims(image_vis, axis=0)

            return (
                torch.tensor(image_ir),
                torch.tensor(image_vis)
            )

    def __len__(self):
        return self.length

if __name__ == '__main__':
    fusion()