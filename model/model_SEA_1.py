import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChannelAttention(nn.Module):
    '''
    '''
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    '''
    '''
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MSFI(nn.Module):
    '''
    Convolutional Block Attention Module. CA+SA
    '''
    def __init__(self, nfeats):
        super(MSFI, self).__init__()
        self.CA = ChannelAttention(num_channels=64)
        self.SA = SpatialAttention()

        self.conv1 = nn.Sequential(
            nn.Conv2d(nfeats, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x1):
        x = torch.cat((x, x1), dim=1)
        x = self.conv1(x)
        CA = self.CA(x)
        atten = x * CA
        SA = self.SA(atten)
        atten = atten * SA

        return atten


class RCB(nn.Module):
    '''
    '''
    def __init__(self, in_dim, out_dim):
        super(RCB, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, (in_dim + out_dim) // 2, 3, 1, 1, bias=True),
            # nn.BatchNorm2d((in_dim + out_dim) // 2),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d((in_dim + out_dim) // 2, out_dim, 5, stride=1, padding=2, bias=True),
            # nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x1 = self.conv3(self.conv2(x))
        if self.in_dim == self.out_dim:
            out = x1 + res
        else:
            out = x1 + self.conv4(res)
        return out


class Multiscale_Reconstruct(nn.Module):
    def __init__(self, in_dim):
        super(Multiscale_Reconstruct, self).__init__()

        self.rcb1_1 = RCB(in_dim=in_dim, out_dim=32)
        self.rcb1_2 = RCB(in_dim=32, out_dim=16)
        self.rcb1_3 = RCB(in_dim=16, out_dim=1)


        self.rcb2_1 = RCB(in_dim=in_dim, out_dim=32)
        self.rcb2_2 = RCB(in_dim=32, out_dim=32)
        self.rcb2_3 = RCB(in_dim=32, out_dim=16)
        self.rcb2_4 = RCB(in_dim=16, out_dim=1)


        self.rcb3_1 = RCB(in_dim=in_dim, out_dim=32)
        self.rcb3_2 = RCB(in_dim=32, out_dim=32)
        self.rcb3_3 = RCB(in_dim=32, out_dim=16)
        self.rcb3_4 = RCB(in_dim=16, out_dim=16)
        self.rcb3_5 = RCB(in_dim=16, out_dim=1)



    def forward(self, x, scale):

        if scale == 1:
            x = self.rcb1_1(x)
            x = self.rcb1_2(x)
            out = self.rcb1_3(x)

        elif scale == 2:
            x = self.rcb2_1(x)
            x = self.rcb2_2(x)
            x = self.rcb2_3(x)
            out = self.rcb2_4(x)


        elif scale == 3:
            x = self.rcb3_1(x)
            x = self.rcb3_2(x)
            x = self.rcb3_3(x)
            x = self.rcb3_4(x)
            out = self.rcb3_5(x)

        elif scale == 'all':
            x1 = self.rcb1_1(x)
            x1 = self.rcb1_2(x1)
            x1 = self.rcb1_3(x1)

            x2 = self.rcb2_1(x)
            x2 = self.rcb2_2(x2)
            x2 = self.rcb2_3(x2)
            x2 = self.rcb2_4(x2)

            x3 = self.rcb3_1(x)
            x3 = self.rcb3_2(x3)
            x3 = self.rcb3_3(x3)
            x3 = self.rcb3_4(x3)
            x3 = self.rcb3_5(x3)

            out = x1 + x2 + x3

        return out

class DIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.con1 = nn.Conv2d(64, 64, padding=1, stride=1, kernel_size=3)
        self.con2 = nn.Conv2d(64, 64, padding=1, stride=1, kernel_size=3)
        self.con3 = nn.Conv2d(64, 64, padding=1, stride=1, kernel_size=3)
        self.con4 = nn.Conv2d(64, 64, padding=1, stride=1, kernel_size=3)
        self.sig = nn.Sigmoid()

    def forward(self, R, F_edge):
        F_edge = F_edge + R
        F_edge1 = self.con1(F_edge)
        att = self.sig(self.con2(F_edge1))
        R = self.con3(R)
        R1 = self.con4(R)
        R1 = R1 * att
        f = R + R1

        return f, F_edge1

class FuseModule(nn.Module):
  """
  """
  def __init__(self, ir_in_dim, vi_in_dim, in_dim = 64):
    super(FuseModule, self).__init__()

    self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
    self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

    self.conv_ir = nn.Sequential(
        nn.Conv2d(ir_in_dim, in_dim, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.2)
    )
    self.conv_vi = nn.Sequential(
        nn.Conv2d(vi_in_dim, in_dim, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.2)
    )

    self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
    self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
    self.sig = nn.Sigmoid()


  def forward(self, ir, vi, edge= None):
    x = self.conv_ir(ir)
    prior = self.conv_vi(vi)
    if edge is not None:
        x = x + edge
        prior = prior + edge
    x_q = self.query_conv(x)
    prior_k = self.key_conv(prior)
    energy = x_q * prior_k
    attention = self.sig(energy)
    attention_x = x * attention
    attention_p = prior * attention
    x_gamma = self.gamma1(torch.cat((x, attention_x), dim=1))
    p_gamma = self.gamma2(torch.cat((prior, attention_p), dim=1))
    x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]
    prior_out = prior * p_gamma[:, [0], :, :] + attention_p * p_gamma[:, [1], :, :]

    return x_out, prior_out


class edge_Interpreter(nn.Module):
    def __init__(self):
        super(edge_Interpreter, self).__init__()

        self.edge_conv1 = nn.Conv2d(128, 64, 3, 1, 1)
        self.RCB1 = RCB(64, 64)
        self.RCB2 = RCB(64, 32)
        self.RCB3 = RCB(32, 16)
        self.edge_conv2 = nn.Conv2d(16, 1, 3, 1, 1)
        self.act = nn.Tanh()

    def forward(self, ir_feat):
        R = self.edge_conv1(ir_feat)
        edge = self.RCB3(self.RCB2(self.RCB1(R)))
        edge = self.edge_conv2(edge)
        edge = self.act(edge)

        return edge, R

class Feature_Recon(nn.Module):
    def __init__(self):
        super(Feature_Recon, self).__init__()
        self.fuse = FuseModule(ir_in_dim=128, vi_in_dim=64)
        self.Multiscale_Reconstruct = Multiscale_Reconstruct(in_dim=192)
        self.out_conv = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.act = nn.Tanh()


    def forward(self, ir_feat, vi_feat, f, edge=None):
        if edge is not None:
            fuse_feat_ir, fuse_feat_vi = self.fuse(ir_feat, vi_feat, edge)
            x = torch.cat((fuse_feat_ir, fuse_feat_vi), dim=1)
            # x = x + edge
            x = torch.cat((x, f), dim=1)
            x = self.Multiscale_Reconstruct(x, scale='all')
            x = self.act(x)
            # x = x + edge
        else:
            fuse_feat_ir, fuse_feat_vi = self.fuse(ir_feat, vi_feat)
            x = torch.cat((fuse_feat_ir, fuse_feat_vi), dim=1)
            x = self.Multiscale_Reconstruct(x, scale='all')
            x = self.act(x)
        return x

class Feature_ReconB(nn.Module):
    def __init__(self):
        super(Feature_ReconB, self).__init__()
        self.fuse = FuseModule(ir_in_dim=128, vi_in_dim=64)
        self.Multiscale_Reconstruct = Multiscale_Reconstruct(in_dim=128)
        self.out_conv = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.act = nn.Tanh()


    def forward(self, ir_feat, vi_feat, edge=None):
        if edge is not None:
            fuse_feat_ir, fuse_feat_vi = self.fuse(ir_feat, vi_feat, edge)
            x = torch.cat((fuse_feat_ir, fuse_feat_vi), dim=1)
            x = x + edge
            x = self.Multiscale_Reconstruct(x, scale = 2)
            x = self.act(x)
            x = x + edge
        else:
            fuse_feat_ir, fuse_feat_vi = self.fuse(ir_feat, vi_feat)
            x = torch.cat((fuse_feat_ir, fuse_feat_vi), dim=1)
            x = self.Multiscale_Reconstruct(x, scale = 2)
            x = self.act(x)
        return x

class Main_Interpreter(nn.Module):
    def __init__(self, nfeats=64):
        super(Main_Interpreter, self).__init__()
        # head
        self.head_conv = nn.Conv2d(1, nfeats, kernel_size=3, stride=1, padding=1)

        # body
        self.RCB1 = RCB(nfeats, nfeats)
        self.RCB2 = RCB(2 * nfeats, nfeats)
        self.RCB3 = RCB(3 * nfeats, nfeats)
        self.RCB4 = RCB(4 * nfeats, nfeats)

        # attention block
        self.MSFI1 = MSFI(2 * nfeats)

        # edge
        self.edge_conv1 = nn.Conv2d(nfeats * 2, nfeats, 3, 1, 1)
        self.edge_conv2 = nn.Conv2d(nfeats, 1, 3, 1, 1)

        # fuse block
        self.fuse = FuseModule(ir_in_dim=2 * nfeats, vi_in_dim=nfeats)
        self.fuse_res = nn.Conv2d(nfeats * 2, nfeats, kernel_size=3, stride=1, padding=1)
        self.Multiscale_Reconstruct = Multiscale_Reconstruct(in_dim=128)

        # tail
        self.out_conv = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(negative_slope=0.2),
            # nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(negative_slope=0.2),
        )
        self.act = nn.Tanh()
        self.norm = nn.BatchNorm2d(1)


    def forward(self, ir, vi):
        # ------------------------------------------------------------------
        ir_feat_0 = self.head_conv(ir)
        vi_feat = self.head_conv(vi)

        # ------------------------------------------------------------------
        ir_feat_1 = self.RCB1(ir_feat_0)
        vi_feat_1 = self.RCB1(vi_feat)

        ir_feat_2 = self.MSFI1(ir_feat_1, vi_feat_1)
        vi_feat = torch.cat((vi_feat, vi_feat_1), dim=1)
        ir_feat_2 = torch.cat((ir_feat_2, ir_feat_1), dim=1)


        # ------------------------------------------------------------------
        ir_feat_2 = self.RCB2(ir_feat_2)
        vi_feat_2 = self.RCB2(vi_feat)
        ir_feat_3 = self.MSFI1(ir_feat_2, vi_feat_2)
        ir_feat_3 = torch.cat((ir_feat_3, ir_feat_2), dim=1)
        vi_feat = torch.cat((vi_feat, vi_feat_2), dim=1)


        # ------------------------------------------------------------------
        ir_feat_3 = self.RCB2(ir_feat_3)
        vi_feat_3 = self.RCB3(vi_feat)
        ir_feat_4 = self.MSFI1(ir_feat_3, vi_feat_3)
        ir_feat_4 = torch.cat((ir_feat_4, ir_feat_3), dim=1)
        vi_feat = torch.cat((vi_feat, vi_feat_3), dim=1)

        # ------------------------------------------------------------------
        ir_feat_4 = self.RCB2(ir_feat_4)
        vi_feat_4 = self.RCB4(vi_feat)
        ir_feat_5 = self.MSFI1(ir_feat_4, vi_feat_4)
        ir_feat_5 = torch.cat((ir_feat_5, ir_feat_4), dim=1)

        return ir_feat_5, vi_feat_4




def unit_test():
    import numpy as np
    x1 = torch.tensor(np.random.rand(2,1,480,640).astype(np.float32))
    x2 = torch.tensor(np.random.rand(2,64,480,640).astype(np.float32))
    # x1, x2 = x1.cuda(), x2.cuda()
    main_model = Main_Interpreter(nfeats=64)
    edge_model = edge_Interpreter()
    decoder = Feature_Recon()
    Dim = DIM()

    ir_feat, vi_feat = main_model(x1, x1)
    # edge_out, R = edge_model(ir_feat)
    print(ir_feat.shape)
    print(vi_feat.shape)
    f, F_edge1 = Dim(x2, x1)
    result = decoder(ir_feat, vi_feat, f, edge = F_edge1)

    # print('output shape1:', edge.shape)
    print('output shape2:', result.shape)
    # assert y.shape == (2,1,480,640), 'output shape (2,1,480,640) is expected!'
    print('test ok!')

if __name__ == '__main__':
    unit_test()