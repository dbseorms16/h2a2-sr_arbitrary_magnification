
from model import common, dct
import torch
import torch.nn as nn
import math

def make_model(args, parent=False):
    return H2A2RDN(args)

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class H2A2SR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(H2A2SR, self).__init__()
        self.scale = args.scale[0]
        self.int_scale = math.floor(self.scale)
        self.float_scale = args.float_scale
        self.total_scale = args.total_scale
        self.res_scale = self.total_scale / self.int_scale
        kernel_size = 3
        act = nn.ReLU(True)

        self.dct = dct.DCT_2D()
        self.idct = dct.IDCT_2D()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size, padding=1)
        self.R1 = common.RCAB(conv, 64, kernel_size, act=act)
        self.R2 = common.RCAB(conv, 64, kernel_size, act=act)
        self.R3 = common.RCAB(conv, 64, kernel_size, act=act)
        self.R4 = common.RCAB(conv, 64, kernel_size, act=act)
        self.R5 = common.RCAB(conv, 64, kernel_size, act=act)
        self.t = nn.Conv2d(64, 3, kernel_size, padding=1)

    def forward(self, x):
        b, c, H, W = x.size()
        outH, outW = int(H * self.res_scale), int(W * self.res_scale)
        x = self.dct(x)
        zeroPad2d = nn.ZeroPad2d((0,int(outW - W), 0, int(outH - H))).to('cuda:0')
        x = zeroPad2d(x)

        mask = torch.ones((outH, outW), dtype=torch.int64, device = torch.device('cuda:0'))
        #diagonal 이 양수일 경우 대각선 라인이 왼쪽 상단으로 향함
        diagonal = 200

        ## lf, hf 나누기
        lf_mask = torch.fliplr(torch.triu(mask, diagonal)) == 1
        lf_mask = lf_mask.unsqueeze(0).expand(x.size())
        lf = x * lf_mask

        hf_mask = torch.fliplr(torch.triu(mask, diagonal)) != 1
        hf_mask = hf_mask.unsqueeze(0).expand(x.size())
        hf = x * hf_mask
        ## 모델 시작
        ## 고주파 집중네트워크
        hf = self.idct(hf)
        hf = self.conv1(hf)
        hf = self.R1(hf)
        hf = self.R2(hf)
        hf = self.R3(hf)
        hf = self.R4(hf)
        hf = self.R5(hf)
        hf = self.t(hf)
        hf = self.dct(hf)

        ## 계수 합치기
        result = lf + hf
        result = self.idct(result)
        return x

class H2A2RDN(nn.Module):
    def __init__(self, args):
        super(H2A2RDN, self).__init__()

        self.h2a2sr = H2A2SR(args)
        self.scale = args.scale[0]
        self.int_scale = math.floor(self.scale)
        
        r = self.int_scale
        G0 = 64
        kSize = 3

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = 16, 8, 64

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        # Up-sampling net
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(r),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        elif r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        x = self.sub_mean(x)
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1

        x = self.UPNet(x)
        x = self.add_mean(x)
        x = self.h2a2sr(x)
        return x
