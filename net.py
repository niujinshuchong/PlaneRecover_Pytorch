import torch
import torch.nn as nn

class ConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out, kernel_size, stride, padding):
        super(ConvTranspose2d, self).__init__()
        self.upsample = nn.Upsample(scale_factor=stride)
        self.conv = nn.Conv2d(in_channel, out, kernel_size, stride=1, padding=padding)
    
    def forward(self, x):
        return self.conv(self.upsample(x))

class PlanePredNet(nn.Module):
    def __init__(self, num_planes):
        super(PlanePredNet, self).__init__()
        self.num_planes = num_planes

        self.relu = nn.ReLU(inplace=True)

        self.cnv1 = nn.Conv2d(3, 32, (7, 7), stride=2, padding=3)
        self.cnv1b = nn.Conv2d(32, 32, (7, 7), stride=1, padding=3)

        self.cnv2 = nn.Conv2d(32, 64, (5, 5), stride=2, padding=2)
        self.cnv2b = nn.Conv2d(64, 64, (5, 5), stride=1, padding=2)

        self.cnv3 = nn.Conv2d(64, 128, (3, 3), stride=2, padding=1)
        self.cnv3b = nn.Conv2d(128, 128, (3, 3), stride=1, padding=1)

        self.cnv4 = nn.Conv2d(128, 256, (3, 3), stride=2, padding=1)
        self.cnv4b = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)

        self.cnv5 = nn.Conv2d(256, 512, (3, 3), stride=2, padding=1) 
        self.cnv5b = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)

        self.cnv6 = nn.Conv2d(512, 512, (3, 3), stride=2, padding=1)
        self.cnv7 = nn.Conv2d(512, 512, (3, 3), stride=2, padding=1)
        self.cnv_param_pred = nn.Conv2d(512, 3*self.num_planes, (3, 3), stride=2, padding=1)

        self.upcnv5 = ConvTranspose2d(512, 256, (3, 3), stride=2, padding=1)
        self.upcnv5b = nn.Conv2d(512, 256, (3, 3), stride=1, padding=1)
        
        self.upcnv4 = ConvTranspose2d(256, 128, (3, 3), stride=2, padding=1)
        self.upcnv4b = nn.Conv2d(256, 128, (3, 3), stride=1, padding=1)

        self.upcnv3 = ConvTranspose2d(128, 64, (3, 3), stride=2, padding=1)
        self.upcnv3b = nn.Conv2d(128+self.num_planes, 64, (3, 3), stride=1, padding=1)

        self.upcnv2 = ConvTranspose2d(64, 32, (3, 3), stride=2, padding=1)
        self.upcnv2b = nn.Conv2d(64+self.num_planes, 32, (3, 3), stride=1, padding=1)

        self.upcnv1 = ConvTranspose2d(32, 16, (3, 3), stride=2, padding=1)
        self.upcnv1b = nn.Conv2d(16+self.num_planes, 16, (3, 3), stride=1, padding=1)

        self.conv_segm4 = nn.Conv2d(128, self.num_planes, (3, 3), stride=1, padding=1)
        self.conv_segm3 = nn.Conv2d(64, self.num_planes, (3, 3), stride=1, padding=1)
        self.conv_segm2 = nn.Conv2d(32, self.num_planes, (3, 3), stride=1, padding=1)
        self.conv_segm1 = nn.Conv2d(16, self.num_planes, (3, 3), stride=1, padding=1)
        
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

    def forward(self, image):
        cnv1 = self.relu(self.cnv1(image))
        cnv1b = self.relu(self.cnv1b(cnv1))

        cnv2 = self.relu(self.cnv2(cnv1b))
        cnv2b = self.relu(self.cnv2b(cnv2))        

        cnv3 = self.relu(self.cnv3(cnv2b))
        cnv3b = self.relu(self.cnv3b(cnv3))

        cnv4 = self.relu(self.cnv4(cnv3b))      
        cnv4b = self.relu(self.cnv4b(cnv4))

        cnv5 = self.relu(self.cnv5(cnv4b))
        cnv5b = self.relu(self.cnv5b(cnv5))

        # plane parameters
        cnv6_plane = self.relu(self.cnv6(cnv5b))
        cnv7_plane = self.relu(self.cnv7(cnv6_plane))
        param_pred = self.cnv_param_pred(cnv7_plane)

        param_avg = torch.mean(torch.mean(param_pred, dim=3), dim=2)     # (b, 3, h, w)
        param_final = 0.01 * param_avg.view(-1, self.num_planes, 3)

        # deconv
        upcnv5 = self.relu(self.upcnv5(cnv5b))
        upcnv5 = torch.cat((upcnv5, cnv4b), dim=1)
        upcnv5b = self.relu(self.upcnv5b(upcnv5))   # (b, 256, 12, 20)

        upcnv4 = self.relu(self.upcnv4(upcnv5b))
        upcnv4 = torch.cat((upcnv4, cnv3b), dim=1)
        upcnv4b = self.relu(self.upcnv4b(upcnv4))   # (b, 128, 24, 40)
        segm4 = self.conv_segm4(upcnv4b)            
        segm4_up = self.up_sample(segm4)

        upcnv3 = self.relu(self.upcnv3(upcnv4b))
        upcnv3 = torch.cat((upcnv3, cnv2b, segm4_up), dim=1)
        upcnv3b = self.relu(self.upcnv3b(upcnv3))
        segm3 = self.conv_segm3(upcnv3b)            
        segm3_up = self.up_sample(segm3)
        
        upcnv2 = self.relu(self.upcnv2(upcnv3b))
        upcnv2 = torch.cat((upcnv2, cnv1b, segm3_up), dim=1)
        upcnv2b = self.relu(self.upcnv2b(upcnv2))
        segm2 = self.conv_segm2(upcnv2b)
        segm2_up = self.up_sample(segm2)

        upcnv1 = self.relu(self.upcnv1(upcnv2b)) 
        upcnv1 = torch.cat((upcnv1, segm2_up), dim=1)
        upcnv1b = self.relu(self.upcnv1b(upcnv1))        
        segm1 = self.conv_segm1(upcnv1b)

        return param_final, [segm1, segm2, segm3, segm4]



