'''
Author: Naiyuan liu & Marco Cheung
Github: https://github.com/NNNNAI
Date: 2022-8-2 11:58:34
LastEditors: Marco Cheung
LastEditTime: 2022-8-2 11:58:34
Description: 
'''

import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.reverse2original import reverse2wholeimage
import os
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet

def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

if __name__ == '__main__':
    opt = TestOptions().parse()

    crop_size = opt.crop_size
    torch.nn.Module.dump_patches = True
    imagenet_std    = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)
    imagenet_mean   = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)
        
    if crop_size == 512:
        # opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'
        
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    model = create_model(opt)
    model.eval()

    spNorm =SpecificNorm()

    from torchvision import transforms as T
    c_transforms = []
    c_transforms.append(T.ToTensor())
    c_transforms = T.Compose(c_transforms)

    mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1,3,1,1)
    stream = torch.cuda.Stream()
        
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode=mode)

    with torch.no_grad():
        pic_a = opt.pic_a_path

        # img_a_whole = cv2.imread(pic_a)
        # img_a_align_crop, _ = app.get(img_a_whole,crop_size)
        # img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
        # img_a = transformer_Arcface(img_a_align_crop_pil)
        # img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # convert numpy to tensor
        # img_id = img_id.cuda()
        image1 = c_transforms(Image.open(pic_a))
        
        ############## Forward Pass ######################

        pic_b = opt.pic_b_path
        
        img_b_whole = cv2.imread(pic_b)
        img_b_align_crop_list, b_mat_list = app.get(img_b_whole,crop_size)
        
        swap_result_list = []
        b_align_crop_tenor_list = []

        with torch.cuda.stream(stream):

            src_image1  = image1.cuda(non_blocking=True)
            src_image1  = src_image1.unsqueeze(0).sub_(mean).div_(std)

            src_image2  = image2.cuda(non_blocking=True)
            src_image2  = src_image2.unsqueeze(0).sub_(mean).div_(std)
        
        # create latent id
        img_id_downsample = F.interpolate(src_image1, size=(112,112))
        latent_id = model.netArc(img_id_downsample)
        latent_id = F.normalize(latent_id, p=2, dim=1)
        
        for b_align_crop in img_b_align_crop_list:

            b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

            # swap_result = model(None, b_align_crop_tenor, latent_id, None, True)[0]
            # swap_result = model.netG(b_align_crop_tenor, latent_id).cpu()

            swap_result = model.netG(src_image2, latent_id).cpu()
            swap_result = swap_result * imagenet_std
            swap_result = swap_result + imagenet_mean
        
            swap_result_list.append(swap_result)
            b_align_crop_tenor_list.append(b_align_crop_tenor)

        if opt.use_mask:
            n_classes = 19
            net = BiSeNet(n_classes=n_classes)
            net.cuda()
            save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
            net.load_state_dict(torch.load(save_pth))
            net.eval()
        else:
            net = None

        reverse2wholeimage(b_align_crop_tenor_list, swap_result_list, b_mat_list, crop_size, img_b_whole, logoclass, \
            os.path.join(opt.output_path, 'result_whole_swapsingle.jpg'), opt.no_simswaplogo,pasring_model =net,use_mask=opt.use_mask, norm = spNorm)

        print(' ')

        print('************ Done ! ************')
