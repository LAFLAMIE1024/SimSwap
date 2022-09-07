import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from models.projected_model import fsModel
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.reverse2original import reverse2wholeimage
# from util.swap_new_model import swap_result_new_model
import os
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet
from tqdm.notebook import tqdm as tqdm

def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.
    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.
    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)

imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

def swap_result_new_model(face_align_crop, model, latent_id):
    img_align_crop = Image.fromarray(cv2.cvtColor(face_align_crop, cv2.COLOR_BGR2RGB))

    img_tensor = transforms.ToTensor()(img_align_crop)
    img_tensor = img_tensor.view(-1, 3, img_align_crop.size[0], img_align_crop.size[1])

    img_tensor = img_tensor.cuda(non_blocking=True)
    img_tensor = img_tensor.sub_(mean).div_(std)

    swap_res = model.netG(img_tensor, latent_id).cpu()
    swap_res = (swap_res * imagenet_std + imagenet_mean).numpy()
    swap_res = swap_res.squeeze(0).transpose((1, 2, 0))

    swap_result = np.clip(255*swap_res, 0, 255)
    swap_result = img2tensor(swap_result / 255., bgr2rgb=False, float32=True)

    return swap_result


if __name__ == '__main__':
    opt = TestOptions().parse()

    crop_size = opt.crop_size
    torch.nn.Module.dump_patches = True

    if crop_size == 512:
      if opt.name == str(512):
        opt.which_epoch = 550000
      else:
        opt.Gdeep = True
        opt.new_model = True

      mode = 'ffhq'
    else:
      mode = 'None'

    logoclass = watermark_image('./simswaplogo/simswaplogo.png')

    if opt.new_model == True:
        model = fsModel()
        model.initialize(opt)
        model.netG.eval()
    else:            
        model = create_model(opt)
        model.eval()
       
    spNorm = SpecificNorm()
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640), mode=mode)

    dataset = opt.eval_dataset_path

    with torch.no_grad():

      # This list is in the form of pair 'tar src' at each line
      with open('/content/drive/MyDrive/swap_list.txt','r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
          pair = line.split()

          res =  pair[0] + '_' + pair[1]
          if not os.path.exists(os.path.join(opt.output_path, res)): 
            os.mkdir(os.path.join(opt.output_path, res))

          for i in range(10):
                        
            res_path = pair[0] + '_' + pair[1] + '_' + str(i) +  '.jpg'
            res_path = os.path.join(opt.output_path, res, res_path)            
            
            if os.path.exists(res_path): continue

            tar = pair[0] + '_' + str(i)
            src = pair[1] + '_' + str(i)

            # pic_b_path = os.path.join(dataset, pair[0], tar + '.png')
            # pic_a_path = os.path.join(dataset, pair[1], src + '.png')
                
            pic_b_path = os.path.join(dataset, tar + '.png')
            pic_a_path = os.path.join(dataset, src + '.png')
 
            pic_a = pic_a_path
            img_a_whole = cv2.imread(pic_a)
                
            try:
                img_a_align_crop, _ = app.get(img_a_whole,crop_size)
            except:
                print("picture not found : " + pic_a)
                
            img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB))
            
            img_a = transformer_Arcface(img_a_align_crop_pil)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

            # convert numpy to tensor
            img_id = img_id.cuda()

            #create latent id
            img_id_downsample = F.interpolate(img_id, size=(112,112))
            latent_id = model.netArc(img_id_downsample)
            latent_id = F.normalize(latent_id, p=2, dim=1)

            ############## Forward Pass ######################

            pic_b = pic_b_path
            img_b_whole = cv2.imread(pic_b)
        
            try:
                img_b_align_crop_list, b_mat_list = app.get(img_b_whole, crop_size)
            except:
                print("picture not found : " + pic_b)

            swap_result_list = []
            b_align_crop_tenor_list = []

            for b_align_crop in img_b_align_crop_list:
                        
                if opt.new_model == True:
                      b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop[0], cv2.COLOR_BGR2RGB))[None,...].cuda()
                      swap_result = swap_result_new_model(b_align_crop, model, latent_id)
                else:
                      b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop, cv2.COLOR_BGR2RGB))[None,...].cuda()
                      swap_result = model(None, b_align_crop_tenor, latent_id, None, True)[0]

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
                res_path, opt.no_simswaplogo, pasring_model=net, use_mask=opt.use_mask, norm=spNorm)

            print(res_path + ' saved!')

        print(' ')
        print('************ Done ! ************')
