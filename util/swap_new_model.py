import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

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


def swap_result_new_model(face_align_crop, model, latend_id):
    img_align_crop = Image.fromarray(cv2.cvtColor(face_align_crop, cv2.COLOR_BGR2RGB))

    img_tensor = transforms.ToTensor()(img_align_crop)
    img_tensor = img_tensor.view(-1, 3, img_align_crop.size[0], img_align_crop.size[1])

    mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)

    img_tensor = img_tensor.cuda(non_blocking=True)
    img_tensor = img_tensor.sub_(mean).div_(std)

    imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

    swap_res = model.netG(img_tensor, latend_id).cpu()
    swap_res = (swap_res * imagenet_std + imagenet_mean).numpy()
    swap_res = swap_res.squeeze(0).transpose((1, 2, 0))

    swap_result = np.clip(255*swap_res, 0, 255)
    swap_result = img2tensor(swap_result / 255., bgr2rgb=False, float32=True)
    return swap_result