import nibabel as nib
import numpy as np
import torch
from torch.nn.functional import pad

def load_image(img_path,isLabel=False):
    out_img = nib.load(img_path).get_data()
    # if len(out_img.shape) == 2:
    #     out_img = np.transpose(out_img, [1, 0])
    # elif len(out_img.shape) == 3:
    #     out_img = np.transpose(out_img, [2, 1, 0])
    # else:
    #     print('Image Size {} not supported!'.format(len(out_img.shape)))
    #     quit()
    if isLabel:
        out_img[out_img == 3] = 0
        out_img[out_img == 4] = 3
    return out_img

def pad_image(img):
    leng = img.size(0)
    outimg = torch.zeros(leng, 240, 240)
    for i in range(leng):
        outimg[i,] = pad((img[i,]).squeeze(), (40, 40, 40, 20))
    outimg = outimg.unsqueeze(1)
    return outimg

def stack_image(img, label=None, pad_img=False):
    print(img.size())
    if pad_img:
        print('Padding Image to 155x240x240')
        img = pad_image(img)
    outimg = torch.cat((img[:-2, ...], img[1:-1, ...], img[2:, ...]), dim=1)
    if label is not None:
        label = label[1:-1, ]
        return outimg, label
    else:
        return outimg