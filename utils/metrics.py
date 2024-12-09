import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


def sensi(vol, gt, tid=1):
    vol = vol == tid  # make it boolean
    gt = gt == tid  # make it boolean
    vol = np.asarray(vol).astype(np.bool)
    gt = np.asarray(gt).astype(np.bool)

    if vol.shape != gt.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    if gt.sum() == 0:
        return -1
    intersection = np.logical_and(vol, gt)
    sen = 1.0 * intersection.sum() / gt.sum()
    return sen


def pospv(vol, gt, tid=1):
    vol = vol == tid  # make it boolean
    gt = gt == tid  # make it boolean
    vol = np.asarray(vol).astype(np.bool)
    gt = np.asarray(gt).astype(np.bool)

    if vol.shape != gt.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    if vol.sum() == 0:
        return -1
    intersection = np.logical_and(vol, gt)
    ppv = 1.0 * intersection.sum() / vol.sum()
    return ppv


def dice(mask_pred, mask_gt):
    """Compute soerensen-dice coefficient.
    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`.
    Args:
      mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
      mask_pred: 3-dim Numpy array of type bool. The predicted mask.
    Returns:
      the dice coeffcient as float. If both masks are empty, the result is NaN.
    """

    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return -1
    volume_intersect = (mask_gt * mask_pred).sum()
    return 2 * volume_intersect / volume_sum


def calMetrics(allres, this_iter, this_class):
    this_res = allres[:this_iter + 1, this_class]
    if this_res[this_res >= 0].size == 0:
        return 0, 0, 0
    else:
        return np.mean(this_res[this_res >= 0]), np.median(this_res[this_res >= 0]), np.std(this_res[this_res >= 0])


def calHD95Metrics(allres, this_iter, this_class):
    this_res = allres[:this_iter + 1, this_class]
    if this_res[this_res >= 0].size == 0:
        return 0, 0, 0
    else:
        return np.mean(this_res[(this_res >= 0) * (this_res != np.Inf)]), np.median(this_res[(this_res >= 0) * (this_res != np.Inf)]), np.std(this_res[(this_res >= 0) * (this_res != np.Inf)])

# def calMetrics(allres, validnum, this_iter, this_class):
#     validnum = validnum.astype(np.bool)
#     this_res = allres[:this_iter + 1, this_class]
#     this_valid = validnum[:this_iter + 1, this_class]
#     if this_res[this_valid].size == 0:
#         return 0, 0, 0
#     else:
#         # return np.mean(this_res[this_res >= 0 and this_res != np.Inf]), np.median(this_res[this_res >= 0 and this_res != np.Inf]), np.std(this_res[this_res >= 0 and this_res != np.Inf])
#         return np.mean(this_res[this_valid]), np.median(this_res[this_valid]), np.std(this_res[this_valid])


class GeneralizedDiceLoss(nn.Module):
    def __init__(self):
        super(GeneralizedDiceLoss, self).__init__()

    def forward(self, logits, target):
        # logits = make_same_size(logits, target)

        size = logits.size()
        N, nclass = size[0], size[1]

        # N x 1 x H x W
        if size[1] > 1:
            pred = F.softmax(logits, dim=1)
        else:
            pred = torch.sigmoid(logits)
            pred = torch.cat([1 - pred, pred], 1)
            nclass = 2

        if len(target.size()) < len(size):
            target = target.unsqueeze(1)
        # N x C x H x W
        pred_one_hot = pred
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot ** 2 + target_one_hot ** 2

        # N x C
        inter = inter.view(N, nclass, -1).sum(2).sum(0)
        union = union.view(N, nclass, -1).sum(2).sum(0)

        # NxC
        dice = 2 * inter / union
        return 1 - dice.mean()


def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert torch.max(tensor).item() < nClasses
    size = list(tensor.size())
    # print(size)
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot


def make_same_size(logits, target):
    assert isinstance(logits, torch.Tensor), "model output {}".format(type(logits))
    size = logits.size()
    # print('Logit size {}, target size {}'.format(size,target.size()))
    if logits.size() != target.size():
        if len(size) == 5:
            logits = F.interpolate(logits, target.size()[2:], align_corners=False, mode='trilinear')
        elif len(size) == 4:
            logits = F.interpolate(logits, target.size()[2:], align_corners=False, mode='bilinear')
        else:
            raise Exception("Invalid size of logits : {}".format(size))
    return logits
