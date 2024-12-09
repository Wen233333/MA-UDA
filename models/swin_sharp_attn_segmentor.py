import torch
import torch.nn as nn
from models.swin_sharp_attn_transformer import SwinTransformer
from models.uper_head import UPerHead
from models.fcn_head import FCNHead


class Swin_Segmentor(nn.Module):
    def __init__(self,
                 pretrain_img_size,
                 in_channel,
                 num_classes,
                 pretrained=None):
        super(Swin_Segmentor, self).__init__()
        self.backbone = SwinTransformer(pretrain_img_size=pretrain_img_size, in_chans=in_channel)
        self.decode_head = UPerHead(num_classes=num_classes)
        self.with_decode_head = True
        self.auxiliary_head = FCNHead(num_classes=num_classes)
        self.with_auxiliary_head = True
        self.init_weights(pretrained=pretrained)

        assert self.with_decode_head

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def extract_feat(self, img):
        """Extract features from images."""
        x, all_attns, all_shifted_attns = self.backbone(img)
        return x, all_attns, all_shifted_attns

    def _decode_head_forward_train(self, x):
        decode_head_output = self.decode_head.forward(x)
        return decode_head_output

    def _auxiliary_head_forward_train(self, x):
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                aux_head_output = aux_head.forward(x)
        else:
            aux_head_output = self.auxiliary_head.forward(x)
        return aux_head_output

    def forward(self, img):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # x: tuple(length:4)
        attn_list = []
        feat, all_attns, all_shifted_attns = self.extract_feat(img)
        attn_list.append(all_attns)
        attn_list.append(all_shifted_attns)
        decode_head_out = self._decode_head_forward_train(feat)
        if self.with_auxiliary_head:
            auxiliary_head_out = self._auxiliary_head_forward_train(feat)
        return decode_head_out, auxiliary_head_out, attn_list, list(feat)


if __name__ == '__main__':
    img = torch.Tensor(69, 1, 192, 192)
    model = Swin_Segmentor(pretrain_img_size=224, in_channel=1, num_classes=4)
    print(model)
    decode_head_out, auxiliary_head_out, attn_list, feat_list = model(img)
    print(len(attn_list))
    print(attn_list[0].shape)
    print(attn_list[1].shape)
    print(len(feat_list))
    print(decode_head_out.shape)
    print(auxiliary_head_out.shape)