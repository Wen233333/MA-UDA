import argparse
import torch
import os
parser = argparse.ArgumentParser(description='cross domain segmentation Training')

parser.add_argument('--source-dataset',
                    help='which dataset to use, mr or ct', default='ct', type=str)
parser.add_argument('--target-dataset',
                    help='which dataset to use, mr or ct', default='mr', type=str)
parser.add_argument('--data-path',
                    help='the path of the dataset', default='/data/jw/data/udaSeg/new_data/', type=str)
parser.add_argument('--image-path',
                    help='the path of the images', default='/data/jw/data/udaSeg/new_data/', type=str)
parser.add_argument('--pretrain-path',
                    help='the path of the images',
                    default='/data/jw/results/Trans_UDA_Media/pretrained_model/swin_tiny_patch4_window7_224.pth', type=str)

parser.add_argument('-k', default=1, type=int, metavar='N',
                    help='k folds cross validation')
parser.add_argument('-b', '--batch-size', default=30, type=int, metavar='N',
                    help='mini-batch size (default: 16)')
parser.add_argument('-vb', '--val-batch-size', default=120, type=int, metavar='N',
                    help='mini-batch size (default: 16)')
parser.add_argument('--test-batch-size', default=30, type=int, metavar='N',
                    help='mini-batch size for test (default: 50)')
parser.add_argument('-j', '--workers', default=5, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('--lr', '--learning-rate', default=[1e-2], type=list, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--lr_D', '--learning-rate-D', default=5e-5, type=float,
                    help='initial learning rate of discriminator')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='Weight decay (default: 5e-4)')

parser.add_argument('--dec-weight', default=1, type=float, metavar='M',
                    help='decode head loss weight')
parser.add_argument('--aux-weight', default=0.4, type=float, metavar='M',
                    help='decode head loss weight')
parser.add_argument('--output_consis_weight', default=1e-3, type=float, metavar='M',
                    help='output_consis_weight')

parser.add_argument('--ori-weight', default=1, type=float, metavar='M',
                    help='output_consis_weight')
parser.add_argument('--gen-weight', default=0.5, type=float, metavar='M',
                    help='output_consis_weight')

parser.add_argument('--attention_weight_list', default=[0.1, 0.1], type=list, metavar='M',
                    help='list of weight for each layer, len = 0 or 2 (Index 0 for Attention; 1 for Shifted-Attention)')
parser.add_argument('--feature_weight_list', default=[0.1], type=list, metavar='M',
                    help='list of weight for each layer')

parser.add_argument('--a_adapt_dims', default=6, type=int, metavar='N',
                    help='a_adapt_dims')

parser.add_argument('--device', default='1, 2, 3, 4', type=str, metavar='G',
                    help='gpu device')
parser.add_argument('-hm', '--hmloss', type=str, default='Vanilla',
                    help="choose the GAN objective(Vanilla, LS).")
parser.add_argument('--stack-image', action='store_false', default=False,
                    help='Use stacked images')
parser.add_argument('--class_balance_weight', default=[1, 1, 1, 1, 1], type=list, metavar='M',
                    help='list of weight of each class for CrossEntropy Loss')

args = parser.parse_args()
args.testname = '{}'.format(os.times())
result_dir = '/data/jw/results/test_output_path2/'\
    .format(args.source_dataset, args.target_dataset, args.lr, args.lr_D, args.output_consis_weight,
            str(args.attention_weight_list), str(args.feature_weight_list), args.aux_weight, args.gen_weight)


def prepare_device():
    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    list_ids = list(range(n_gpu))
    return device, list_ids

