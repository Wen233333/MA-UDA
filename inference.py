import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1,0'
import torch.nn as nn
import torch
import torchvision
import numpy as np
from models.swin_sharp_attn_segmentor import Swin_Segmentor
import warnings
from utils.calDist import compute_surface_distances, compute_average_surface_distance, compute_dice_coefficient
from utils.metrics import sensi, pospv, calMetrics
from utils.utils import stack_image
import nibabel as nib
import SimpleITK as sitk
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
warnings.filterwarnings('ignore')


def prepare_device():
    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    list_ids = list(range(n_gpu))
    return device, list_ids


device, device_ids = prepare_device()

val_num_classes = 4
num_classes = 4
badrate = 0.8
# label     1      2      3     4
gtname = ['MYO', 'LAC', 'LVC', 'AA']

test_size = None
data_path = '/datapath'
target_dataset = 'mr'
test_batch_size = 1
stack_image = False

result_path = '/results/'
model_dir = 'model_dir/'
model_name = 'model_name.pth'
result_dir = os.path.join(result_path, model_dir, 'test')
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)


def add(x):
    log_file.write(x + "\n")
    print(x)


def test(model, val_path):
    model.eval()
    all_tests = []

    for image in os.listdir(val_path):
        if image.startswith('image'):
            all_tests.append(image)

    if test_size is not None:
        all_tests = np.random.permutation(all_tests)[:test_size]
    alldsc = np.zeros([len(all_tests), num_classes])
    meandsc = np.zeros([num_classes])
    mediandsc = np.zeros([num_classes])
    stddsc = np.zeros([num_classes])

    allppv = np.zeros([len(all_tests), num_classes])
    meanppv = np.zeros([num_classes])
    medianppv = np.zeros([num_classes])
    stdppv = np.zeros([num_classes])

    allsen = np.zeros([len(all_tests), num_classes])
    meansen = np.zeros([num_classes])
    mediansen = np.zeros([num_classes])
    stdsen = np.zeros([num_classes])

    allasd = np.zeros([len(all_tests), num_classes])
    meanasd = np.zeros([num_classes])
    medianasd = np.zeros([num_classes])
    stdasd = np.zeros([num_classes])

    baddsclist = []
    badsenlist = []
    badppvlist = []
    badasdlist = []
    img_id_list = []
    pred_list = []
    label_list = []

    with torch.no_grad():
            for i, img_id in enumerate(all_tests):
                add('\n')
                add('Predict for Image # {}'.format(img_id))
                o_img = nib.load(os.path.join(val_path, img_id)).get_fdata()
                o_img = np.transpose(o_img, [2, 1, 0])
                o_labels = nib.load(os.path.join(val_path, 'gth'+img_id[5:])).get_fdata()
                o_labels = np.transpose(o_labels, [2, 1, 0])
                print(o_img.shape)
                print(o_labels.shape)
                img = np.zeros(o_img.shape)
                labels = np.zeros(o_img.shape)
                for j in range(img.shape[0]):
                    img[j, ] = np.rot90(o_img[j, ][:], k=3)
                    labels[j, ] = np.rot90(o_labels[j, ][:], k=3)

                img = torch.tensor(img).to(device).float().unsqueeze(1)
                labels = torch.tensor(labels).to(device).long()

                print(img.size(), labels.size())
                if stack_image:
                    img, labels = stack_image(img, labels)
                label_size = labels.size()
                output = img.new(label_size[0], num_classes+1, label_size[1], label_size[2])

                sub_epoch = img.size(0) // test_batch_size
                for j in range(0, sub_epoch):
                    batch_start = j * test_batch_size
                    batch_end = min(j * test_batch_size + test_batch_size, img.size(0))
                    this_img = img[batch_start:batch_end]
                    with torch.no_grad():
                        decode_head_out, auxiliary_head_out, _, feat_list = model(this_img)
                        output[batch_start:batch_end, ...] = decode_head_out
                pred = output.argmax(dim=1)
                pred = pred.cpu().numpy()
                labels = labels.cpu().numpy()

                img_id_list.append(img_id)
                pred_list.append(pred)
                label_list.append(labels)

                for j in range(0, val_num_classes):  # Calc DSC and the mean for organs

                    tmppred = (pred==(j+1)).astype(np.int8)
                    tmplabels = (labels==(j+1)).astype(np.int8)

                    matOut = (tmppred).astype(np.int8)
                    labelimg = (tmplabels).astype(np.int8)

                    thisdsc = compute_dice_coefficient(labelimg, matOut)
                    thissen = sensi(matOut, labelimg)
                    thisppv = pospv(matOut, labelimg)
                    distDic = compute_surface_distances(labelimg, matOut, spacing_mm=(1, 1, 1))
                    thisasd = compute_average_surface_distance(distDic)
                    add('\tDSC\t||\tSEN\t||\tPPV\t||\tASD\t')
                    add('\t{:.4f}\t||\t{:.4f}\t||\t{:.4f}\t||\t{:.4f} , {:.4f}\t'.format(thisdsc, thissen, thisppv,
                                                                                         thisasd[0], thisasd[1]))

                    if thisdsc < badrate:
                        baddsclist.append(img_id + "_" + gtname[j])
                    alldsc[i, j] = thisdsc
                    meandsc[j], mediandsc[j], stddsc[j] = calMetrics(alldsc, i, j)

                    if thissen < badrate:
                        badsenlist.append(img_id + "_" + gtname[j])
                    allsen[i, j] = thissen
                    meansen[j], mediansen[j], stdsen[j] = calMetrics(allsen, i, j)

                    if thisppv < badrate:
                        badppvlist.append(img_id + "_" + gtname[j])
                    allppv[i, j] = thisppv
                    meanppv[j], medianppv[j], stdppv[j] = calMetrics(allppv, i, j)

                    if thisasd[0] > 1.5 or thisasd[1] > 1.5:
                        badasdlist.append(img_id + "_" + gtname[j])
                    allasd[i, j] = np.mean(thisasd)
                    meanasd[j], medianasd[j], stdasd[j] = calMetrics(allasd, i, j)
    add('')
    add('=' * 10 + 'SUMMARY' + '=' * 10)
    for i in range(num_classes):
        add('Evaluation for Tumor Type: {}'.format(gtname[i]))
        add('-' * 30)
        add('\tMean DSC\t|\tMedian DSC\t||\tMean SEN\t|\tMedian SEN\t||\tMean PPV\t|\tMedian PPV\t||\tMean ASD\t|\tMedian ASD\t')
        add('\t{:.4f}+/-{:.4f}\t|\t{:.4f}\t\t||\t{:.4f}+/-{:.4f}\t\t|\t{:.4f}\t||\t{:.4f}+/-{:.4f}\t|\t{:.4f}\t||\t{:.4f}+/-{:.4f}\t|\t{:.4f}\t'.format(
                meandsc[i], stddsc[i], mediandsc[i], meansen[i], stdsen[i], mediansen[i], meanppv[i], stdppv[i],
                medianppv[i],
                meanasd[i], stdasd[i], medianasd[i]))
    return np.mean(meandsc), np.mean(meanasd), img_id_list, pred_list, label_list


def main():
    global log_file
    log_file = open(os.path.join(result_dir, "TestLog.txt"), 'a+')
    test_data_path = os.path.join(data_path, target_dataset+'_test')

    if stack_image:
        model = Swin_Segmentor(pretrain_img_size=256, in_channel=3, num_classes=num_classes + 1)
        print(model)
    else:
        model = Swin_Segmentor(pretrain_img_size=256, in_channel=1, num_classes=num_classes + 1)
        print(model)

    if torch.cuda.device_count() > 1:
        add("Use " + str(torch.cuda.device_count()) + " GPUs!")
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    check_point_path = os.path.join(result_path, model_dir, model_name)
    add('Check Point: {} :'.format(check_point_path))
    checkpoint = torch.load(check_point_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    mean_dsc, mean_asd, img_id_list, pred_list, label_list = test(model, test_data_path)
    add('DSC: {};\tASD: {}'.format(mean_dsc, mean_asd))
    log_file.close()

    for i in range(len(img_id_list)):
        save_pred = np.transpose(pred_list[i], [2, 1, 0])
        save_pred = sitk.GetImageFromArray(save_pred.astype(np.int8))
        sitk.WriteImage(save_pred, os.path.join(result_dir, 'pred_' + img_id_list[i] + '.nii.gz'))
        save_label = np.transpose(label_list[i], [2, 1, 0])
        save_label = sitk.GetImageFromArray(save_label.astype(np.int8))
        sitk.WriteImage(save_label, os.path.join(result_dir, 'label_' + img_id_list[i] + '.nii.gz'))


if __name__ == '__main__':
    main()
