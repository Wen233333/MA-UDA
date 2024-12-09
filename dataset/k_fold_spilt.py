import os
import numpy as np


def k_fold_dataset_split(totaluse, ratio, dir_path):
    allfiles = os.listdir(dir_path)
    allfiles = allfiles[:int(len(allfiles) * totaluse)]
    numfiles = len(allfiles)
    num_files_part = int(numfiles * ratio)
    shufind = np.random.permutation(allfiles)
    filepart = [shufind[:num_files_part]]
    filepart.append(shufind[num_files_part:2 * num_files_part])
    filepart.append(shufind[2 * num_files_part:3 * num_files_part])
    filepart.append(shufind[3 * num_files_part:4 * num_files_part])
    filepart.append(shufind[4 * num_files_part:])

    for i in range(0, 5):
        trainfn = './ct_atlas2mri_chaos_train_%d' % (i + 1) + '.txt'
        testfn = './ct_atlas2mri_chaos_val_%d' % (i + 1) + '.txt'

        trainfiles = []
        for j in range(0, 5):
            if j == i:
                continue
            trainfiles.extend(filepart[j])

        testfiles = filepart[i]

        with open(trainfn, 'w') as f:
            for kk in trainfiles:
                f.write('%s' % kk)
                f.write('\n')

        with open(testfn, 'w') as f:
            for kk in testfiles:
                f.write('%s' % kk)
                f.write('\n')


if __name__ == '__main__':
    dir_path='/datapath'
    totaluse = 1
    ratio = 0.2
    k_fold_dataset_split(totaluse, ratio, dir_path)
