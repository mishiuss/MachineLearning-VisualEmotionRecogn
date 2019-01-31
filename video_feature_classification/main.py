# -*- coding: utf-8 -*-
import os, sys
import cv2
import random
import numpy as np
from tqdm import tqdm
import pickle

sys.path.append('../')
from pytorch.common.datasets_parsers.av_parser import AVDBParser

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from accuracy import Accuracy


def get_data(dataset_root, file_list, max_num_clips=0, max_num_samples=50):
    dataset_parser = AVDBParser(dataset_root, os.path.join(dataset_root, file_list),
                                max_num_clips=max_num_clips, max_num_samples=max_num_samples,
                                ungroup=False, load_image=True)
    data = dataset_parser.get_data()
    print('clips count:', len(data))
    print('frames count:', dataset_parser.get_dataset_size())
    return data

def calc_features(data):
    orb = cv2.ORB_create()

    progresser = tqdm(iterable=range(0, len(data)),
                      desc='calc video features',
                      total=len(data),
                      unit='files')

    feat, targets = [], []
    for i in progresser:
        clip = data[i]

        # TODO: придумайте способы вычисления признаков по изображению с использованием ключевых точек
        # используйте библиотеку OpenCV
        if 0: # distance between landmarks
            for sample in clip.data_samples:
                dist = []
                lm_ref = sample.landmarks[30] # point on the nose
                for j in range(len(sample.landmarks)):
                    lm = sample.landmarks[j]
                    dist.append(np.sqrt((lm_ref[0] - lm[0]) ** 2 + (lm_ref[1] - lm[1]) ** 2))
                feat.append(dist)
                targets.append(sample.labels)
        elif 1: # descriptors of landmarks
            rm_list = []
            for sample in clip.data_samples:
                # make image border
                bordersize = 15
                border = cv2.copyMakeBorder(sample.image, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                                            borderType=cv2.BORDER_CONSTANT, value=[0]*3)

                # make keypoint list
                keypoints = []
                for k in range(18, 68):
                    keypoints.append(cv2.KeyPoint(x=sample.landmarks[k][0]+bordersize,
                                                  y=sample.landmarks[k][1]+bordersize,
                                                  _size=128))

                # compute the descriptors with ORB
                keypoints_actual, descriptors = orb.compute(border, keypoints)
                if len(keypoints_actual) != len(keypoints):
                    rm_list.append(sample)
                    continue

                descriptors = np.concatenate(descriptors)
                feat.append(descriptors)
                targets.append(sample.labels)

            for sample in rm_list:
                clip.data_samples.remove(sample)
        else:
            rm_list = []
            for j in range(len(clip.data_samples)):
                VolData = []
                target_blob = []
                bordersize = 25
                for k in range(-2, 3):
                    t = min(max(0, j+k), len(clip.data_samples)-1)
                    gray_img = cv2.cvtColor(clip.data_samples[t].image, cv2.COLOR_BGR2GRAY)
                    # make image border
                    gray_img = cv2.copyMakeBorder(gray_img, top=bordersize, bottom=bordersize, left=bordersize,
                                                right=bordersize,
                                                borderType=cv2.BORDER_CONSTANT, value=[0] * 3)
                    VolData.append(gray_img)
                    target_blob.append(clip.data_samples[t].labels)
                try:
                    feat.append(get_LBPTOP(np.asarray(VolData).transpose(1,2,0), clip.data_samples[j].landmarks[18:68], bordersize))
                    targets.append(np.median(target_blob))
                except:
                    rm_list.append(clip.data_samples[j])

            for sample in rm_list:
                clip.data_samples.remove(sample)

    print('feat count:', len(feat))
    return np.asarray(feat, dtype=np.float32), np.asarray(targets, dtype=np.float32)

def classification(X_train, X_test, y_train, y_test, accuracy_fn, pca_dim):
    if pca_dim > 0:
        pass
        # TODO: выполните сокращение размерности признаков с использованием PCA

    # shuffle
    combined = list(zip(X_train, y_train))
    #random.shuffle(combined)
    X_train[:], y_train[:] = zip(*combined)

    # TODO: используйте классификаторы из sklearn
    classifiers = []
    classifiers.append(RandomForestClassifier(n_estimators=150, max_depth=50))
    #classifiers.append(svm.SVC(kernel='linear', gamma=5.0, C=150))

    for clf in classifiers:
        print(clf)
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        accuracy_fn.by_frames(y_pred)
        accuracy_fn.by_clips(y_pred)


if __name__ == "__main__":
    experiment_name = 'exp_1'
    max_num_clips = 0 # загружайте только часть данных для отладки кода
    use_dump = False # используйте dump для быстрой загрузки рассчитанных фич из файла

    # dataset dir
    base_dir = '/media/olga/Data/Yandex_Disk/school_ML/DATABASES'
    if 1:
        train_dataset_root = base_dir + '/Ryerson/Video'
        train_file_list = base_dir + '/Ryerson/train_data_with_landmarks.txt'
        test_dataset_root = base_dir + '/Ryerson/Video'
        test_file_list = base_dir + '/Ryerson/test_data_with_landmarks.txt'
    elif 1:
        train_dataset_root = base_dir + '/OMGEmotionChallenge-master/omg_TrainVideos/preproc/frames'
        train_file_list = base_dir + '/OMGEmotionChallenge-master/omg_TrainVideos/preproc/train_data_with_landmarks.txt'
        test_dataset_root =base_dir + '/OMGEmotionChallenge-master/omg_ValidVideos/preproc/frames'
        test_file_list = base_dir + '/OMGEmotionChallenge-master/omg_ValidVideos/preproc/valid_data_with_landmarks.txt'

    if not use_dump:
        # load dataset
        train_data = get_data(train_dataset_root, train_file_list, max_num_clips=0)
        test_data = get_data(test_dataset_root, test_file_list, max_num_clips=0)

        # get features
        train_feat, train_targets = calc_features(train_data)
        test_feat, test_targets = calc_features(test_data)

        accuracy_fn = Accuracy(test_data, experiment_name=experiment_name)

        #with open(experiment_name + '.pickle', 'wb') as f:
        #    pickle.dump([train_feat, train_targets, test_feat, test_targets, accuracy_fn], f, protocol=2)
    else:
        with open(experiment_name + '.pickle', 'rb') as f:
            train_feat, train_targets, test_feat, test_targets, accuracy_fn = pickle.load(f)

    # run classifiers
    classification(train_feat, test_feat, train_targets, test_targets, accuracy_fn=accuracy_fn, pca_dim=100)