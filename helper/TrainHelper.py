from sklearn.model_selection import train_test_split

from models.RegressionTypeModel import RegressionTypeModel
from models.TestDataset import TestDataset
from models.TrainDataModel import TrainDataModel
from models.RegressionDataset import RegressionDataset
from PIL import Image
from skimage.io import imread
from skimage.transform import resize

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class TrainHelper:
    def get_file_name(self, type: RegressionTypeModel):
        if type == RegressionTypeModel.E_W_ROU or type == RegressionTypeModel.E_W_DEP:
            return " 주름(눈가) 분석사진.jpg"

        elif type == RegressionTypeModel.F_W_ROU or type == RegressionTypeModel.F_W_DEP:
            return " 주름(미간) 분석사진.jpg"

        elif type == RegressionTypeModel.C_A or type == RegressionTypeModel.C_O:
            return " 유분량(U존) 분석사진.jpg"

        elif type == RegressionTypeModel.F_A or type == RegressionTypeModel.F_O:
            return " 유분량(T존) 분석사진.jpg"

        elif type == RegressionTypeModel.M:
            return " 멜라닌(색소) 분석사진.jpg"

        elif type == RegressionTypeModel.NM:
            return " 멜라닌(비색소) 분석사진.jpg"

        else:
            raise Exception("Unknown type: %s" % type.name)

    def data_preprocess(self, type: RegressionTypeModel, CSV_PATH, DATA_PATH):
        labels = ['an_e_w_rou', 'an_e_w_dep', 'an_f_w_rou', 'an_f_w_dep', 'seb_c_o', 'seb_f_o', 'cor_c_a', 'cor_f_a',
                  'mex_m', 'mex_nm']
        csv_df = pd.read_csv(CSV_PATH)

        ids = csv_df['sam_num']

        imgs = []

        for id in ids:
            img_path = DATA_PATH + '/%s' % id.upper() + self.get_file_name(type)
            img = imread(img_path)
            imgs.append(resize(img, (224, 224, 1)))

        y = csv_df[labels[type.value]]
        imgs = np.array(imgs)
        y = np.array(y)

        print(y)

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        X_train, X_test, Y_train, Y_test = train_test_split(imgs, y, test_size=0.2, random_state=1016)

        train_dataset = RegressionDataset(X_train, Y_train, transform=transform)
        valid_dataset = RegressionDataset(X_test, Y_test, transform=transform)

        return train_dataset, valid_dataset