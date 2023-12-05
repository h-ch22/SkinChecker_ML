#
# System Environment
# OS: Windows 11 22H2(22621.2715)
# GPU: NVIDIA Geforce RTX 3060
# RAM: 32GB
# CUDA 12.3
# PyTorch 2.1.1
# Python 3.11
# Scikit-learn 1.3.2
# Scikit-image 0.22.0
# Numpy 1.24.1
# Pandas 2.1.3
# Matplotlib 3.8.2

import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
import sys

from torch.utils.data import DataLoader
from helper.TrainHelper import TrainHelper
from models.RegressionTypeModel import RegressionTypeModel
from models.RegressionModel import RegressionModel


def install_libraries():
    library_list = ["torch", "torchvision", "torchaudio", "scikit-learn", "scikit-image",
                    "numpy", "pandas", "matplotlib"]

    for lib in library_list:
        print("Installing Library %s" % lib)
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])


def print_result(MAES, MSES, LOSSES):
    f = open("Results.txt", "w+")

    for i in range(0, 10):
        f.write("Type: %d\t\tLoss:%.3f\t\tMAE: %.3f\t\tMSE: %.3f\n" % (i, LOSSES[i], MAES[i], MSES[i]))

    f.close()


def train(type, model, train_dataloader, valid_dataloader):
    EPOCHS = 100
    last_best_mae = 100.0
    B_MSE = 0.0
    B_LOSS = 0.0

    for epoch in range(EPOCHS):
        model.train()

        for x, y in train_dataloader:
            x, y = x.float(), y.float()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)

            loss = criterion(outputs, y.view(-1, 1).float())
            loss.backward()
            optimizer.step()

        model.eval()
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0

        with torch.no_grad():
            for x, y in valid_dataloader:
                x, y = x.float(), y.float()
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                mse = criterion(outputs, y.view(-1, 1).float())
                mae = nn.L1Loss()(outputs, y.view(-1, 1).float())

                total_mse += mse.item() * x.size(0)
                total_mae += mae.item() * x.size(0)
                total_samples += x.size(0)

                print(f'label: {y}\tpredicted: {outputs}')

            average_mse = total_mse / len(valid_dataloader)
            average_mae = total_mae / len(valid_dataloader)

            if average_mae < last_best_mae:
                last_best_mae = average_mae
                B_LOSS = loss.item()
                B_MSE = average_mse
                torch.save(model, './outputs/best_%d.pt' % type)

            print(
                f'Epoch {epoch}/{EPOCHS}\tloss: {loss.item()}\tMSE: {average_mse}\tMAE: {average_mae}\tBest: {last_best_mae}')

    return B_LOSS, B_MSE, last_best_mae


def write_results(TEST_CSV, TEST_PORTABLE_PATH):
    results = []

    for i in range(0, 10):
        test_dataset = helper.test_data_preprocess(RegressionTypeModel(i), TEST_CSV, TEST_PORTABLE_PATH)
        test_loader = DataLoader(test_dataset)
        model = torch.load(r'./outputs/best_%d.pt' % i)
        model = model.to(device)

        for x in test_loader:
            x = x.to(device)
            x = x.float()
            outputs = model(x)
            results.append(outputs)

    print(results)

if __name__ == '__main__':
    install_libraries()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    TRAIN_PORTABLE_PATH = r'./data/TRAIN/PORTABLE'
    TRAIN_DETAIL_PATH = r'./data/TRAIN/DETAIL'

    TEST_PORTABLE_PATH = r'./data/TEST/PORTABLE'
    TEST_DETAIL_PATH = r'./data/TEST/DETAIL'

    TRAIN_CSV = r'./data/TRAIN.csv'
    TEST_CSV = r'./data/TEST.csv'

    helper = TrainHelper()

    write_results(TEST_CSV, TEST_PORTABLE_PATH)

    model = RegressionModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = model.to(device)

    MAES = []
    MSES = []
    LOSSES = []

    for i in range(0, 10):
        train_dataset, valid_dataset = helper.data_preprocess(RegressionTypeModel(i), TRAIN_CSV, TRAIN_PORTABLE_PATH)

        train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, shuffle=False)

        LOSS, MSE, MAE = train(i, model, train_dataloader, valid_dataloader)

        MAES.append(MAE)
        MSES.append(MSE)
        LOSSES.append(LOSS)

    print_result(MAES, MSES, LOSSES)
