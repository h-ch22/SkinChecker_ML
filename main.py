import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from helper.TrainHelper import TrainHelper
from models.RegressionTypeModel import RegressionTypeModel
from models.RegressionModel import RegressionModel
from torchsummary import summary


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


if __name__ == '__main__':
    device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

    TRAIN_PORTABLE_PATH = r'/Users/hachangjin/Desktop/2023/SkinChecker/DATA/TRAIN/PORTABLE'
    TRAIN_DETAIL_PATH = r'/Users/hachangjin/Desktop/2023/SkinChecker/DATA/TRAIN/DETAIL'

    TEST_PORTABLE_PATH = r'/Users/hachangjin/Desktop/2023/SkinChecker/DATA/TEST/PORTABLE'
    TEST_DETAIL_PATH = r'/Users/hachangjin/Desktop/2023/SkinChecker/DATA/TEST/DETAIL'

    TRAIN_CSV = r'/Users/hachangjin/Desktop/2023/SkinChecker/DATA/TRAIN.csv'
    TEST_CSV = r'/Users/hachangjin/Desktop/2023/SkinChecker/DATA/TEST.csv'

    helper = TrainHelper()

    model = RegressionModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(summary(model, (1, 224, 224)))

    MAES = []
    MSES = []
    LOSSES = []

    model = model.to(device)

    for i in range(0, 10):
        train_dataset, valid_dataset = helper.data_preprocess(RegressionTypeModel(i), TRAIN_CSV, TRAIN_PORTABLE_PATH)

        train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, shuffle=False)

        LOSS, MSE, MAE = train(i, model, train_dataloader, valid_dataloader)

        MAES.append(MAE)
        MSES.append(MSE)
        LOSSES.append(LOSS)

    print_result(MAES, MSES, LOSSES)
