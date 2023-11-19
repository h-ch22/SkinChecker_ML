import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from helper.TrainHelper import TrainHelper
from models.RegressionTypeModel import RegressionTypeModel
from models.RegressionModel import RegressionModel
from torchsummary import summary

if __name__ == '__main__':
    device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

    TRAIN_PORTABLE_PATH = r'/Users/hachangjin/Desktop/2023/SkinChecker/DATA/TRAIN/PORTABLE'
    TRAIN_DETAIL_PATH = r'/Users/hachangjin/Desktop/2023/SkinChecker/DATA/TRAIN/DETAIL'

    TEST_PORTABLE_PATH = r'/Users/hachangjin/Desktop/2023/SkinChecker/DATA/TEST/PORTABLE'
    TEST_DETAIL_PATH = r'/Users/hachangjin/Desktop/2023/SkinChecker/DATA/TEST/DETAIL'

    TRAIN_CSV = r'/Users/hachangjin/Desktop/2023/SkinChecker/DATA/TRAIN.csv'
    TEST_CSV = r'/Users/hachangjin/Desktop/2023/SkinChecker/DATA/TEST.csv'

    helper = TrainHelper()

    train_dataset_0, valid_dataset_0 = helper.data_preprocess(RegressionTypeModel.E_W_ROU, TRAIN_CSV, TRAIN_PORTABLE_PATH)

    train_dataloader = DataLoader(train_dataset_0, batch_size=10, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset_0, shuffle=False)

    model = RegressionModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(summary(model, (3, 224, 224)))

    model = model.to(device)

    EPOCHS = 20
    last_best_acc = 0.0

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
            accuracy = 1 - (total_mse / total_samples)

            if average_mae > last_best_acc:
                last_best_acc = average_mae
                torch.save(model, './outputs/best.pt')

            print(f'Epoch {epoch}/{EPOCHS}\tloss: {loss.item()}\tAccuracy: {average_mae}\tBest: {last_best_acc}')
