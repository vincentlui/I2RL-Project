import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from network.network import PatientClassificationNet, PatientGroupNet
from argparse import ArgumentParser


def _train_classification_net(net, trainloader, testloader, threshold, epoch=5):
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(epoch):
        total_loss = 0
        net.train()
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            #             print(loss)
            loss.backward()
            optimizer.step()
            total_loss += loss

        net.eval()
        correct_true = 0
        predicted_true = 0
        target_true = 0
        total = 0
        for i, data in enumerate(testloader):
            inputs, labels = data
            outputs = net(inputs)
            prediction = outputs > threshold
            correct_true += ((prediction == 1) * (labels == 1)).sum().item()
            target_true += labels.sum().item()
            predicted_true += prediction.sum().item()
            total += labels.size(0)
        recall = correct_true / target_true
        precision = correct_true / predicted_true
        f1_score = 2 * precision * recall / (precision + recall)
        print(f'epoch {epoch}: recall: {recall} precision: {precision} f1_score: {f1_score} loss:{total_loss}')


def _prepare_data():
    df = pd.read_excel(r'data.xlsx')
    df = df[~df['Fallnummer'].isnull()]

    X = df.loc[:, ['AGE', 'Admission type 2', 'No. of times sent to ICU', 'FA ab para']]
    X['FA ab para'] = X['FA ab para'].replace([1, 5, 10, 14, 16, 21, 22], X['FA ab para'].max() + 1)
    X = torch.Tensor(X.to_numpy())
    fa_val = X[:, 3].unique()
    tmp = X[:, 3].unsqueeze(1).expand((-1, len(fa_val))) == fa_val
    X = torch.cat((X[:, :3], tmp.float()), dim=1)
    X[:, 2] = (X[:, 2] > 1)
    X = (X - X.mean(0)) / X.std(0)

    Y = df.loc[:, ['Tod']]
    Y = torch.Tensor(Y.to_numpy())

    num_feature = X.shape[-1]

    num_train = int(len(X) * 0.8)
    shuffle_indices = np.arange(len(X))
    np.random.shuffle(shuffle_indices)
    X = X[shuffle_indices]
    Y = Y[shuffle_indices]
    X_train = X[:num_train]
    y_train = Y[:num_train]
    X_test = X[num_train:]
    y_test = Y[num_train:]
    dataset_train = TensorDataset(X_train, y_train)
    dataset_test = TensorDataset(X_test, y_test)
    trainloader = DataLoader(dataset_train, batch_size=128, shuffle=True)
    testloader = DataLoader(dataset_test, batch_size=128, shuffle=True)

    return num_train, num_feature, trainloader, testloader, X_train, y_train, X_test, y_test


def _calculate_class_boundary(net, X, num_train, num_class, percetage):
    outputs = net(X)
    v, _ = outputs.sort(0)
    class_boundary = torch.Tensor([v[int(num_train * percetage[i] - 1)] for i in range(0, num_class)])
    return class_boundary


def _evaluate_patient_class(net, testloader, threshold):
    net.eval()
    correct_true = 0
    predicted_true = 0
    target_true = 0
    total = 0
    for i, data in enumerate(testloader):
        inputs, labels = data
        outputs = net(inputs)
        prediction = outputs > threshold
        correct_true += ((prediction == 1) * (labels == 1)).sum().item()
        target_true += labels.sum().item()
        predicted_true += prediction.sum().item()
        total += labels.size(0)
    recall = correct_true / target_true
    precision = correct_true / predicted_true
    f1_score = 2 * precision * recall / (precision + recall)
    print(f'recall: {recall} precision: {precision} f1_score: {f1_score}')
    return recall, precision, f1_score



def _evaluate_group(net, X_train, X_test):
    out = net(X_train)
    out2 = net(X_test)
    train_dist = [torch.sum(out == i).item() for i in range(1, net.num_class + 1)]
    test_dist = [torch.sum(out2 == i).item() for i in range(1, net.num_class + 1)]
    print('class distribution')
    print('train class distribution', train_dist)
    print('test class distribution', test_dist)
    return train_dist, test_dist


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--class", dest="num_class",
                        help="Number of class", default=3)
    parser.add_argument("-b", "--percentage", dest="percentage",
                        help="Percentage for class boundary",
                        nargs="*", type=float, default=[0.33, 0.67, 1])

    args = parser.parse_args()
    num_class = int(args.num_class)
    percentage = args.percentage

    num_train, num_feature, trainloader, testloader, X_train, y_train, X_test, y_test = _prepare_data()

    net = PatientClassificationNet(num_feature, 256)
    threshold = y_train.nonzero().size(0) / num_train
    _train_classification_net(net, trainloader, testloader, threshold, epoch=50)
    result = _evaluate_patient_class(net, testloader, threshold)

    class_boundary = _calculate_class_boundary(net, X_train, num_train, num_class, percentage)
    group_net = PatientGroupNet(net, class_boundary)

    train_dist, test_dist = _evaluate_group(group_net, X_train, X_test)

    PATH = './model/patient_group_net.pth'
    torch.save(group_net.state_dict(), PATH)

    # Write network info
    txt_path = './model/network_config.txt'
    f = open(txt_path, "w")
    f.write(f'num train: {num_train}' + '\n')
    f.write(f'num feature: {num_feature}' + '\n')
    f.write(f'Death classification metrics: {result}' + '\n')
    f.write(f'train group distribution: {train_dist}' + '\n')
    f.write(f'test group distribution: {test_dist}' + '\n')
    f.write(f'num_class: {len(class_boundary)}' + '\n')
    f.write(f'class boundary: {class_boundary}' + '\n')



if __name__ == '__main__':
    main()