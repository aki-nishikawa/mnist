import argparse
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from model import LeNet
import time

epochs = 10
b_size = 100

dataset_train = torchvision.datasets.MNIST('~/mnist', train = True, download = True, transform = torchvision.transforms.ToTensor())
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = b_size, shuffle = True, num_workers = 0)
dataset_test = torchvision.datasets.MNIST('~/mnist', train = False, download = True, transform = torchvision.transforms.ToTensor())
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = b_size, shuffle = True, num_workers = 0)

def train():
    network.train()
    optimizer = torch.optim.Adam(network.parameters(), lr = 0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for i, (train_data, train_label) in enumerate(dataloader_train, 0):
            train_data = train_data.to(dev)
            train_label = train_label.to(dev)
            optimizer.zero_grad()
            train_output = network(train_data)
            loss = criterion(train_output, train_label)
            loss.backward()
            optimizer.step()
            print("[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
                % (epoch, epochs, i, len(dataloader_train), loss.item())
            )

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, test_labels) in dataloader_test:
            images = images.to(dev)
            test_labels = test_labels.to(dev)
            test_outputs = network(images)
            predicted = torch.max(test_outputs.data, 1)
            total += test_labels.size(0)
            for i in range(len(predicted[1])):
                if(predicted[1][i] == test_labels[i]):
                    correct = correct + 1
    print('Accuracy: {:.2f} %'.format(100 * float(correct/total)))


if __name__ == "__main__":
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #dev = 'cpu'
    network = LeNet()
    network = network.to(dev)
    print("START")
    tim_s = time.perf_counter()
    train()
    test()
    tim = time.perf_counter() - tim_s
    print("END")
    print("所要時間%d秒"%(tim))
