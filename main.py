import argparse
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from model import LeNet
from func import *

epochs = 10
b_size = 100
lr = 0.001
network = LeNet()
criterion = nn.CrossEntropyLoss()

batch = []
train_loss = []
test_loss = []
accuracy = []


dataset_train = torchvision.datasets.MNIST('~/mnist', train = True, download = True, transform = torchvision.transforms.ToTensor())
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = b_size, shuffle = True, num_workers = 0)
dataset_test = torchvision.datasets.MNIST('~/mnist', train = False, download = True, transform = torchvision.transforms.ToTensor())
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = b_size, shuffle = True, num_workers = 0)

def train():
    network.train()
    optimizer = torch.optim.Adam(network.parameters(), lr = lr)

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
            if( not (i+1) % (6000/b_size)):
                batch.append( 10*epoch + (i+1)/60)
                train_loss.append(loss.item())
                test()

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for (test_data, test_label) in dataloader_test:
            test_data = test_data.to(dev)
            test_label = test_label.to(dev)
            test_output = network(test_data)
            predicted = torch.max(test_output.data, 1)
            total += test_label.size(0)
            for i in range(len(predicted[1])):
                if(predicted[1][i] == test_label[i]):
                    correct = correct + 1
        loss = criterion(test_output, test_label)
        test_loss.append(loss.item())
        accuracy.append(float(correct/total))


if __name__ == "__main__":
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = network.to(dev)
    train()
    loss_fig(batch, train_loss, test_loss)
    acc_fig(batch, accuracy)
