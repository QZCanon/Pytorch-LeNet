import torch
import torchvision
import torch.utils
import torch.nn as nn
from LeNet import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib .pyplot as plt
import numpy as np

if __name__ == '__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=36,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                             shuffle=False, num_workers=0)
    test_data_iter = iter(testloader)
    test_image, test_label = test_data_iter.next()

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # def imshow(img):
    #     img = img / 2 + 0.5     # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # # print labels
    # print(' '.join(f'{classes[test_label[j]]:5s}' for j in range(4)))
    # # show images
    # imshow(torchvision.utils.make_grid(test_image))
    net = LeNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):
        running_loss = 0.0 # 损失累加
        for step, data in enumerate(trainloader, start=0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 500 == 499:
                with torch.no_grad():
                    outputs = net(test_image)
                    pre_y = torch.max(outputs, dim=1)[1] # 找最大的index，[1]表示只需要index，不需要知道值是多少
                    accuracy = (pre_y == test_label).sum().item() / test_label.size(0)
                    print('[%d, %5d], train_loss: %0.3f test_accuracy: %0.3f' %
                          (epoch + 1, step+1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('finished traning')
    save_path = './LeNet.pth'
    torch.save(net.state_dict(), save_path)
