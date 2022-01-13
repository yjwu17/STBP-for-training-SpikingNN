# -*- coding: utf-8 -*-
"""
Hybrid plasticity SNN on Fashion-MNIST

@author: yjwu

Python 3.5.2

"""

from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
from mnist_model import*

data_path =  r'./'
saving_names = '.'

train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

best_acc = 0.  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])
criterion = nn.MSELoss()
total_best_acc = []
total_acc_record = []
total_hid_state = []


snn = SNN_Model()
snn.to(device)
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)
acc_record = []
for epoch in range(num_epochs):
    running_loss = 0.
    snn.train()
    start_time = time.time()
    total = 0.
    correct = 0.

    for i, (images, targets) in enumerate(train_loader):
        snn.zero_grad()
        optimizer.zero_grad()

        images = images.float().to(device)
        outputs, spikes = snn(input=images , wins = time_window)
        targets_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
        loss = criterion(outputs.cpu(), targets_)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        if i % (60000/batch_size/4) == 0:
            print('Train Accuracy of the model : %.3f' % (100 * correct / total))
    print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'%(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,running_loss ))
    correct = 0.
    total = 0.
    running_loss = 0.

    optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)
    snn.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs, sumspike = snn(input=inputs,  wins=time_window)
            targets_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), targets_)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())

        acc = 100. * float(correct) / float(total)

    print('Iters:', epoch,'\n\n\n')
    print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))

    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)
    if epoch > 30:
        if acc > best_acc  :
            best_acc = acc
            print(acc)
            print('Saving..')
print(' best acc:', best_acc)





