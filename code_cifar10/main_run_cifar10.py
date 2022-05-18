from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os,time
os.environ['CUDA_VISIBLE_DEVICES'] = "-"

from spiking_cnn_model import*
names = 'spiking_cnn_model'
data_path = '.' # input your path

# Data preprocessing
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root= data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0,drop_last =True )

testset = torchvision.datasets.CIFAR10(root= data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0,drop_last =True)

net = SCNN()
net = net.to(device)

criterion = nn.MSELoss() # Mean square error loss
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])
optimizer = assign_optimizer(net,lrs=learning_rate)
# using SGD+CosineAnnealing could achieve better results
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-8)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # starts = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs  = inputs.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
        loss = criterion(outputs.cpu(), labels_)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.cpu().max(1)
        total += targets.size(0)

        correct += predicted.eq(targets).sum().item()
        if batch_idx%200==0:
            elapsed = time.time() -starts
            print(batch_idx,'Loss: %.5f | Acc: %.5f%% (%d/%d)'
                       %(train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Time past: ', elapsed, 's', 'Iter number:', epoch)
    loss_train_record.append(train_loss)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)

            labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), labels_)
            test_loss += loss.item()
            _, predicted = outputs.cpu().max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(testloader),'Loss: %.5f | Acc: %.5f%% (%d/%d)'
              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        loss_test_record.append(test_loss)


    # Save checkpoint.
    acc = 100.*correct/total
    acc_record.append(acc)


    if best_acc<acc:
        best_acc = acc
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
            'loss_train_record': loss_train_record,
            'loss_test_record': loss_test_record,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + names + '.t7')


for epoch in range(start_epoch, start_epoch+num_epochs):

    starts = time.time()
    train(epoch)
    test(epoch)
    elapsed =  time.time() - starts
    optimizer = lr_scheduler(optimizer, epoch, init_lr=learning_rate, lr_decay_epoch=35)
    print (" \n\n\n\n\n\n\n")
    print('Time past: ',elapsed,'s', 'Iter number:', epoch)