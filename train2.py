import torch
import torch.nn as nn
import torch.optim as optim
from predict.Data import StateData
from predict.Models import CustomPrediction

torch.set_default_tensor_type('torch.cuda.FloatTensor')
data_set = StateData('training_data.txt')
trainloader = torch.utils.data.DataLoader(data_set, batch_size=4,
                                          shuffle=True, num_workers=2)


net = CustomPrediction('train8.pd')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')