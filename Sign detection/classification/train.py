import model_Le
from Data_loader import *
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


a,b = load_traffic_sign_data('E:\ex_python\sign_detection\\backup\\train.p')

trainloader = DataLoader(
    dataset=MyDataset(images=a,labels=b),
    batch_size=4,
    shuffle=True
    )
model = model_Le.LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

loss = 10000000000000

def train_model(model,train_loader,optimizer,epochs=5):
    # loss_sum = 0
    # m_loss =[]
    # m_epoch = []
    # m_train_acc =[]
    for epoch in range(epochs):
        correct = 0
        for index, (data,target) in enumerate(train_loader):
            out = model(data)
            target = target.type(torch.LongTensor)
            loss = criterion(out,target.long())
            pred = out.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss_sum += loss.item()

        # loss_ave = loss_sum/(epoch+1)
        print("epoch:%s,loss:%s,train_acc:%s" % (epoch+1, loss.item(), correct / len(train_loader.dataset)))
        # m_epoch.append(epoch+1)
        # m_loss.append(loss.item())
        # m_train_acc.append(correct / len(train_loader.dataset))
    # plt.plot(m_epoch, m_train_acc)
    # plt.title('train accuacy')
    # plt.xlabel('epoch')
    # plt.ylabel('test_acc/loss')
    # plt.show()
    # plt.plot(m_epoch, m_loss)
    #
    # plt.title('train loss')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.show()

    # torch.save(model.state_dict(), 'E:\ex_python\sign_detection\\backup\\trained_model.pth')
    # print('trained_model1.pth was saved')

if __name__=='__main__':
    train_model(model,trainloader,optimizer,epochs=50)