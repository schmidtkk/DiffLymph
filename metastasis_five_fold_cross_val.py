import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gc


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 正样本的权重系数
        self.gamma = gamma  # 调节因子

    def forward(self, inputs, targets):
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # pt是预测概率
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # 第一层：3 -> 64
        self.fc2 = nn.Linear(64, 128)  # 第二层：64 -> 128
        self.fc3 = nn.Linear(128, 64)  # 第三层：128 -> 64
        self.fc4 = nn.Linear(64, 32)  # 第四层：64 -> 32
        self.fc5 = nn.Linear(32, 1)  # 输出层：32 -> 1
        self.dropout = nn.Dropout(0.1)  # 使用Dropout防止过拟合
        self.relu = nn.ReLU()  # 使用ReLU激活函数
        self.sigmoid = nn.Sigmoid()  # 使用Sigmoid激活函数将输出转换为概率
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x


torch.cuda.set_device(7)  # 设置使用的GPU
all_features = []
all_labels = []
all_cases = []
with open('metastasis-short-long-axis.csv') as f:
    lines = f.readlines()[1:]
# with open('/mnt/889cdd89-1094-48ae-b221-146ffe543605/gwd/dataset/RecT500/PRLN/prln_target.csv') as f:
#     targets = f.readlines()
for line in lines:
    features = [float(x) for x in line.strip().split(',')[-3:]]
    label = 0 if line.strip().split(',')[2] == '6' else 1
    all_features.append(torch.tensor(features))
    all_labels.append(label)
    all_cases.append(line.strip())

features = torch.stack(all_features)
labels = torch.tensor(all_labels)

torch.manual_seed(1234)  # 设置随机种子
perm = torch.randperm(len(features))  # 随机打乱数据集的索引
features = features[perm]
labels = labels[perm]
cases = [all_cases[i] for i in perm]

# divide into five folds
fold_size = len(features) // 5
feature_folds = torch.split(features, fold_size)
label_folds = torch.split(labels, fold_size)
cases_folds = [cases[i:i+fold_size] for i in range(0, len(cases), fold_size)]
feature_folds = list(feature_folds)
label_folds = list(label_folds)
if len(feature_folds) != 5:
    feature_folds[4] = torch.cat([feature_folds[4], feature_folds[5]])
    label_folds[4] = torch.cat([label_folds[4], label_folds[5]])
    cases_folds[4] += cases_folds[5]
    feature_folds.pop()
    label_folds.pop()
    cases_folds.pop()
gc.collect()

with open('metastasis_critical_cases.csv', 'w') as f:
    f.write('')

BATCH_SIZE = 64
for i in range(5):
    train_X = torch.cat([feature_folds[j] for j in range(5) if j != i])
    train_y = torch.cat([label_folds[j] for j in range(5) if j != i])
    test_X = feature_folds[i]
    test_y = label_folds[i]
    inds = torch.arange(len(test_y))
    test_y = torch.stack([test_y, inds], dim=1)
    test_cases = cases_folds[i]
    
    train_dataset = TensorDataset(train_X, train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = TensorDataset(test_X, test_y)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleClassifier().cuda()
    criterion = FocalLoss(alpha=5, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    

    model.train()  # 训练模式
    num_epochs = 100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_dataloader:
            batch_X = batch_X.cuda()
            batch_y = batch_y.cuda()
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, LR: {scheduler.get_last_lr()[0]:.6f}, Loss: {loss.item():.4f}')
        # scheduler.step()

    model.eval()  # 测试模式
    with torch.no_grad():
        correct = 0
        total = 0
        critical_cases = []
        for batch_X, batch_y in test_dataloader:
            batch_X = batch_X.cuda()
            batch_y = batch_y.cuda()
            outputs = model(batch_X)
            logits = outputs.squeeze()
            predicted = logits > 0.5
            total += batch_y.size(0)
            ground_truth = batch_y[:, 0]
            inds = batch_y[:, 1]
            correct += (predicted == ground_truth).sum().item()
            critical_cases += [test_cases[i] for i in range(len(ground_truth))
                               if predicted[i] != ground_truth[i] and abs(logits[i] - ground_truth[i]) >= 0.8]
        
        with open('metastasis_critical_cases.csv', 'a') as f:
            for case in critical_cases:
                f.write(case + '\n')
        
        print(f'Accuracy of the model on the {i+1}-th fold: {100 * correct / total:.2f}%')


