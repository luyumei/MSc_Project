from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from PC_model_blocks import SetAbstraction_3_branches, SetAbstraction_1_branch

# num_class = 2

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

class whole_model(nn.Module):
    def __init__(self,num_class,normal_channel=False):
        super(whole_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        # 点云特征提取模块，用于从点云数据中提取局部和全局特征
        self.sa1 = SetAbstraction_3_branches(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = SetAbstraction_3_branches(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = SetAbstraction_1_branch(None, None, None, 640 + 3, [256, 512, 1024], True)
        # 全连接层和批归一化层，用于进行特征的线性变换和归一化
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        # Dropout层，用于在训练过程中进行随机失活，以防止过拟合
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        # 根据是否有法线通道，提取输入点云的坐标和法线信息
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        # LogSoftmax层，用于在模型的输出上应用log softmax激活函数
        x = F.log_softmax(x, -1)

        # 在第三个PointNetSetAbstractionMsg模块中提取的全局特征，以供后续任务使用
        return x,l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss

classifier = whole_model(num_class=2, normal_channel=False)
criterion = get_loss()
classifier.apply(inplace_relu)

learning_rate=0.001

optimizer = Adam(
    classifier.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    eps=1e-08
)

