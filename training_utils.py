from data.DataLoad_utils import generate_dataset
from torch.utils.data import DataLoader
import torch
from construct_PC_model import optimizer
import numpy as np
from tqdm import tqdm
# batch_size=10
# num_workers=0
# train_dataset,val_dataset,test_dataset=generate_dataset()
# train_DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# val_DataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
# test_DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# for point, label in test_DataLoader:
#     print(point.shape)
#     print(label.shape)

print(torch.__version__)

def judging_computation_device():
    """auto detect which device to use"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        x = torch.ones(1, device=device)
        print(x)
        print('Using device:', device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps:0")
        x = torch.ones(1, device=device)
        print(x)
        print('Using device:', device)
    else:
        device = "cpu"
        print("GPU acceleration device not found.")
        print('Using device:', device)

    return device

device=judging_computation_device()

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def val(model, loader, num_class=2):
    mean_correct = []#每个batch的准确率
    class_acc = np.zeros((num_class, 3))#每个类别的准确率、样本数和计数
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        points, target = points.to(device), target.to(device)

        points = points.transpose(2, 1)#适应模型输入的格式
        pred, _ = classifier(points)#前向传播，得到预测值 pred
        pred_choice = pred.data.max(1)[1]#预测的类别索引 pred_choice
        #计算每个类别的准确率
        print(np.unique(target.cpu()))
        for cat in np.unique(target.cpu()):#获取真实标签中的唯一类别，遍历每个类别
            #在每个类别上，计算预测类别等于真实类别的样本数，将其加到对应类别的 class_acc 中
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1#同时，将该类别的样本数加1

        correct = pred_choice.eq(target.long().data).cpu().sum()#计算预测类别与真实标签相等的样本数
        mean_correct.append(correct.item() / float(points.size()[0]))#将其添加到 mean_correct 列表中，用于后续计算平均准确率

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]#计算每个类别的准确率,class_acc 的第三列设为准确率除以样本数的比例
    class_acc = np.mean(class_acc[:, 2])#平均类别准确率
    instance_acc = np.mean(mean_correct)#平均实例准确率

    return instance_acc, class_acc