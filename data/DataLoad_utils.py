"""
If points are not uniformly distributed across object’s surface -> could be difficult for PointNet to classify them

-> find a way to perform a uniform random sampling (face_weight=mesh.area)
i.e. uniformly sample a fixed number of points on the mesh faces (object’s surface) according to their face area
"""
from data.vtp_to_point_cloud import vtp_to_point_cloud
import numpy as np
import os
import pickle
from torch.utils.data import Dataset
from data.mesh_and_cut import *


num_samples = 1024  # 替换为您希望的采样点数 cut1 - 0.05:510, 0.01:2425; cut2 - 0.05:(1024) 0.01:3052; dome - 0.05:86, 0.01:384; ninja - 0.05:146, 0.01:611
def validate_mesh_resolution():
    print('mesh_resolution in DataLoad_utils:' + str(mesh_resolution))

def validate_cut_type():
    print('cut_type in DataLoad_utils:' + str(cut_type))
train_dict,val_dict,test_dict=vtp_to_point_cloud(mesh_resolution=mesh_resolution,cut_type=cut_type,num_samples=num_samples)



train_length = len(train_dict)
# print(train_length)
val_length = len(val_dict)
test_length= len(test_dict)

def normalise(three_col_coordinates):
    avg = np.mean(three_col_coordinates, axis=0)  # x的均值，y的均值，z的均值
    three_col_coordinates -= avg
    # std_dev = np.std(three_col_coordinates)  # 这种方法可能会导致不同坐标轴的缩放不一致，因为默认计算所有数据的标准差，而不是每个坐标轴的标准差
    # 每个坐标轴的缩放都一致，点云被缩放到一个单位球内
    distance_to_orig = np.sqrt(np.sum(three_col_coordinates ** 2, axis=1))
    max_distance_to_orig = np.max(distance_to_orig)
    coordinates_bounded_by_unit_sphere = three_col_coordinates / max_distance_to_orig

    return coordinates_bounded_by_unit_sphere

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def preprocess_data(split:str=None,mesh_resolutions:int=mesh_resolution,cut_type:str=cut_type,only_these_points:int=None):
    """

    :param split: train, val, test
    :param mesh_resolutions: 1, 5
    :param cut_type: cut1, cut2, dome, ninja
    :param only_these_points: cut1 - 0.05 max 510, 0.01 max 2429; dome - 0.05 max 86, 0.01 max 384; ninja - 0.05 max 146, 0.01 max 611
    :return:
    """
    pre_processed_data_path= f'./data/processed_data/ready2feed_point_cloud/{split}/area_00{mesh_resolutions}_{cut_type}_points_and_labels.dat'
    os.makedirs(name=os.path.dirname(pre_processed_data_path), exist_ok=True)
    dataset_dir=os.path.expanduser(f'./data/processed_data/extracted_point_cloud/area-00{mesh_resolutions}/{cut_type}/')
    if 'train'==split:
            list_of_points = [None] * train_length
            list_of_labels = [None] * train_length
            dict_switch=train_dict
    elif 'val'==split:
        list_of_points = [None] * val_length
        list_of_labels = [None] * val_length
        dict_switch=val_dict
    elif 'test'==split:
        list_of_points = [None] * test_length
        list_of_labels = [None] * test_length
        dict_switch=test_dict
    if not os.path.exists(pre_processed_data_path):
        print(f'Pre-processing {split} data to {pre_processed_data_path}')

        for count, id in enumerate(list(dict_switch.keys())):# e.g. p548_FAAtDQUSDAAsCRMVFgAxCQAC_LICA
            file_path = dataset_dir + id + f'_{cut_type}.csv'
            print(file_path)
            point_set = np.loadtxt(file_path, delimiter=',').astype(np.float32)  # 加载数据文件中的点云数据，将其转换为 NumPy 数组
            point_set = farthest_point_sample(point=point_set, npoint=only_these_points)
            # point_set = point_set[0:only_these_points, :]#直接取了点云的前这么多行
            point_set[:, 0:3] = normalise(point_set[:, 0:3])
            list_of_points[count] = point_set

            status=dict_switch[id]# ruptured/unruptured
            if 'ruptured' == status:
                label = 1
                label = np.array(label).astype(np.int32)#让它变成[0]/[1]
            elif 'unruptured' == status:
                label = 0
                label = np.array(label).astype(np.int32)
            list_of_labels[count] = label

        with open(pre_processed_data_path, 'wb') as f:
            pickle.dump([list_of_points, list_of_labels], f)
        print(f'Pre-processed {split} data saved to {pre_processed_data_path}')
        print('=' * 40)
    else:
        print(f'Found processed {split} data: {pre_processed_data_path}')
        with open(pre_processed_data_path, 'rb') as f:
            list_of_points, list_of_labels = pickle.load(f)
        print('Load succeeded!')
    return list_of_points,list_of_labels,len(dict_switch)

class AneuXDataLoader(Dataset):
    def __init__(self,split:str,mesh_resolutions:int,cut_type:str,how_many_points:int):
        self.list_of_points, self.list_of_labels,self.dataset_length=preprocess_data(split=split,mesh_resolutions=mesh_resolutions,cut_type=cut_type,only_these_points=how_many_points)
    def __len__(self):
        return self.dataset_length
    def __getitem__(self, index):
        point_set, label = self.list_of_points[index], self.list_of_labels[index]
        # point_set = point_set[:, 0:3]
        return point_set, label

def generate_dataset():
    train_dataset = AneuXDataLoader(split='train',mesh_resolutions=mesh_resolution,cut_type=cut_type,how_many_points=num_samples)
    val_dataset = AneuXDataLoader(split='val',mesh_resolutions=mesh_resolution,cut_type=cut_type,how_many_points=num_samples)
    test_dataset = AneuXDataLoader(split='test',mesh_resolutions=mesh_resolution,cut_type=cut_type,how_many_points=num_samples)

    return train_dataset,val_dataset,test_dataset