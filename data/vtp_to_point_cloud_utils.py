import vtk
import numpy as np
from vtkmodules.numpy_interface import dataset_adapter as dsa
import os
def generate_vtps_list(vtp_dir:str,cut_type:str,train_ids_and_status:dict,val_ids_and_status:dict,test_ids_and_status:dict):
    train_vtp_files_paths = [vtp_dir+key+f'_{cut_type}'+'.vtp' for key in train_ids_and_status]
    # print(train_vtp_files_paths)
    val_vtp_files_paths = [vtp_dir+key+f'_{cut_type}'+'.vtp' for key in val_ids_and_status]
    test_vtp_files_paths = [vtp_dir+key+f'_{cut_type}'+'.vtp' for key in test_ids_and_status]
    all_vtp_files_paths = train_vtp_files_paths+val_vtp_files_paths+test_vtp_files_paths
    print(all_vtp_files_paths)

    return all_vtp_files_paths

def read_vtp_file(file_path:str):
    """读取VTP文件"""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()

    return polydata

# def randomly_sample_points(num_points:int, num_samples:int, point_count_lst:list=None):
#     """
#     deprecated
#     随机选取一定数量的点出来
#     :param num_samples: 希望每个点云各留（相同数量的）多少个点
#     :return:
#     """
#     # 获取点云数据
#     point_count_lst.append(num_points)
#
#     # 从点云中均匀随机选择指定数量的点
#     random_indices = np.random.choice(num_points, size=num_samples, replace=False)
#     sampled_points = vtk.vtkPoints()
#
#     for index in random_indices:
#         point = points.GetPoint(index)
#         sampled_points.InsertNextPoint(point)
#
#     # 创建采样后的PolyData对象
#     sampled_polydata = vtk.vtkPolyData()
#     sampled_polydata.SetPoints(sampled_points)
#
#     return sampled_polydata, point_count_lst

def polydata_to_numpy(polydata):
    """把VTP文件中保存的坐标们转储到nparray中去"""
    numpy_array_of_points = dsa.WrapDataObject(polydata).Points
    # numpy_array_of_points = vtk_to_numpy(polydata.GetPoints().GetData())
    return numpy_array_of_points
