"""
meshes represented only by "triangular" faces???

convert a mesh file to a point cloud
"""
import subprocess
from data.spilt_csv import csv_dataset_to_dict,display_split_balance
from data.extra_process_for_cut2 import special_spilt_dict_for_cut2
from data.vtp_to_point_cloud_utils import generate_vtps_list,read_vtp_file,polydata_to_numpy
import os
import numpy as np
import matplotlib.pyplot as plt
def visualise_point_cloud(numpy_array):
    # 创建一个3D子图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    ax.scatter(numpy_array[:, 0], numpy_array[:, 1], numpy_array[:, 2], s=5, c='b', marker='o')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
def vtp_to_point_cloud(mesh_resolution:int,cut_type:str,num_samples:int):
    subprocess.run(["python", './data/spilt_csv.py'])

    train_ids_and_status = csv_dataset_to_dict(dataset_path='./data/processed_data/aneurisk_and_aneurist.csv')
    val_ids_and_status = csv_dataset_to_dict(dataset_path='./data/processed_data/hug2016.csv')
    test_ids_and_status = csv_dataset_to_dict(dataset_path='./data/processed_data/hug2016snf.csv')

    # mesh_resolution=1
    # cut_type='ninja'
    vtp_dir=f"./data/original_data/remeshed/area-00{mesh_resolution}/{cut_type}/"
    if cut_type == 'cut2':
        train_ids_and_status, val_ids_and_status, test_ids_and_status=special_spilt_dict_for_cut2(vtp_dir=vtp_dir,train_ids_and_status=train_ids_and_status,val_ids_and_status=val_ids_and_status,test_ids_and_status=test_ids_and_status)

    display_split_balance(dictionary=train_ids_and_status)
    display_split_balance(dictionary=val_ids_and_status)
    display_split_balance(dictionary=test_ids_and_status)

    all_vtp_files_paths=generate_vtps_list(vtp_dir=vtp_dir,cut_type=cut_type,train_ids_and_status=train_ids_and_status,val_ids_and_status=val_ids_and_status,test_ids_and_status=test_ids_and_status)
    # num_samples = 1024  # 替换为您希望的采样点数 cut1 - 0.05:510, 0.01:2425; cut2 - 0.05:(1024) 0.01:3052; dome - 0.05:86, 0.01:384; ninja - 0.05:146, 0.01:611
    # 点云即将的存储目录
    csvs_directory = f"./data/processed_data/extracted_point_cloud/area-00{mesh_resolution}/{cut_type}/"
    # 创建新目录（如果不存在）
    if not os.path.exists(csvs_directory):
        os.makedirs(csvs_directory)

    # point_count_lst=[]
    # count=0
    for this_path in all_vtp_files_paths:
        # count+=1
        polydata = read_vtp_file(file_path=this_path)
        points = polydata.GetPoints()
        num_points = points.GetNumberOfPoints()
        if num_points < num_samples : continue
        # 进行均匀随机采样
        # sampled_polydata, _ = randomly_sample_points(num_points=num_points, num_samples=num_samples, point_count_lst=point_count_lst)
        # 这部分数据集各有多少个点
        # point_count_lst.sort()
        # 转换为NumPy数组
        numpy_array = polydata_to_numpy(polydata=polydata)

        # if 1==count: break
        # CSV文件名们
        csv_file_name = os.path.basename(this_path).replace(".vtp", ".csv")#提取原路径中的文件名
        # CSV储存路径
        output_path = os.path.join(csvs_directory, csv_file_name)
        np.savetxt(output_path, numpy_array, delimiter=",", fmt='%.6f')

    print(numpy_array)
    print(numpy_array.shape)
    # visualise_point_cloud(numpy_array)
    # print(point_count_lst)
    # print(len(point_count_lst))

    matching_files = []
    for filename in os.listdir(csvs_directory):
        if not filename.startswith('.'):# and filename.endswith(f'_{cut_type}')
            parts = filename.rsplit('_', 1)
            new_filename = parts[0]  # 取消最后一个下划线及其之后的部分
            matching_files.append(new_filename)

    # print(matching_files)
    train_dict = {key: train_ids_and_status[key] for key in set(matching_files) & set(train_ids_and_status.keys())}
    # print(train_dict)
    val_dict = {key: val_ids_and_status[key] for key in set(matching_files) & set(val_ids_and_status.keys())}
    test_dict = {key: test_ids_and_status[key] for key in set(matching_files) & set(test_ids_and_status.keys())}

    return train_dict,val_dict,test_dict