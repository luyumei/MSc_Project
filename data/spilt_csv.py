import pandas as pd
import os
import csv
df = pd.read_csv(filepath_or_buffer='./data/original_data/clinical.csv')#path for clinical.csv
grouped_df=df.groupby('source')
# grouped_df.groups.keys()#dict_keys(['aneurisk', 'aneurist', 'hug2016', 'hug2016snf'])
#hug2016(val)
hug2016=grouped_df.get_group(name='hug2016')
# hug2016['status'].value_counts()#U:R=263:87
hug2016.to_csv('./data/processed_data/hug2016.csv', index=False)
#hug2016snf(test)
hug2016snf=grouped_df.get_group(name='hug2016snf')
hug2016snf['status'].value_counts()#U:R=79:41
hug2016snf.to_csv('./data/processed_data/hug2016snf.csv', index=False)
#aneurisk+aneurist
aneurisk=grouped_df.get_group(name='aneurisk')
aneurist=grouped_df.get_group(name='aneurist')
aneurisk_and_aneurist=pd.concat([aneurisk, aneurist], axis=0)
# aneurisk_and_aneurist.shape#(265, 11)
aneurisk_and_aneurist['status'].value_counts()#R:U=133:132
aneurisk_and_aneurist.to_csv('./data/processed_data/aneurisk_and_aneurist.csv', index=False)


def csv_dataset_to_dict(dataset_path: str):
    """CSV->模型ID：破裂状态"""
    this_split_ids_and_status = {}
    with open(os.path.expanduser(dataset_path), 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # 跳过标题行
        # 遍历每一行并将指定列的数据存储到字典中
        for row in csvreader:  # 遍历每一行数据
            id_of_aneu = row[1]  # 获取第二列的值
            rupture_status = row[3]  # 获取第四列的值
            if rupture_status != '':
                this_split_ids_and_status[id_of_aneu] = rupture_status

    return this_split_ids_and_status


def display_split_balance(dictionary: dict):
    """训练集和测试集的"""
    unique_values = {}

    for value in dictionary.values():
        if value in unique_values:
            unique_values[value] += 1
        else:
            unique_values[value] = 1

    return print(unique_values)