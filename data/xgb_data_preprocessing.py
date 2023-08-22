import pandas as pd
import os
import numpy as np
df1 = pd.read_csv('data/clinical.csv')  # 替换为第一张表的文件路径
df2 = pd.read_csv('data/morpho-per-cut.csv')  # 替换为第二张表的文件路径
# df2
def merge_csvs(df1:pd.DataFrame=df1,df2:pd.DataFrame=df2):
    merged_data = []

    for index, row in df2.iterrows():
        id_index = index + 3  # 假设ID值在第二张表的第 n 行后（算表头吗？含到的那一行还是存在数据的那一行之前？）
        if id_index < df2.shape[0]:
            id_name = df2.iloc[id_index, 1]  # ID在第二张表的第二列
            matching_row = df1[df1['dataset'] == id_name]  # ID在第一张表的dataset列中

            if not matching_row.empty:
                # 提取第一张表中的某几列数值，假设你想要的列为'Column1'和'Column2'
                values_to_merge = matching_row[['location', 'side', 'sex', 'age', 'status']].values.flatten()
                merged_data.append(values_to_merge)
            else:
                merged_data.append([None, None, None, None, None])  # 如果找不到匹配的行，可以插入适当的标记
        else:
            break

    # 将合并的数据转换为DataFrame，并添加到第二张表中
    merged_df = pd.DataFrame(merged_data, columns=['location', 'side', 'sex', 'age', 'status'])
    df2 = pd.concat([df2, merged_df], axis=1)

    return df2

df2=merge_csvs(df1,df2)

def sink_rows(df2:pd.DataFrame=df2):
    columns_to_sink = ['location', 'side', 'sex', 'age', 'status']
    rows_to_sink = 3

    # 将指定的列数据下沉指定的行数
    for col in columns_to_sink:
        df2[col] = df2[col].shift(rows_to_sink)

    return df2

df2=sink_rows(df2)

# df2

def delete_index(df2:pd.DataFrame):
    # 要删除的连续行的起始索引和结束索引
    start_index = 0
    end_index = 2
    df2 = df2.drop(df2.index[start_index:end_index + 1])
    df2 = df2.reset_index(drop=True)
    df2 = df2.dropna(subset=['status'])

    return df2

df2= delete_index(df2)

#grouping
# 要删除的连续行的起始索引和结束索引
start_index = 0
end_index = 2
df2 = df2.drop(df2.index[start_index:end_index+1])
df2 = df2.reset_index(drop=True)
df2 = df2.dropna(subset=['status'])
# df2
df2_by_source=df2.groupby(by='type')
# df2_by_source.groups.keys()#dict_keys(['aneurisk', 'aneurist', 'hug2016', 'hug2016snf'])
aneurisk=df2_by_source.get_group(name='aneurisk')#train_df_part1
aneurist=df2_by_source.get_group(name='aneurist')#train_df_part2
aneurisk_aneurist=pd.concat(objs=[aneurisk, aneurist], axis=0)#train_df
# aneurisk_aneurist
hug2016=df2_by_source.get_group(name='hug2016')#val_df
hug2016snf=df2_by_source.get_group(name='hug2016snf')#test_df

def pre_process(df2:pd.DataFrame):
    # 要删除的列名
    column_to_delete = 'type'
    # 删除指定的列
    df2 = df2.drop(column_to_delete, axis=1)

    # 切割类型分组

    grouped = df2.groupby('Unnamed: 2')
    dfs_by_category = {category: group for category, group in grouped}

    # 打印每一组对应的 DataFrame
    for category, df_group in dfs_by_category.items():
        print(f"Category: {category}")
        print(df_group)
        print()

    return dfs_by_category

def generate_dataset(which_df:pd.DataFrame,cut_type:str):
    """

    :param which_df: aneurisk_aneurist,hug2016,or hug2016snf
    :param cut_type:
    :return:
    """
    dfs_by_category = pre_process(which_df)
    cut_type=cut_type
    cut1_df=dfs_by_category[cut_type]
    # cut1_df
    # 删除第一和第二列
    cut1_df = cut1_df.iloc[:, 2:]

    # 空着的年龄都计为0
    cut1_df['age'].fillna(0, inplace=True)
    cut1_df['side'].fillna('Unknown', inplace=True)

    # 去除所有包含空值的行
    cut1_df = cut1_df.dropna(axis=1)

    # 重置索引
    cut1_df = cut1_df.reset_index(drop=True)
    # cut1_df
    print('Has missing value:'+str(cut1_df.isna().any().any()))
    # cut1_df.dtypes
    cut1_df['location'] = pd.factorize(cut1_df['location'])[0]
    cut1_df['side'] = pd.factorize(cut1_df['side'])[0]
    cut1_df['sex'] = pd.factorize(cut1_df['sex'])[0]
    cut1_df['status'] = pd.factorize(cut1_df['status'])[0]
    # cut1_df
    cut1_df = cut1_df.astype(float)
    # cut1_df.dtypes
    cut1_train_df=cut1_df
    # cut1_df.iloc[:,28]
    cut1_train_X_ndarray=cut1_train_df.iloc[:, :-1].values
    cut1_train_y_ndarray=cut1_train_df.iloc[:, -1].values
    # cut1_train_X_ndarray[:,124]
    print('y_shape:'+str(cut1_train_X_ndarray.shape))
    print('y_shape:'+str(cut1_train_y_ndarray.shape))
    os.makedirs(name='./data/xgb', exist_ok=True)
    if which_df is aneurisk_aneurist:
        np.save(file=f'./data/xgb/{cut_type}_X_train_ndarray', arr=cut1_train_X_ndarray)
        np.save(file=f'./data/xgb/{cut_type}_y_train_ndarray', arr=cut1_train_y_ndarray)
    elif which_df is hug2016:
        np.save(file=f'./data/xgb/{cut_type}_X_val_ndarray', arr=cut1_train_X_ndarray)
        np.save(file=f'./data/xgb/{cut_type}_y_val_ndarray', arr=cut1_train_y_ndarray)
    elif which_df is hug2016snf:
        np.save(file=f'./data/xgb/{cut_type}_X_test_ndarray', arr=cut1_train_X_ndarray)
        np.save(file=f'./data/xgb/{cut_type}_y_test_ndarray', arr=cut1_train_y_ndarray)

generate_dataset(which_df=aneurisk_aneurist, cut_type='ninja')
generate_dataset(which_df=hug2016, cut_type='ninja')
generate_dataset(which_df=hug2016snf, cut_type='ninja')