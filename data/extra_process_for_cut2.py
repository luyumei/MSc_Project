import os
def special_spilt_dict_for_cut2(vtp_dir:str,train_ids_and_status:dict,val_ids_and_status:dict,test_ids_and_status:dict,cut_type:str='cut2'):
    if cut_type == 'cut2':
        matching_files = []
        for filename in os.listdir(vtp_dir):
            if not filename.startswith('.'):  # and filename.endswith(f'_{cut_type}')
                parts = filename.rsplit('_', 1)
                new_filename = parts[0]  # 取消最后一个下划线及其之后的部分
                matching_files.append(new_filename)

        with open('./data/processed_data/cut2_matching_files.txt', 'w') as file:
            for item in matching_files:
                file.write(str(item) + '\n')

        train_ids_and_status = {key: train_ids_and_status[key] for key in
                                set(matching_files) & set(train_ids_and_status.keys())}
        val_ids_and_status = {key: val_ids_and_status[key] for key in set(matching_files) & set(val_ids_and_status.keys())}
        test_ids_and_status = {key: test_ids_and_status[key] for key in
                               set(matching_files) & set(test_ids_and_status.keys())}

        return train_ids_and_status,val_ids_and_status,test_ids_and_status