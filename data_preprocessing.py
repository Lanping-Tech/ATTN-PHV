import os
import pandas as pd

from utils import *

def read_original_excel_data(file_path, relation):
    plant_dict = {}
    pathogen_dict = {}
    relation_dict = {}
    data = pd.read_excel(file_path)
    for indexs in data.index:
        plant_name = data.loc[indexs][0]
        plant_seq = data.loc[indexs][1].replace(' ', '').upper()
        pathogen_name = data.loc[indexs][2]
        pathogen_seq = data.loc[indexs][3].replace(' ', '').upper()
        if plant_name not in plant_dict:
            plant_dict[plant_name] = plant_seq
        if pathogen_name not in pathogen_dict:
            pathogen_dict[pathogen_name] = pathogen_seq
        relation_name = plant_name + '--' + pathogen_name
        if relation_name not in relation_dict:
            relation_dict[relation_name] = relation
    
    return plant_dict, pathogen_dict, relation_dict

def read_original_excel_files(file_folder):
    plant_dict, pathogen_dict, relation_dict = {}, {}, {}
    for file_name in os.listdir(file_folder):
        print('读取文件：', file_name)
        relation = -1
        if '正' in file_name or '负' in file_name:
            if '正' in file_name:
                relation = 1
            if '负' in file_name:
                relation = 0
        else:
            raise Exception('无法判断文件{}是正样本还是负样本！！！'.format(file_name))

        file_path = os.path.join(file_folder, file_name)
        plants, pathogens, relations = read_original_excel_data(file_path, relation)
        plant_dict.update(plants)
        pathogen_dict.update(pathogens)
        relation_dict.update(relations)
    return plant_dict, pathogen_dict, relation_dict

def relation_data_augmentation(seq_name1, seq_name2, seq1, seq2, relation, word_size=3):
    seq1_aug = [seq2ids(seq1[start_index:], word_size) for start_index in range(word_size)]
    seq2_aug = [seq2ids(seq2[start_index:], word_size) for start_index in range(word_size)]
    relations_dict_aug = {}
    for id1 in range(len(seq1_aug)):
        seq1_aug_name1 = seq_name1 + '_' + str(id1)
        for id2 in range(id1+1, len(seq1_aug)):
            seq1_aug_name2 = seq_name1 + '_' + str(id2)
            relations_dict_aug[seq1_aug_name1 + '--' + seq1_aug_name2] = (seq1_aug[id1], seq1_aug[id2], 1)
    
    for id1 in range(len(seq2_aug)):
        seq2_aug_name1 = seq_name2 + '_' + str(id1)
        for id2 in range(id1+1, len(seq2_aug)):
            seq2_aug_name2 = seq_name2 + '_' + str(id2)
            relations_dict_aug[seq2_aug_name1 + '--' + seq2_aug_name2] = (seq2_aug[id1], seq2_aug[id2], 1)

    for id1 in range(len(seq1_aug)):
        seq1_aug_name = seq_name1 + '_' + str(id1)
        for id2 in range(len(seq2_aug)):
            seq2_aug_name = seq_name2 + '_' + str(id2)
            relations_dict_aug[seq1_aug_name + '--' + seq2_aug_name] = (seq1_aug[id1], seq2_aug[id2], relation)

    return relations_dict_aug


def data_augmentation(plant_dict, pathogen_dict, relation_dict):
    relation_dict_aug = {}
    for relation_name, relation in relation_dict.items():
        seq1_name, seq2_name = relation_name.split('--')
        seq1 = plant_dict[seq1_name]
        seq2 = pathogen_dict[seq2_name]
        relations_dict_aug_part = relation_data_augmentation(seq1_name, seq2_name, seq1, seq2, relation)
        for relation_name_aug, relation_aug in relations_dict_aug_part.items():
            seq_aug_name1, seq_aug_name2 = relation_name_aug.split('--')
            pair_name1 = seq_aug_name1 + '--' + seq_aug_name2
            pair_name2 = seq_aug_name2 + '--' + seq_aug_name1
            if pair_name1 not in relation_dict_aug and pair_name2 not in relation_dict_aug:
                relation_dict_aug[relation_name_aug] = relation_aug
    return relation_dict_aug

def export_data(relation_dict_aug):
    with open('data/data_aug.txt', 'w') as f:
        for relation_name, relation in relation_dict_aug.items():
            seq1, seq2, relation = relation
            seq1 = [str(i) for i in seq1]
            seq2 = [str(i) for i in seq2]
            seq1 = ','.join(seq1)
            seq2 = ','.join(seq2)
            f.write(seq1 + '\t' + seq2 + '\t' + str(relation) + '\n')
            f.write('{}\t{}\t{}\n'.format(seq1, seq2, relation))

if __name__ == '__main__':
    import collections
    file_folder = 'o_data'
    plant_dict, pathogen_dict, relation_dict = read_original_excel_files(file_folder)
    print('统计原始数据中的样本数量：', collections.Counter(list(relation_dict.values())))
    relation_dict_aug = data_augmentation(plant_dict, pathogen_dict, relation_dict)
    print('统计增强数据中的样本数量：', collections.Counter([item[2] for item in relation_dict_aug.values()]))
    export_data(relation_dict_aug)
    print('数据增广完成！！！')



        





    





