from cgi import test
import os
import random

# source_path = 'new_data/Ara-All_pathogen_train_'
# target_path = 'new_data/all_pair.txt'
# all_data = {}
# for i in range(10):
#     file_path = source_path + str(i) + '.txt'
#     with open(file_path, 'r') as f:
#         lines = list(f.readlines())
#         lines.pop(0)
#         for line in lines:
#             line = line.strip()
#             items = line.split(' ')
#             seq1_name = items[1]
#             seq2_name = items[2]
#             relation = items[0]
#             pair_name1 = seq1_name + '--' + seq2_name
#             pair_name2 = seq2_name + '--' + seq1_name
#             if pair_name1 not in all_data and pair_name2 not in all_data:
#                 all_data[pair_name1] = relation

# with open(target_path, 'w') as f:
#     for pair_name, relation in all_data.items():
#         f.write(pair_name + ' ' + relation + '\n')

# source_path = 'new_data/all_pair.txt'
# count_0 = 0
# count_1 = 0
# with open(source_path, 'r') as f:
#     for line in f.readlines():
#         line = line.strip()
#         items = line.split(' ')
#         relation = items[1]
#         if relation == '0':
#             count_0 += 1
#         else:
#             count_1 += 1

# print(count_0)
# print(count_1)

from utils import *


source_folder = 'new_data'
source_files = ['Gor.txt', 'Hpa.txt', 'Psy.txt']
pair_dict = {}
fa_dict = {}
fa_set = set()
for source_file in source_files:
    file_path = os.path.join(source_folder, source_file)
    with open(file_path) as fa:        
        for line in fa:
            # 去除末尾换行符
            line = line.strip()
            if line.startswith('>'):
                # 去除 > 号
                seq_names = line[1:]
                seq_name, others = seq_names.split('\t')
                others = others.split(',')
                for other in others:
                    fa_set.add(other)
                fa_set.add(seq_name)
                if len(others) >= 3:
                    train_num = int(len(others) * 0.8)
                    random.shuffle(others)
                    train_others, test_others = others[:train_num], others[train_num:]
                else:
                    train_others, test_others = others, []

                for other in train_others:
                    pair_name1 = seq_name + '--' + other
                    pair_name2 = other + '--' + seq_name
                    if pair_name1 not in pair_dict and pair_name2 not in pair_dict:
                        pair_dict[pair_name1] = 1
                        print(pair_name1, 1)

                for other in test_others:
                    pair_name1 = seq_name + '--' + other
                    pair_name2 = other + '--' + seq_name
                    if pair_name1 not in pair_dict and pair_name2 not in pair_dict:
                        pair_dict[pair_name1] = 0
                        print(pair_name1, 0)
                fa_dict[seq_name] = ''
            else:
                # 去除末尾换行符并连接多行序列
                fa_dict[seq_name] += line.replace('\n','')

file_path = 'new_data/Arabidopsis_sequences.fasta.txt'
with open(file_path) as fa:        
    for line in fa:
        # 去除末尾换行符
        line = line.strip()
        if line.startswith('>'):
            # 去除 > 号
            seq_name = line[1:]
            # fa_dict[seq_name] = ''
        else:
            # 去除末尾换行符并连接多行序列
            if seq_name in fa_set:
                fa_dict[seq_name] = line.replace('\n','')
            else:
                continue

aug_fa_dict = {}
aug_relation_dict = {}
word_size = 3
for seq_name, seq in fa_dict.items():
    seq_aug = [seq2ids(seq[start_index:], word_size) for start_index in range(word_size)]

    for sub_seq_id in range(len(seq_aug)):
        sub_seq_name = seq_name + '_' + str(sub_seq_id)
        for sub_seq_id2 in range(sub_seq_id + 1, len(seq_aug)):
            sub_seq_name2 = seq_name + '_' + str(sub_seq_id2)
            pair_name1 = sub_seq_name + '--' + sub_seq_name2
            pair_name2 = sub_seq_name2 + '--' + sub_seq_name
            if pair_name1 not in aug_relation_dict and pair_name2 not in aug_relation_dict:
                aug_relation_dict[pair_name1] = 1
                aug_relation_dict[pair_name2] = 1
        aug_fa_dict[sub_seq_name] = seq_aug[sub_seq_id]

for pair_name, relation in pair_dict.items():
    seq1_name, seq2_name = pair_name.split('--')
    for sub_seq_id1 in range(word_size):
        sub_seq_name1 = seq1_name + '_' + str(sub_seq_id1)
        for sub_seq_id2 in range(word_size):
            sub_seq_name2 = seq2_name + '_' + str(sub_seq_id2)
            pair_name1 = sub_seq_name1 + '--' + sub_seq_name2
            pair_name2 = sub_seq_name2 + '--' + sub_seq_name1
            if pair_name1 not in aug_relation_dict and pair_name2 not in aug_relation_dict:
                aug_relation_dict[pair_name1] = relation
                aug_relation_dict[pair_name2] = relation

aug_fa_name_list = {name:idx for idx, name in enumerate(aug_fa_dict.keys())}
aug_fa_name_list2 = {idx:name for name, idx in aug_fa_name_list.items()}

with open('data/all_seq_name.txt', 'w') as f:
    for idx in range(len(aug_fa_name_list2)):
        f.write(aug_fa_name_list2[idx] + '\n')
print('all_seq_name.txt done')

with open('data/all_seq.txt', 'w') as f:
    for idx in range(len(aug_fa_name_list2)):
        f.write(' '.join([str(id) for id in aug_fa_dict[aug_fa_name_list2[idx]]]) + '\n')
print('all_seq.txt done')

with open('data/all_edge.txt', 'w') as f:
    for pair_name, relation in aug_relation_dict.items():
        seq1_name, seq2_name = pair_name.split('--')
        f.write(str(aug_fa_name_list[seq1_name]) + ' ' + str(aug_fa_name_list[seq2_name]) + '\n')
print('all_edge.txt done')

for pair_name, relation in aug_relation_dict.items():
    if relation == 1:
        with open('data/train_edge.txt', 'a') as f:
            seq1_name, seq2_name = pair_name.split('--')
            f.write(str(aug_fa_name_list[seq1_name]) + ' ' + str(aug_fa_name_list[seq2_name]) + '\n')
    else:
        with open('data/test_edge.txt', 'a') as f:
            seq1_name, seq2_name = pair_name.split('--')
            f.write(str(aug_fa_name_list[seq1_name]) + ' ' + str(aug_fa_name_list[seq2_name]) + '\n')

print('train_edge.txt done')
print('test_edge.txt done')



