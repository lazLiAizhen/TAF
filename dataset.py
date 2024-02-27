import os
from utils import *
import torch
import numpy as np
import torch
import  random
from torch.utils.data import DataLoader, Dataset

def matrix_one_hot(x, class_count):
	return torch.eye(class_count)[x,:]

def seed_everything(seed=0):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(0)

def preprocess(args):#预处理
    dataset_path = args.dataset_path  #"../processed_data/"
    print("dataset_path: ", dataset_path)
    # normcounts = pd.read_csv(dataset_path + 'pbmc_expression.csv', index_col=0)  #
    # labels = pd.read_csv(dataset_path + 'pbmc_labels.csv', index_col=0)
    # domain_labels = pd.read_csv(dataset_path + 'batch_labels.csv', index_col=0,dtype={'batch':str})#!!!!
    # data_set = {'features': normcounts.values, 'labels': labels.iloc[:, 0].values,
    #            'accessions': domain_labels.iloc[:, 0].values}#数据集字典（特征、cell标签、域标签：mouse、human）

    print('loading dataset: %s' % args.dataset_name)
    if args.dataset_name == 'pbmc_':
        if args.dataselect == 0:
            all_set = np.load(os.path.join(args.dataset_path, 'pbmc_.npz'), allow_pickle=True)  ###allow_pickle参数
            args.train_set = {'features': all_set['features'][8098:], 'labels': all_set['labels'][8098:],
                              'accessions': all_set['accessions'][8098:]}
            args.test_set = {'features': all_set['features'][:8098], 'labels': all_set['labels'][:8098],
                             'accessions': all_set['accessions'][:8098]}  # 理解为1个batch--->目标域？测试
        if args.dataselect == 1:
            all_set = np.load(os.path.join(args.dataset_path, 'pbmc_.npz'), allow_pickle=True)  ###allow_pickle参数
            args.train_set = {'features': all_set['features'][:8098], 'labels': all_set['labels'][:8098],
                              'accessions': all_set['accessions'][:8098]}
            args.test_set = {'features': all_set['features'][8098:], 'labels': all_set['labels'][8098:],
                             'accessions': all_set['accessions'][8098:]}  #
        return 0

    elif args.dataset_name == 'pancreas_':
        all_set = np.load(os.path.join(args.dataset_path, 'original_pancreas.npz'), allow_pickle=True)  ###allow_pickle参数
        dataset={'features': all_set['features'], 'labels': all_set['labels'],
                            'accessions': all_set['accessions']}

        data_set=all_set
        ######方法1：######
        kwargs = {'num_workers': 0, 'pin_memory': True}  # 字典
        batch_size = args.batch_size
        source_name = args.source_name
        target_name = args.target_name
        # print(type(data_set['accessions'][0]))
        # print(data_set['accessions'])

        # pancreas
        domain_to_indices1 = np.where(data_set['accessions'] == source_name[0])[0]
        domain_to_indices2 = np.where(data_set['accessions'] == source_name[1])[0]
        domain_to_indices3 = np.where(data_set['accessions'] == source_name[2])[0]
        # print(type(domain_to_indices1))
        # print(domain_to_indices3.shape)
        source1_set = {'features': data_set['features'][domain_to_indices1],
                       'labels': data_set['labels'][domain_to_indices1],
                       'accessions': data_set['accessions'][domain_to_indices1]}  # 源域数据集1
        source2_set = {'features': data_set['features'][domain_to_indices2],
                       'labels': data_set['labels'][domain_to_indices2],
                       'accessions': data_set['accessions'][domain_to_indices2]}  # 源域数据集2
        source3_set = {'features': data_set['features'][domain_to_indices3],
                       'labels': data_set['labels'][domain_to_indices3],
                       'accessions': data_set['accessions'][domain_to_indices3]}  # 源域数据集3

        domain_to_indices_all = np.concatenate((domain_to_indices1, domain_to_indices2, domain_to_indices3))

        source_set = {'features': data_set['features'][domain_to_indices_all],
                       'labels': data_set['labels'][domain_to_indices_all],
                       'accessions': data_set['accessions'][domain_to_indices_all]}

        domain_to_indices = np.where(data_set['accessions'] == target_name)[0]
        target_set = {'features': data_set['features'][domain_to_indices],
                      'labels': data_set['labels'][domain_to_indices],
                      'accessions': data_set['accessions'][domain_to_indices]}  # 目标域数据集
        # print('source1 labels:', np.unique(source1_set['labels']), ' target labels:', np.unique(target_set['labels']))
        # target_set_eval = {'features': data_set['features'][domain_to_indices],
        #                    'labels': data_set['labels'][domain_to_indices],
        #                    'accessions': data_set['accessions'][domain_to_indices]}  # 目标域数据（测试）集--评估
        # print(source1_set['features'].shape, source2_set['features'].shape, source3_set['features'].shape,
        #       target_set['features'].shape)  # 输出源域、目标域样本维度

        source1_data = torch.utils.data.TensorDataset(
            torch.FloatTensor(source1_set['features']),
            torch.LongTensor(matrix_one_hot(source1_set['labels'], int(max(source1_set['labels']) + 1)).long()))####
        source1_loader = torch.utils.data.DataLoader(source1_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                                     **kwargs)  # 封装源域数据集1
        source2_data = torch.utils.data.TensorDataset(
            torch.FloatTensor(source2_set['features']),
            matrix_one_hot(source2_set['labels'], int(max(source2_set['labels']) + 1)).long()) ####
        source2_loader = torch.utils.data.DataLoader(source2_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                                     **kwargs)  # 封装源域数据集2

        source3_data = torch.utils.data.TensorDataset(
            torch.FloatTensor(source3_set['features']),
            torch.LongTensor(matrix_one_hot(source3_set['labels'], int(max(source3_set['labels']) + 1)).long())) ####
        source3_loader = torch.utils.data.DataLoader(source3_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                                     **kwargs)  # 封装源域数据集3

        target_data = torch.utils.data.TensorDataset(
            torch.FloatTensor(target_set['features']),
            torch.LongTensor(matrix_one_hot(target_set['labels'], int(max(target_set['labels']) + 1)).long()))
        target_train_loader = torch.utils.data.DataLoader(target_data, batch_size=batch_size, shuffle=True,
                                                          drop_last=True,
                                                          **kwargs)
        target_test_loader = torch.utils.data.DataLoader(target_data, batch_size=batch_size, shuffle=False,
                                                         drop_last=False,
                                                         **kwargs)  # 封装目标域数据集
        class_num = max(max(source1_set['labels']), max(source2_set['labels']),
                        max(source2_set['labels'])) + 1  # 细胞类型数量
        class_num_target = max(target_set['labels']) + 1
        # print(class_num, class_num_target)
        input_sizes_list = [source1_set['features'].shape[1], source2_set['features'].shape[1], \
                            source3_set['features'].shape[1], target_set['features'].shape[1]]  #####

        return source_set,target_set,source1_loader, source2_loader, source3_loader, \
               target_train_loader, target_test_loader,input_sizes_list,class_num,class_num_target

    else:
        print('Wrong name, cannot find the dataset.')
        return dataset_path
    # return args.train_set,args.test_set
    # return dataset


# class MyDataset_source(Dataset):
#     def __init__(self,data_X,label_y,x2,z):
#         super(MyDataset_source, self).__init__()
#         self.X, self.y =data_X,label_y
#         self.X2,self.Z=x2,z
#         print(self.X.shape)
#
#     def __iter__(self):
#         return list(zip(self.X, self.y,self.X2,self.Z))
#
#     def __len__(self):
#         return self.X.shape[0]
#
#     def __getitem__(self, item):
#         return item, self.X[item], self.y[item],self.X2[item],self.Z[item]

class MyDataset(Dataset):
    def __init__(self,data_X,label_y):
        super(MyDataset, self).__init__()
        self.X, self.y =data_X,label_y
        print(self.X.shape)

    def __iter__(self):
        return list(zip(self.X, self.y))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return item, self.X[item], self.y[item]

def matrix_one_hot(x, class_count):
	return torch.eye(class_count)[x,:]

def get_dataloader(args):
    train_set, test_set = preprocess(args)
    source_dataset=MyDataset(torch.FloatTensor(train_set['features']),
        torch.LongTensor(matrix_one_hot(train_set['labels'], int(max(train_set['labels']) + 1)).long()))
    target_dataset=MyDataset(torch.FloatTensor(test_set['features']),
                              torch.LongTensor(matrix_one_hot(test_set['labels'],int(max(test_set['labels']) + 1)).long()))
    return source_dataset,target_dataset


