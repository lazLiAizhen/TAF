import argparse#参数解析 #1、引入模块
import os
import numpy as np
import torch
import  random
from dataset import preprocess
from dataset import get_dataloader
from dataset import matrix_one_hot
from TAF import *
from models import *
# from TAF import  *  #7networks
from TAF_shareweight import *  #5networks
def seed_everything(seed=666):
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

seed_everything(666)

if __name__ == "__main__":
    seed_everything(666)
    parser = argparse.ArgumentParser(
        description='scDRLN: Domain adversarial representation learning Network')  # 2、建立解析对象
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')  # 256  #3、增加属性：给xx实例增加一个aa属性 # xx.add_argument("aa")
    parser.add_argument('--embedding_size', type=int, default=256, help='embedding_size')  # 256   100
    parser.add_argument('--source_name', type=str, default=[0,1,3])#'pancreas_'
    parser.add_argument('--target_name', type=str, default=2)
    parser.add_argument('--dataset_name', type=str, default='pancreas_')  # _single     'pancreas_'
    parser.add_argument('--dataselect', type=int, default=0)
    parser.add_argument('--result_path', type=str, default='./results/')
    parser.add_argument('--dataset_path', type=str, default='./processed_data/')
    parser.add_argument('--num_iterations', type=int, default=50010, help="num_iterations")  # 50010  3010
    # parser.add_argument('--centerloss_coeff', type=float, default=1.0,
    #                     help='regularization coefficient for center loss')  # 中心损失正则化系数#2.0  1.0
    parser.add_argument('--centroid_coeff', type=float, default=1.0,
                        help='regularization coefficient for semantic loss')  # 语义损失正则化系数 #2.0 1.5
    parser.add_argument('--DA_coeff', type=float, default=1.0,
                        help="regularization coefficient for domain alignment loss")  # 域对齐损失正则化系数 #1.0!  #0.8
    parser.add_argument('--GA_coeff', type=float, default=1.0,
                        help="regularization coefficient for deep GCCA loss")
    parser.add_argument('--Cl_coeff', type=float, default=1.0,
                        help="regularization coefficient for classifier cl loss")
    # parser.add_argument('--ls_m', type=float, default=0.1,
    #                     help="regularization coefficient for source margin loss")  # 源域margin对齐
    parser.add_argument('-son_embeding', '--son_embeding', type=int, default=128,
                        help='da_sonnet')#1.0
    parser.add_argument('--cell_th', type=int, default=20, help='cell_th')
    parser.add_argument('--epoch_th', type=int, default=15000, help='epoch_th')  # 15000  1000
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('-c', '--ckpts', type=str, default='ckpts/', help='model checkpoints path')
    parser.add_argument("-o", "--output", type=str, default='original_pancreas_single',
                        help='Save model filepath')  # scquery
    parser.add_argument('--lm_coeff', type=float, default=1.0,
                        help="regularization coefficient for contrastive loss")#0.0001
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient for VAT loss')  # VAT损失正则化系数
    parser.add_argument('--xi', type=float, default=10.0, metavar='XI',
                        help='hyperparameter of VAT')
    parser.add_argument('--eps', type=float, default=1.0, metavar='EPS',
                        help='hyperparameter of VAT')
    parser.add_argument('--ip', type=int, default=1, metavar='IP',
                        help='hyperparameter of VAT')

    parser.add_argument('--BNM_coeff', type=float, default=0.2, help="regularization coefficient for BNM loss")  #
    # parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
    #                     help='regularization coefficient for VAT loss')  # VAT损失正则化系数
    # parser.add_argument('--xi', type=float, default=10.0, metavar='XI',
    #                     help='hyperparameter of VAT')
    # parser.add_argument('--eps', type=float, default=1.0, metavar='EPS',
    #                     help='hyperparameter of VAT')
    # parser.add_argument('--ip', type=int, default=1, metavar='IP',
    #                     help='hyperparameter of VAT')


    args = parser.parse_args()  # 4、属性给与args实例：add_argument 返回到 args 子类实例
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id  # '0,1,2,3'
    print(args)

    torch.cuda.empty_cache()  # 释放显存

    ######0######
    # train_set, test_set = preprocess(args)  ###!
    # source_dataset, target_dataset = get_dataloader(args)
    # print(source_dataset,target_dataset)

    ######1######
    # preprocess(args)  ###
    #
    #已封装到dataset

    # scDRLN(args, train_set, test_set)
    if not os.path.exists(args.ckpts):
        os.mkdir(args.ckpts)

    model_path = os.path.join(args.ckpts, args.output)  # os.path.join()函数用于路径拼接文件路径，可以传入多个路径
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    log_file = 'dataset.%s_output%s_dataselect%d_lambda%.2f_lm_coeff%.2f_semantic%.2f_reconstruct%.2f_epoch%d_batch_size%d.txt' \
               % (args.dataset_name, args.output, args.dataselect, args.DA_coeff, args.lm_coeff,
                  args.centroid_coeff, args.GA_coeff, args.num_iterations, args.batch_size)
    with open(os.path.join(args.ckpts, args.output, log_file), 'w') as fw:
        # fw.write(str(args)+'\n')
        # scDRLN(args, train_set, test_set, source_dataset, target_dataset, fw)

        ######方法1：0######
        # batch_size = args.batch_size
        # source_name = args.source_name
        # target_name = args.target_name
        # test_set_eval = {'features': test_set['features'], 'labels': test_set['labels'],
        #                  'accessions': test_set['accessions']}  # 目标域数据（测试）集--评估
        #
        # kwargs = {'num_workers': 0, 'pin_memory': True}  # 字典
        #
        # source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        #                                             **kwargs)  # 封装源域数据集
        #
        # target_data = torch.utils.data.TensorDataset(
        #     torch.FloatTensor(test_set['features']),
        #     torch.LongTensor(matrix_one_hot(test_set['labels'], int(max(test_set['labels']) + 1)).long()))
        # target_loader = torch.utils.data.DataLoader(target_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        # target_test_loader = torch.utils.data.DataLoader(target_data, batch_size=batch_size, shuffle=False, drop_last=False,
        #                                                  **kwargs)  # 封装目标域数据集
        # # class_num = max(train_set['labels']) + 1  # 训练集细胞类型数量
        # # class_num_test = max(test_set['labels']) + 1  # 测试集
        # print(source_loader)
        # print(iter(source_loader))
        # print(iter(source_loader).__next__()[0].shape,iter(source_loader).__next__()[1].shape)
        #
        # print(target_loader)
        # print(iter(target_loader))
        # print(iter(target_loader).__next__()[0].shape,iter(target_loader).__next__()[1].shape)


    # print(source1_loader)
    # print(iter(source1_loader))
    # print(iter(source1_loader).__next__()[0].shape,iter(source1_loader).__next__()[1].shape)
    # print(target_loader)
    # print(iter(target_loader))
    # print(iter(target_loader).__next__()[0].shape,iter(target_loader).__next__()[1].shape)

        source_set,target_set,source1_loader, source2_loader, source3_loader, target_train_loader, \
                  target_test_loader,input_sizes_list,class_num,class_num_target=preprocess(args)
        layer_size1 = [256, 512, 128]
        layer_size2 = [256, 512, 128]
        layer_size3 = [256, 512, 128]
        layer_size4 = [256, 512, 128]
        layer_sizes_list = [layer_size1 , layer_size2 , layer_size3 , layer_size4]
        print(layer_sizes_list)
        # model = scDGCCA(layer_sizes_list, input_sizes_list, 13)
        # train(model.cuda(), source1_loader, source2_loader, source3_loader, target_train_loader, target_test_loader)
        TAF(args,source_set,target_set,source1_loader,source2_loader,source3_loader,target_train_loader,class_num,class_num_target,fw)

