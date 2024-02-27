import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random
import math
import  scanpy as sc
from sklearn.metrics import roc_auc_score, f1_score,accuracy_score,recall_score,precision_score

# seed_everything()
import torch
import numpy as np
seed=666
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import random
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

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
seed_everything(666)

import lr_schedule
import loss_utility
from utils import *
from models import *
from evaluate import evaluate_summary, evaluate_multibatch
from vat import VATLoss
from loss_utility import *

# torch.set_num_threads(2)#神经网络中多线程数设置

def MGC(args,train_set,test_set,source1_loader,source2_loader,source3_loader,target_train_loader,class_num,class_num_test,f):
    seed_everything(666)
    batch_size = args.batch_size
    source_name=args.source_name
    target_name=args.target_name
    kwargs = {'num_workers': 0, 'pin_memory': True}  # 字典
    test_set_eval = {'features': test_set['features'], 'labels': test_set['labels'],
                     'accessions': test_set['accessions']}  # 目标域数据（测试）集--评估


    ### re-weighting the classifier   #调整分类器权重
    cls_num_list = [np.sum(train_set['labels'] == i) for i in range(class_num)]#每一类的细胞类型具体数量--训练集(all sources)
    #from https://github.com/YyzHarry/imbalanced-semi-self/blob/master/train.py
    # # Normalized weights based on inverse number of effective data per class.
    #2019 Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss
    #2020 Rethinking the Value of Labels for Improving Class-Imbalanced Learning
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cls_num_list)#求n次方
    per_cls_weights = (1.0 - beta) / np.array(effective_num)#每一类的权重
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()


    ## set base network(特征提取器+标签分类器)
    embedding_size = args.embedding_size  #嵌入层维度
    base_network= FeatureExtractor(num_inputs=train_set['features'].shape[1], embed_size = embedding_size).cuda()#特征提取器


    da_net1 = DAsonNet1(embedding_size,args.son_embeding).cuda()
    da_net2 = DAsonNet2(embedding_size, args.son_embeding).cuda()
    da_net3 = DAsonNet3(embedding_size, args.son_embeding).cuda()

    label_predictor1 = LabelPredictor1(args.son_embeding, class_num).cuda()#把特征提取结果放入标签预测器
    label_predictor2 = LabelPredictor2(args.son_embeding, class_num).cuda()  # 把特征提取结果放入标签预测器
    label_predictor3 = LabelPredictor3(args.son_embeding, class_num).cuda()  # 把特征提取结果放入标签预测器

    total_model1 = nn.Sequential(base_network,da_net1, label_predictor1)#NN网络
    total_model2 = nn.Sequential(base_network,da_net2, label_predictor2)  # NN网络
    total_model3 = nn.Sequential(base_network,da_net3,label_predictor3)  # NN网络

    gcca_loss = GCCA_loss
    # optimizer_centloss = torch.optim.SGD([{'params': gcca_loss.parameters()}], lr=0.5)############中心损失优化器

    # print("output size of FeatureExtractor and LabelPredictor: ", base_network.output_num(), class_num)
    # ad_net = scAdversarialNetwork(base_network.output_num(), 1024).cuda()#域鉴别器网络#####---实例？#1024   64



    ## set optimizer
    config_optimizer = {"lr_type": "inv", "lr_param": {"lr": 0.001, "gamma": 0.001, "power": 0.75}}
    parameter_list = base_network.get_parameters()+ da_net1.get_parameters()+ \
                     da_net2.get_parameters()+ da_net3.get_parameters()\
                     + label_predictor1.get_parameters()+label_predictor2.get_parameters()+label_predictor3.get_parameters()#多个网络参数列表
    optimizer = optim.SGD(parameter_list, lr=1e-3, weight_decay=5e-4, momentum=0.9, nesterov=True)#weight_decay=5e-4  1e-5
    schedule_param = config_optimizer["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[config_optimizer["lr_type"]]

    ## train
    # len_train_source = len(source_loader)#源batch数量
    # len_train_target = len(target_loader)#目标batch数量
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    epoch_global = 0.0

    # flag = False
    # s_global_centroid = torch.zeros(class_num, embedding_size).cuda()#质心(每类细胞的平均嵌入)
    # t_global_centroid = torch.zeros(class_num, embedding_size).cuda()
    best_acc=0
    best_precision=0
    best_epoch1=0
    best_epoch2 = 0
    num=args.num_iterations//250

    for epoch in range(args.num_iterations):#####50010
        if epoch % (20) == 0 and epoch != 0:#取余数 #2500  500  250
            feature_target = base_network(torch.FloatTensor(test_set['features']).cuda())#target——提取特征
            feature_targets1 = da_net1.forward(feature_target)
            feature_targets2 = da_net2.forward(feature_target)
            feature_targets3 = da_net3.forward(feature_target)

            output_target1 = label_predictor1.forward(feature_targets1)#target-标签预测
            output_target2 = label_predictor2.forward(feature_targets2)  # target-标签预测
            output_target3 = label_predictor3.forward(feature_targets3)  # target-标签预测

            softmax_out1 = nn.Softmax(dim=1)(output_target1)
            softmax_out2 = nn.Softmax(dim=1)(output_target2)
            softmax_out3 = nn.Softmax(dim=1)(output_target3)

            # softmax_out = (softmax_out1 + softmax_out2 + softmax_out3) / 3
            w1 = 1004
            w2 = 2285
            w3 = 638
            w4=2394
            softmax_out = (w1 / (w1 + w2 + w3)) * softmax_out1 + (w2 / (w1 + w2 + w3)) * softmax_out2 + (
                        w3 / (w1 + w2 + w3)) * softmax_out3  # w1,w2,w3是否为samples_nums

            predict_prob_arr, predict_label_arr = torch.max(softmax_out, 1)#https://blog.csdn.net/u013066730/article/details/112175384
            # 函数会返回两个tensor，第一个tensor是每行的最大值，softmax的输出中最大的是1，所以第一个tensor是全1的tensor；
            # 第二个tensor是每行最大值的索引。函数会返回两个tensor，第一个tensor是每行的最大值，softmax的输出中最大的是1，
            # 所以第一个tensor是全1的tensor；第二个tensor是每行最大值的索引。or是每行最大值的索引。

            result_path = args.result_path #"../results/"
            # model_file = result_path + 'final_model_' + str(epoch) + source_name + target_name+'.ckpt'
            # torch.save({'base_network': base_network.state_dict(), 'label_predictor': label_predictor.state_dict()}, model_file)

            if not os.path.exists(result_path):
                os.makedirs(result_path)
            with torch.no_grad():#表明当前计算不需要反向传播，使用之后，强制后边的内容不进行计算图的构建
                code_arr_s1 = base_network(Variable(torch.FloatTensor(train_set['features'][:1004]).cuda()))#源域-提特征
                code_arr_s2 = base_network(Variable(torch.FloatTensor(train_set['features'][1004:3289]).cuda()))
                code_arr_s3 = base_network(Variable(torch.FloatTensor(train_set['features'][3289:]).cuda()))
                code_arr_t = base_network(Variable(torch.FloatTensor(test_set_eval['features']).cuda()))#目标域-提特征
                code_arr = np.concatenate((code_arr_s1.cpu().data.numpy(),code_arr_s2.cpu().data.numpy(),\
                                           code_arr_s3.cpu().data.numpy(), code_arr_t.cpu().data.numpy()), 0)#将源特征和目标特征连接

                code_arr_t1 = da_net1.forward(code_arr_t)
                code_arr_t2 = da_net2.forward(code_arr_t)
                code_arr_t3 = da_net3.forward(code_arr_t)

                pre_target_eval_target1 = label_predictor1.forward(code_arr_t1)  # target-标签预测
                pre_target_eval_target2 = label_predictor2.forward(code_arr_t2)  # target-标签预测
                pre_target_eval_target3 = label_predictor3.forward(code_arr_t3)  # target-标签预测
                pre_softmax_out1 = nn.Softmax(dim=1)(pre_target_eval_target1)
                pre_softmax_out2 = nn.Softmax(dim=1)(pre_target_eval_target2)
                pre_softmax_out3 = nn.Softmax(dim=1)(pre_target_eval_target3)
                # pre_softmax_out = (pre_softmax_out1 + pre_softmax_out2 + pre_softmax_out3) / 3

                w1 = 1004
                w2 = 2285
                w3 = 638
                w4 = 2394
                pre_softmax_out = (w1 / (w1 + w2 + w3)) * pre_softmax_out1 + (w2 / (w1 + w2 + w3)) * pre_softmax_out2 + (
                        w3 / (w1 + w2 + w3)) * pre_softmax_out3  # w1,w2,w3是否为samples_nums

                predict_prob_arr_eval, predict_label_arr_eval = torch.max(pre_softmax_out, 1)


            digit_label_dict = pd.read_csv(args.dataset_path + 'pancreas_dict_labels.csv',index_col=0)#读取标签字典 !!!!!
            digit_label_dict = pd.DataFrame(zip(digit_label_dict.iloc[:,0], digit_label_dict.index), columns=['digit','label'])
            digit_label_dict = digit_label_dict.to_dict()['label']#构造为字典{0: 'alpha',……}
            # transform digit label to cell type name
            y_pred_label = [digit_label_dict[x] if x in digit_label_dict else x for x in predict_label_arr.cpu().data.numpy()]

            pred_labels_file = result_path + 'pred_labels_' + "_" + str(target_name) + "_" + str(epoch) + ".csv"
            pd.DataFrame([predict_prob_arr.cpu().data.numpy(), y_pred_label],  index=["pred_probability", "pred_label"]).to_csv(pred_labels_file, sep=',')
            embedding_file = result_path + 'embeddings_' + "_" + str(target_name) + "_" + str(epoch)+ ".csv"
            pd.DataFrame(code_arr).to_csv(embedding_file, sep=',')


            ### only for evaluation
            acc_by_label = np.zeros( class_num_test)
            all_label = test_set['labels']
            for i in range(class_num_test):
                acc_by_label[i] = np.sum(predict_label_arr.cpu().data.numpy()[all_label == i] == i) / np.sum(all_label == i)
            np.set_printoptions(suppress=True)#用于控制Python中小数的显示精度
            auc = roc_auc_score(all_label, softmax_out.cpu().detach().numpy(), multi_class='ovr')
            recall = recall_score(all_label, predict_label_arr.cpu().data.numpy(), average='macro')
            precisions = precision_score(all_label, predict_label_arr.cpu().data.numpy(), average='macro')
            f1 = f1_score(all_label, predict_label_arr.cpu().data.numpy(), average='macro')  ######
            overall_acc=accuracy_score(all_label, predict_label_arr.cpu().data.numpy())
            print('iter:', epoch, "average acc over all test cell types: ", round(np.nanmean(acc_by_label), 4))
            print("overall acc: ", np.round(overall_acc, 4))
            print("acc of each test cell type: ", np.round(acc_by_label, 4))
            print("average auc test cell types: ", np.round(auc, 4))
            print("average precision test cell types: ", np.round(precisions, 4))
            print("average recall  test cell types: ", np.round(recall, 4))
            print("average f1  test cell types: ", np.round(f1, 4))
            f.write("Epoch %d, test accuracy = %.4f, precision = %.4f, auc = %.4f,f1_macro=%.4f\n" % (
                epoch, overall_acc, np.nanmean(acc_by_label), auc,f1))
            if best_acc<overall_acc:
                best_acc=overall_acc
                best_epoch1=epoch
            if best_precision<np.nanmean(acc_by_label):
                best_precision=np.nanmean(acc_by_label)
                best_epoch2=epoch
            if epoch==num*250:
                f.write("Epoch %d, best accuracy = %.4f\n" % (best_epoch1, best_acc))
                f.write("Epoch %d, best precision = %.4f\n" % (best_epoch2, best_precision ))

            # 可视化
            code_arr_s=np.concatenate((code_arr_s1.cpu().data.numpy(),code_arr_s2.cpu().data.numpy(),\
                                           code_arr_s3.cpu().data.numpy()),0)
            adata1=sc.AnnData(code_arr_s)#.cpu().data.numpy()
            digit_label_dict = pd.read_csv('./processed_data/pancreas_dict_labels.csv', index_col=0)  # 读取标签字典 !!!!!
            print(np.where(digit_label_dict['digit_label'].values == 3)[0][0])  # 返回对应label的索引值
            listk = []
            for i in range(len(train_set['labels'])):
                t = np.where(digit_label_dict['digit_label'].values == train_set['labels'][i])[0][0]
                k = digit_label_dict.index[t]
                listk.append(k)
            # print(listk)
            # print(type(np.array(listk)))
            adata1.obs['celltype']=np.array(listk)
            adata1.obs['batch'] = train_set['accessions'].astype(str)
            adata1.obs['BATCH'] = train_set['accessions'].astype(str)

            adata2=sc.AnnData(code_arr_t.cpu().data.numpy())
            listk2 = []
            for i in range(len(predict_label_arr_eval.cpu().data.numpy())):
                t2 = np.where(digit_label_dict['digit_label'].values == predict_label_arr_eval.cpu().data.numpy()[i])[0][0]
                k2 = digit_label_dict.index[t2]
                listk2.append(k2)
            adata2.obs['celltype']=np.array(listk2)
            adata2.obs['batch'] = test_set['accessions'].astype(str)
            adata2.obs['BATCH'] = test_set['accessions'].astype(str)
            adata = sc.AnnData.concatenate(adata1, adata2, batch_key='BATCH')

            # adata=sc.AnnData.concatenate(adata1,adata2)
            # sc.tl.tsne(adata)
            # sc.pl.tsne(adata,color='batch')
            # sc.pl.tsne(adata, color='celltype')

            sc.tl.pca(adata)#
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)

            # adata.obs['batch'] = ['10X 3\'' if i == '0' else '10X 5\'' for i in adata.obs['batch']]###pbmc_
            fig1=sc.pl.umap(adata, color='batch', return_fig=True, title="", legend_loc="", frameon=False)  # ,save='_figure3_deepMNN_batch.pdf'
            fig1.savefig('pancreas3/'+str(epoch)+'ours_batch.png', dpi=1000, format='png', bbox_inches='tight')
            fig2=sc.pl.umap(adata, color='celltype', return_fig=True,title="", legend_loc="", frameon=False)  # ,save='_figure3_deepMNN_celltype.pdf'
            fig2.savefig('pancreas3/'+str(epoch)+'ours _celltype.png', dpi=1000, format='png', bbox_inches='tight')

            # sc.pl.umap(adata, color='batch')
            # sc.pl.umap(adata, color='celltype')
            # 保存target预测结果
            # pd.DataFrame(adata2.obs['celltype'].values).to_csv("pancreas4/celltype"+str(epoch)+"result.csv")


            #batch-effect evaluate
            div_score, div_score_all, ent_score, sil_score = evaluate_multibatch(code_arr, train_set, test_set_eval, epoch)
            # results_file = result_path + source_name + "_" + target_name + "_" + str(epoch)+ "_acc_div_sil.csv"
            # evel_res = [overall_acc,np.nanmean(acc_by_label), div_score, div_score_all, ent_score, sil_score]
            # pd.DataFrame(evel_res, index = ["overall_acc","precision","div_score","div_score_all","ent_score","sil_score"], columns=["values"]).to_csv(results_file, sep=',')
            # pred_labels_file = result_path + source_name + "_" + target_name + "_" + str(epoch) + "_pred_labels.csv"
            # pd.DataFrame([predict_label_arr.cpu().data.numpy(), all_label],  index=["pred_label", "true_label"]).to_csv(pred_labels_file, sep=',')



        ## train one iter
        base_network.train(True)
        da_net1.train(True)
        da_net2.train(True)
        da_net3.train(True)
        label_predictor1.train(True)
        label_predictor2.train(True)
        label_predictor3.train(True)


        optimizer = lr_scheduler(optimizer, epoch, **schedule_param)
        optimizer.zero_grad()
        # optimizer_centloss.zero_grad()


        iter_source1 = iter(source1_loader)
        iter_source2 = iter(source2_loader)
        iter_source3 = iter(source3_loader)
        iter_target = iter(target_train_loader) #iter()函数用来生成迭代器。

        inputs_source1, labels_source1= iter_source1.__next__()
        inputs_source2, labels_source2 = iter_source2.__next__()
        inputs_source3, labels_source3 = iter_source3.__next__()
        inputs_target, labels_target = iter_target.__next__()

        inputs_source1, labels_source1= inputs_source1.cuda(),  labels_source1.cuda()
        inputs_source2, labels_source2 = inputs_source2.cuda(), labels_source2.cuda()
        inputs_source3, labels_source3 = inputs_source3.cuda(), labels_source3.cuda()
        inputs_target, labels_target = inputs_target.cuda(), labels_target.cuda()

        feature_source1 = base_network(inputs_source1)###############
        feature_source2 = base_network(inputs_source2)
        feature_source3 = base_network(inputs_source3)
        feature_target = base_network(inputs_target)

        # features = torch.cat((feature_source, feature_target), dim=0)

        feature_sources1=da_net1.forward(feature_source1)
        feature_targets1 = da_net1.forward(feature_target)
        feature_sources2=da_net2.forward(feature_source2)
        feature_targets2 = da_net1.forward(feature_target)
        feature_sources3=da_net3.forward(feature_source3)
        feature_targets3 = da_net1.forward(feature_target)

        output_source1 = label_predictor1.forward(feature_sources1)############
        output_source2 = label_predictor2.forward(feature_sources2)
        output_source3 = label_predictor3.forward(feature_sources3)
        output_target1 = label_predictor1.forward(feature_targets1)#############
        output_target2 = label_predictor2.forward(feature_targets2)
        output_target3 = label_predictor3.forward(feature_targets3)

        softmax_out1 = nn.Softmax(dim=1)(output_target1)
        softmax_out2 = nn.Softmax(dim=1)(output_target2)
        softmax_out3 = nn.Softmax(dim=1)(output_target3)
        # softmax_out = (softmax_out1 + softmax_out2 + softmax_out3) / 3

        w1 = 1004
        w2 = 2285
        w3 = 638
        w4 = 2394
        softmax_out = (w1 / (w1 + w2 + w3)) * softmax_out1 + (w2 / (w1 + w2 + w3)) * softmax_out2 + (
                w3 / (w1 + w2 + w3)) * softmax_out3  # w1,w2,w3是否为samples_nums

        predict_prob_arr, predict_label_arr = torch.max(softmax_out, 1)

        ########domain alignment loss

        base = 1.0  # sigma for MMD
        sigma_list = [1, 2, 4, 8, 16]
        sigma_list = [sigma / base for sigma in sigma_list]
        transfer_loss = loss_utility.mix_rbf_mmd2(feature_sources1, feature_targets1, sigma_list)
        transfer_loss+= loss_utility.mix_rbf_mmd2(feature_sources2, feature_targets2, sigma_list)
        transfer_loss+= loss_utility.mix_rbf_mmd2(feature_sources3, feature_targets3, sigma_list)

        #gcca
        gcca_losses = gcca_loss([feature_source1, feature_source2, feature_source3, feature_target])


        # ######## VAT and BNM loss
        # LDS should be calculated before the forward for cross entropy
        vat_loss = VATLoss(xi=args.xi, eps=args.eps, ip=args.ip)
        lds_loss = vat_loss(total_model1, inputs_target)
        lds_loss += vat_loss(total_model2, inputs_target)
        lds_loss += vat_loss(total_model3, inputs_target)

        softmax_tgt1 = nn.Softmax(dim=1)(output_target1[:, 0:class_num])
        _, s_tgt1, _ = torch.svd(softmax_tgt1)
        BNM_loss = -torch.mean(s_tgt1)

        softmax_tgt2 = nn.Softmax(dim=1)(output_target2[:, 0:class_num])
        _, s_tgt2, _ = torch.svd(softmax_tgt2)
        BNM_loss += -torch.mean(s_tgt2)

        softmax_tgt3 = nn.Softmax(dim=1)(output_target3[:, 0:class_num])
        _, s_tgt3, _ = torch.svd(softmax_tgt3)
        BNM_loss += -torch.mean(s_tgt3)

        #loss cl1
        cl1_loss1 = torch.mean(torch.abs(softmax_out1 # 1-2  1-3
                                        - torch.nn.functional.softmax(softmax_out2, dim=1)))
        cl1_loss1 += torch.mean(torch.abs(torch.nn.functional.softmax(softmax_out1, dim=1)
                                         - torch.nn.functional.softmax(softmax_out3, dim=1)))
        cl1_loss1=cl1_loss1/2
        cl1_loss2 = torch.mean(torch.abs(torch.nn.functional.softmax(softmax_out2, dim=1)
                                         - torch.nn.functional.softmax(softmax_out1, dim=1)))
        cl1_loss2 += torch.mean(torch.abs(torch.nn.functional.softmax(softmax_out2, dim=1)
                                         - torch.nn.functional.softmax(softmax_out3, dim=1)))
        cl1_loss2=cl1_loss2/2
        cl1_loss3 = torch.mean(torch.abs(torch.nn.functional.softmax(softmax_out3, dim=1)
                                         - torch.nn.functional.softmax(softmax_out1, dim=1)))
        cl1_loss3 += torch.mean(torch.abs(torch.nn.functional.softmax(softmax_out3, dim=1)
                                         - torch.nn.functional.softmax(softmax_out2, dim=1)))
        cl1_loss3=cl1_loss3/2
        cl1_losses=cl1_loss1+cl1_loss2+cl1_loss3

        ######classifier CrossEntropyLoss
        classifier_loss = nn.CrossEntropyLoss(weight=per_cls_weights)(output_source1, torch.max(labels_source1, dim=1)[1])#weight=per_cls_weights
        classifier_loss += nn.CrossEntropyLoss(weight=per_cls_weights)(output_source2, torch.max(labels_source2, dim=1)[1])  # weight=per_cls_weights
        classifier_loss += nn.CrossEntropyLoss(weight=per_cls_weights)(output_source3, torch.max(labels_source3, dim=1)[1])  # weight=per_cls_weights
        # classifier_loss = loss_utility.CrossEntropyLoss(labels_source.float(), nn.Softmax(dim=1)(output_source))

        # #source domain contrastive loss
        # lm_sou = lc(feature_source, feature_source2, train_Z)




        epoch_th = args.epoch_th

        if epoch > epoch_th:
            lds_loss = torch.FloatTensor([0.0]).cuda()
        if epoch <= args.num_iterations:
            progress = epoch / args.epoch_th  # args.num_iterations
        else:
            progress = 1
        lambd = 2 / (1 + math.exp(-10 * progress)) - 1


        total_loss = classifier_loss+ lambd *args.Cl_coeff*cl1_losses + lambd *args.GA_coeff*gcca_losses \
                        +lambd *args.DA_coeff * transfer_loss \
                         + lambd * args.BNM_coeff * BNM_loss\
                         + lambd * args.alpha * lds_loss

                            #+lambd * args.DA_coeff * transfer_loss #0.01*cl1_losses

        total_loss.backward()
        optimizer.step()



