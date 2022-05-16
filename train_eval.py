# coding: UTF-8
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from tensorboardX import SummaryWriter

from utils import (build_dataset, build_iterator, clamp, get_hyper_para,
                   get_time_dif)


# 权重初始化，默认xavier
def init_network(model, method='kaiming', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def grid_search(model, config, word):
    '''
    基于网格搜索的方式，选取最佳超参配置
    '''
    best_acc = 0.0
    best_report = ''
    best_confusion = ''
    param_set = set()
    for seed in range(config.search_num):
        # 保证每次结果一致
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        
        start_time = time.time()
        print("Loading data...")
        vocab, train_data, dev_data, test_data = build_dataset(config, word)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        
        # 重新模型初始化
        init_network(model)
        if seed == 0:
            # 初始值使用之前最好配置得出baseline
            epsilon, alpha, attack_iters = config.epsilon, config.alpha, config.attack_iters
            # 对最加参数设定初始值
            best_epsilon, best_alpha, best_attack_iters = epsilon, alpha, attack_iters
        else:
            epsilon, alpha, attack_iters, param_set = get_hyper_para(config.param_grid, seed, param_set)
        cur_best_acc, cur_report, cur_confusion = train(config, model, train_iter, dev_iter, test_iter, epsilon, alpha, attack_iters)
        if cur_best_acc > best_acc:
            best_acc = cur_best_acc
            best_report = cur_report
            best_confusion = cur_confusion
            best_epsilon, best_alpha, best_attack_iters = epsilon, alpha, attack_iters
        print(" best_epsilon, best_alpha, best_attack_iters\t", best_epsilon, best_alpha, best_attack_iters)
    # 在最后进行输出
    print(" best_epsilon, best_alpha, best_attack_iters\t", best_epsilon, best_alpha, best_attack_iters)
    print("Precision, Recall and F1-Score...")
    print(best_report)
    print("Confusion Matrix...")
    print(best_confusion)


def train(config, model, train_iter, dev_iter, test_iter, epsilon=None, alpha=None, attack_iters=None):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    # 针对对抗模型进行噪音初始化
    if config.model_name in ['FREE', 'PGD', "FGSM"]:
        delta = torch.zeros(config.batch_size, config.pad_size, config.embed).to(config.device)
        delta.requires_grad = True
        epsilon = torch.tensor(epsilon)

    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    if config.model_name == 'FREE':
        num_epochs = int(math.ceil(config.num_epochs/attack_iters)*attack_iters)
    else:
        num_epochs = config.num_epochs
    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for _, (trains, labels) in enumerate(train_iter):
            if config.model_name == "PGD":
                # 每一个batch均进行一个初始化
                delta = torch.zeros(config.batch_size, config.pad_size, config.embed).to(config.device)
                delta.requires_grad = True
                # 经过attack_iters的迭代次数，生成最终的delta，并进行梯度回传
                for _ in range(attack_iters):
                    outputs = model(trains, delta[:trains[0].size(0)])
                    model.zero_grad()
                    loss = F.cross_entropy(outputs, labels)
                    loss.backward()
                    grad = delta.grad.detach()

                    delta.data = delta + alpha * torch.sign(grad)
                    delta.data[:trains[0].size(0)] = clamp(delta[:trains[0].size(0)], -epsilon, epsilon)
                    delta.grad.zero_()
                # 扰动循环更新之后，进行梯度求导
                delta = delta.detach()
                # 重新由最终的delta计算并更新梯度
                outputs = model(trains, delta[:trains[0].size(0)])
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
            elif config.model_name == "FREE":
                # 仅在第一轮前进行初始化
                # 经过attack_iterss的迭代次数，每一轮均生成delta并进行梯度回传
                for _ in range(attack_iters):
                    outputs = model(trains, delta[:trains[0].size(0)])
                    model.zero_grad()
                    loss = F.cross_entropy(outputs, labels)
                    loss.backward()
                    # 更新delta，方便下一轮模型参数更新
                    grad = delta.grad.detach()
                    delta.data = delta + epsilon * torch.sign(grad)
                    delta.data[:trains[0].size(0)] = clamp(delta[:trains[0].size(0)], -epsilon, epsilon)
                    # delta梯度置零
                    delta.grad.zero_()
                    # 更新模型
                    optimizer.step()
            elif config.model_name == 'FGSM':
                delta.data.uniform_(-epsilon, epsilon)
                delta.data[:trains[0].size(0)] = clamp(delta[:trains[0].size(0)], -epsilon, epsilon)
                # 先过一边导数，来获取delta的导数
                outputs = model(trains, delta[:trains[0].size(0)])
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                # 求出delta的导数
                grad = delta.grad.detach()
                delta.data = delta + alpha * torch.sign(grad)
                delta.data[:trains[0].size(0)] = clamp(delta[:trains[0].size(0)], -epsilon, epsilon)
                delta.grad.zero_()
                # 更新梯度
                outputs = model(trains, delta[:trains[0].size(0)])
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
            else:
                outputs = model(trains)
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
        # 一个epoch后进行更新学习率
        print('current learning rate\t', scheduler.get_last_lr())
        scheduler.step()
    writer.close()
    return test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    return test_acc, test_report, test_confusion


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
