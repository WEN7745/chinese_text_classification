# coding: UTF-8
import argparse
import time
from importlib import import_module

import numpy as np
import torch

from train_eval import grid_search, init_network, train

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, FREE, FGSM, PGD')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--search', default=False, type=bool, help='True for using grid parameters to search params, False for using current best parameters ')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model   # TextCNN, FREE, FGSM, PGD
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    # 保证每次结果一致
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样   
    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    # 模型初始化
    model = x.Model(config).to(config.device)
    init_network(model)
    print(model.parameters)
    config.n_vocab = len(vocab)

    if config.model_name in ['FREE', 'PGD', "FGSM"]:
        if args.search:
            # 进行网格化搜索
            print("**********************************")
            print("starting grid parameter searching")
            print("**********************************")
            grid_search(model, config, args.word)
        else:
            # 直接使用当前预设好的超参进行训练
            print("**********************************")
            print("using hyper parameter training")
            print("**********************************")
            epsilon, alpha, attack_iters = config.epsilon, config.alpha, config.attack_iters
            train(config, model, train_iter, dev_iter, test_iter, epsilon, alpha, attack_iters)
            print("best_epsilon, best_alpha, best_attack_iters\t", epsilon, alpha, attack_iters)
    else:
        # 非对抗训练模型
        print("**********************************")
        print("training no adversarial learning model")
        print("**********************************")
        train(config, model, train_iter, dev_iter, test_iter)
