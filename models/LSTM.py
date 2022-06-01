import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class Config(object):
    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'LSTM'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.class_list = [
            x.strip()
            for x in open(dataset +
                          '/data/class.txt', encoding='utf-8').readlines()
        ]  # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'  # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)
        self.epsilon = 0.05  # 绕动上下限
        self.alpha = None  # 扰动更新幅度
        self.attack_iters = 2  # 每一个batch的扰动detla更新次数
        self.param_grid = {
            'epsilon': [0.01, 0.02, 0.05, 0.1, 0.2],
            'alpha': [None],
            'attack_iters': [3, 5, 8, 10]
        }  # 搜索网格参数
        self.search_num = 10  # 参数搜索次数
        self.lstm_units = 256  # 多少个单元
        self.num_layer = 2  # 循环层数
        self.bidirectional = True  # 是否启用双向
        self.hidden_dim = 1024  # 是否启用双向


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab,
                                          config.embed,
                                          padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed,
                            config.lstm_units,
                            num_layers=config.num_layer,
                            bidirectional=config.bidirectional,
                            batch_first=True)
        self.fc1 = nn.Linear(config.lstm_units * 2, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.device = config.device

    def forward(self, trains):
        text, text_lengths = trains
        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(embedded,
                                               text_lengths.float().to('cpu'),
                                               batch_first=True,
                                               enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed_embedded)
        # 获取正向和反向最后一个state
        cat = torch.cat((hidden[-1, :, :], hidden[-2, :, :]), dim=1)
        # cat = hidden.squeeze(0)
        rel = self.relu(cat)
        dense1 = self.fc1(rel)
        drop = self.dropout(dense1)
        preds = self.fc2(drop)
        return preds
