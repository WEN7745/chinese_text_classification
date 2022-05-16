# Chinese-Text-Classification-Pytorch
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

本项目参照 https://github.com/649453932/Chinese-Text-Classification-Pytorch 继续进行修改
针对 Fast is better than free: Revisiting adversarial training 进行在文本上的迁移
并于基线TextCNN进行对比


## 环境
python 3.7  
pytorch 1.1  
tqdm  
sklearn  
tensorboardX

## 复现
根据论文进行复现
    FGSM: 
        涉及参数：
            epsilon：为每次扰动delta的上限值，用于限制扰动的力度
            alpha：为每次delta更新时的幅度，根据对应梯度的正负，来加减更新
        扰动delta更新方式：
            对于每一个训练数据的batch，利用以下步骤进行更新
                step1 初始化delta：基于delta.data.uniform_(-epsilon, epsilon)，初始化扰动参数delta
                step2 计算delta梯度： 基于当前初始化的扰动delta的梯度grad，来更新delta.data = delta + alpha * torch.sign(grad)
                step3 稳定delta值: 对当前delta进行基于epsilon上下限的截断，以保证扰动参数delta的稳定性
                step4 更新模型参数: 基于更新后的delta值，进行梯度计算并回传来更新模型参数

    PGD：
        涉及参数：
            epsilon：为每次扰动delta的上限值，用于限制扰动的力度
            alpha：为每次delta更新时的幅度，根据对应梯度的正负，来加减更新
            attack_iters：针对每一批训练数据，delta参数更新次数
        扰动delta更新方式：
            对于每一个训练数据的batch，利用以下步骤进行更新
                step1 初始化delta：基于delta.data.uniform_(-epsilon, epsilon)，初始化扰动参数delta
                step2 更新delta梯度： 基于当前初始化的扰动delta的梯度grad，来更新delta.data = delta + alpha * torch.sign(grad)
                step3 稳定delta值: 对当前delta进行基于epsilon上下限的截断，以保证扰动参数delta的稳定性
                step4 扰动更新次数判断：若扰动delta的更新次数**大于设定值**，则跳出；否则重复step2和step3
                step5 更新模型参数: 基于更新后的delta值，进行梯度计算并回传来更新模型参数

    FREE：
        涉及参数：
            epsilon：为每次扰动delta的上限值，用于限制扰动的力度;同时也为delta的抖动幅度
            attack_iters：针对每一批训练数据，delta参数更新次数
        扰动delta更新方式：
            step1 模型初始化时，进行delta的初始化：基于delta.data.uniform_(-epsilon, epsilon)，初始化扰动参数delta
            对于每一个训练数据的batch，利用以下步骤进行更新
                step2 计算delta梯度： 基于当前初始化的扰动delta的梯度grad，来更新delta.data = delta + epsilon * torch.sign(grad)
                step3 稳定delta值: 对当前delta进行基于epsilon上下限的截断，以保证扰动参数delta的稳定性
                step4 更新模型参数: 基于**原始的delta值**，进行梯度计算并回传来更新模型参数
                step5 扰动更新次数判断：若扰动delta的更新次数**大于设定值**，则跳出；否则重复step2、step3和step4
    ## 调优
针对以下三种超参数，采用网格式调优
最终得出最佳参数epsilon、alpha、attack_iters
    {
        PGD:{
            epsilon:0.05, 
            alpha:0.01, 
            attack_iters:2
            },
        FREE:
            {
            epsilon:0.05, 
            alpha:None,
            attack_iters:2
            }
        FGSM : 
            {
            epsilon:0.05, 
            alpha:0.2,
            attack_iters:None
            },
    }

## 最终结果

结果如下：
    {
        TextCNN:
            {
            precision: 0.9101,
            recall: 0.9098,
            f1-score: 0.9098,
            },
        PGD:{
            precision: 0.9154,
            recall: 0.9154,
            f1-score: 0.9153,
            },
        FREE:
            {
            precision: 0.9196,
            recall: 0.9196,
            f1-score: 0.9194,
            },
        FGSM : 
            {
            precision: 0.9220,
            recall: 0.9220,
            f1-score: 0.9219,
            },
    }

## 使用说明
```
# 训练并测试：
# TextCNN
python run.py --model TextCNN

# FREE
python run.py --model FREE

# PGD
python run.py --model PGD

# FGSM
python run.py --model FGSM

# 如需启动网格搜索，默认不启动:
python run.py --model FGSM --search True
```

### 参数
模型都在models目录下，超参定义和模型定义在同一文件中。  
扰动参数为:epsilon、alpha、attack_iters

## 论文
[1] Fast is better than free: Revisiting adversarial training
