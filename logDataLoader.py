import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# 加载日志数据
ea = event_accumulator.EventAccumulator('./THUCNews/log/FGSM/05-15_18.15/events.out.tfevents.1652609723.DESKTOP-QMBGVLL')
# ea = event_accumulator.EventAccumulator('./THUCNews/log/FGSM/05-15_19.23/events.out.tfevents.1652613791.DESKTOP-QMBGVLL')
ea.Reload()
print(ea.scalars.Keys())

val_loss_train = ea.scalars.Items('loss/train')
val_loss_dev = ea.scalars.Items('loss/dev')
val_acc_train = ea.scalars.Items('acc/train')
val_acc_dev = ea.scalars.Items('acc/dev')
# 转为绘图数据
loss_train_step = [i.step for i in val_loss_train][60:]
loss_dev_step = [i.step for i in val_loss_dev][60:]
acc_train_step = [i.step for i in val_acc_train][60:]
acc_dev_step = [i.step for i in val_acc_dev][60:]

loss_train = [i.value for i in val_loss_train][60:]
loss_dev = [i.value for i in val_loss_dev][60:]
acc_train = [i.value for i in val_acc_train][60:]
acc_dev = [i.value for i in val_acc_dev][60:]
# 绘图
fig,subs = plt.subplots(2, 2,constrained_layout=True)
fig.suptitle('FGSM eps=0.1 alpha=0.02 noise_iters=None f1-score=91.95%')
subs[0][0].plot(loss_train_step, loss_train, linewidth=1, color='r',marker='o',
         markerfacecolor='blue',markersize=3)
subs[0][0].set_xlabel("step")
subs[0][0].set_ylabel("loss")
subs[0][0].set_title("loss_train")

subs[0][1].plot(loss_dev_step, loss_dev, linewidth=1, color='r',marker='o',
         markerfacecolor='blue',markersize=3)
subs[0][1].set_xlabel("step")
subs[0][1].set_ylabel("loss")
subs[0][1].set_title("loss_dev")

subs[1][0].plot(acc_train_step, acc_train, linewidth=1, color='r',marker='o',
         markerfacecolor='blue',markersize=3)
subs[1][0].set_xlabel("step")
subs[1][0].set_ylabel("acc")
subs[1][0].set_title("acc_train")

subs[1][1].plot(acc_dev_step, acc_dev, linewidth=1, color='r',marker='o',
         markerfacecolor='blue',markersize=3)
subs[1][1].set_xlabel("step")
subs[1][1].set_ylabel("acc")
subs[1][1].set_title("acc_dev")

plt.show()
