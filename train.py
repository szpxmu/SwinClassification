import os
import torch.nn.functional as F
import torch
import tqdm
from torch.utils.data import DataLoader
from Src.models import swin_transformer_base
from utils import dataloader, utils
from torch.utils import data
from torch import nn
from torch import optim


if __name__ == '__main__':
    # 网络实例化
    device = 'cuda'
    input_shape = [224, 224]
    save_acc_max = 0
    weights_path = './weights'
    class_names, num_classes = utils.get_classes('./model_data/cls_classes.txt')
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 1     ### default 32

    UnFreeze_Epoch = 200
    Unfreeze_batch_size = 1   ### default 32

    UnFreeze_flag = False

    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    model = swin_transformer_base(pretrained=True, num_classes=num_classes).to(device)

    train_annotation_path = "cls_train.txt"
    test_annotation_path = 'cls_test.txt'

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(test_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    train_dataset = dataloader.DataGenerator(train_lines, input_shape, True)
    val_dataset = dataloader.DataGenerator(val_lines, input_shape, False)

    batch_size = Freeze_batch_size
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4,
                              pin_memory=True,
                              drop_last=True, collate_fn=dataloader.detection_collate, sampler=None)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=4,
                            pin_memory=True,
                            drop_last=True, collate_fn=dataloader.detection_collate, sampler=None)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01
    nbs = 256
    lr_limit_max = 1e-3
    lr_limit_min = 1e-5
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    optimizer = optim.Adam(model.parameters(), Init_lr_fit, betas=(.9, 0.999),
                           weight_decay=0)
    lr_scheduler_func = utils.get_lr_scheduler("cos", Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
    # 开始训练
    model.freeze_backbone()
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        if epoch >= Freeze_Epoch and not UnFreeze_flag:
            batch_size = Unfreeze_batch_size
            nbs = 256
            lr_limit_max = 1e-3
            lr_limit_min = 1e-5
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            lr_scheduler_func = utils.get_lr_scheduler("cos", Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

            model.Unfreeze_backbone()

            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4,
                                      pin_memory=True,
                                      drop_last=True, collate_fn=dataloader.detection_collate, sampler=None)
            val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=4,
                                    pin_memory=True,
                                    drop_last=True, collate_fn=dataloader.detection_collate, sampler=None)
            UnFreeze_flag = True
        utils.set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        dt_size = len(train_loader.dataset)
        dt_size_val = len(val_loader.dataset)
        epoch_loss = 0
        total_accuracy = 0
        pbar = tqdm.tqdm(
            total=dt_size // batch_size,
            desc=f'Epoch {epoch + 1} / {UnFreeze_Epoch}',
            postfix=dict,
            miniters=.3
        )
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # 前向传播
            out = model(images)
            loss = criterion(out, labels)

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            accuracy = torch.mean((torch.argmax(F.softmax(out, dim=-1), dim=-1) == labels).type(torch.FloatTensor))
            total_accuracy += accuracy.item()

            pbar.set_postfix(**{
                'train_loss': epoch_loss / (i + 1),
                'train_accuracy': total_accuracy / (i + 1),
            })
            pbar.update(1)
        pbar.close()
        pbar = tqdm.tqdm(
            total=dt_size_val // batch_size,
            desc=f'Val_Epoch {epoch + 1} / {UnFreeze_Epoch}',
            postfix=dict,
            miniters=.3
        )
        epoch_loss_val = 0
        val_accuracy = 0
        model.eval()
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)
                accuracy = torch.mean(
                    (torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == labels).type(torch.FloatTensor))
                val_accuracy += accuracy.item()
            epoch_loss_val += loss.item()
            pbar.set_postfix(**{
                'val_loss': epoch_loss_val / (i + 1),
                'val_accuracy': val_accuracy / (i + 1),
            })
            pbar.update(1)
        pbar.close()
        if save_acc_max < val_accuracy / i:
            save_acc_max = val_accuracy / i
            torch.save(model.state_dict(), weights_path + '/weights.pth')
    print("训练完成！")
