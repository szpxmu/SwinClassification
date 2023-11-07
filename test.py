import os

import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
from sklearn.metrics import *
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import dataloader, utils
import torch
from Src.models import swin_transformer_base

if __name__ == '__main__':
    device = 'cuda'
    input_shape = [224, 224]
    weights_path = './weights'
    class_names, num_classes = utils.get_classes('./model_data/cls_classes.txt')
    model = swin_transformer_base(pretrained=False, num_classes=num_classes).to(device)

    key = model.load_state_dict(torch.load(weights_path + '/weights.pth', map_location=device))
    print(key)
    model.eval()
    test_annotation_path = 'cls_test.txt'

    with open(test_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    val_dataset = dataloader.DataGenerator(val_lines, input_shape, False)

    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=4,
                            pin_memory=True,
                            drop_last=True, collate_fn=dataloader.detection_collate, sampler=None)
    out_labels = []
    pre_labels = []
    for images, labels in tqdm.tqdm(val_loader):
        images = images.to(device)
        labels = labels.numpy()
        out_labels.append(labels)
        with torch.no_grad():
            outputs = model(images)
            outputs = F.softmax(outputs, dim=-1)
            outputs = torch.argmax(outputs, dim=-1).cpu().numpy()
        pre_labels.append(outputs)

    out_labels = np.array(out_labels)
    pre_labels = np.array(pre_labels)

    cm = confusion_matrix(out_labels, pre_labels)
    sns.heatmap(cm, annot=True)
    plt.show()
    print('acc is :', accuracy_score(out_labels, pre_labels))
    print('precision is :', precision_score(out_labels, pre_labels, average='weighted'))
    print('recall is :', recall_score(out_labels, pre_labels, average='weighted'))
    print('f1 is :', f1_score(out_labels, pre_labels, average='weighted'))
