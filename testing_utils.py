import numpy as np
from tqdm import tqdm
from training_utils import judging_computation_device
from torch import zeros

def test(model, loader, num_class=2, vote_num=3,device=judging_computation_device()):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))

    predictions = []
    labels = []

    all_features=[]

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        points, target = points.to(device), target.to(device)

        points = points.transpose(2, 1)
        vote_pool = zeros(target.size()[0], num_class).to(device)

        if 1 != vote_num:
          for _ in range(vote_num):
              pred, _ = classifier(points)
              vote_pool += pred
        elif 1 == vote_num:
          pred, learned_features = classifier(points)
          vote_pool += pred
          features= learned_features # Assuming you want to extract trans_feat, modify as needed
          all_features.append(features.detach().cpu().numpy())
        pred = vote_pool / vote_num
        predictions.append(pred.cpu().numpy())
        labels.append(target.cpu().numpy())
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    if 1 == vote_num:
      all_features = np.concatenate(all_features, axis=0)
      np.save('./results/pc_nn_features.npy', all_features)#for visulise

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    if 1 != vote_num:
      return instance_acc, class_acc, predictions, labels
    elif 1 == vote_num:
      return instance_acc, class_acc, predictions, labels, all_features