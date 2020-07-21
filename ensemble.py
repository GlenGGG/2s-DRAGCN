import argparse
import pickle

import numpy as np
from tqdm import tqdm

def cutLabel(label):
    idx = []
    print("label type: {}".format(type(label)))
    print(label[0,0],label[1,0])
    for (i, l) in enumerate(label[1]):
        # duo
        #print(type(l), l)
        l=int(l)
        if (l >= 49 and l<=59) or (l>=105 and l<=119):
            if l>=49 and l<=59:
                l = l-49
            else:
                l = l-105 + 11
            idx.append(i)
            label[1,i]=l

    label = label[:,idx]
    return label

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='ntu/xsub', choices={'kinetics', 'ntu/xsub', 'ntu/xview','ntu-120/xset','ntu-120/xsub'},
                    help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation')
parser.add_argument('--model', default='agcn', help='agcn or dragcn', choices={'agcn','dragcn'})
arg = parser.parse_args()

dataset = arg.datasets
label = open('./data/' + dataset + '/val_label.pkl', 'rb')
label=pickle.load(label)
label = np.array(label)
label=cutLabel(label)
model=arg.model
r1 = open('./work_dir/' + dataset + '/'+model+'_test_joint/epoch1_test_score.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open('./work_dir/' + dataset + '/'+model+'_test_bone/epoch1_test_score.pkl', 'rb')
r2 = list(pickle.load(r2).items())
right_num = total_num = right_num_5 = 0
print("len(r1): ",len(r1))
print("len(r2): ", len(r2))
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    r = r11 + r22 * arg.alpha
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
acc = right_num / total_num
acc5 = right_num_5 / total_num
print(acc, acc5)

