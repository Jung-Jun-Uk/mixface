import sys
import random
import os
import argparse
from pathlib import Path

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
from dataloader.utils import kface_accessories_converter, kface_expressions_converter, kface_luces_converter, kface_pose_converter

# We have already created a pair of kface in 'data/kface-test-pair.txt'
# Do not run this code. Just use 'data/kface-test-pair.txt'. That is our benchmark testset.


def create_kface_condition_pair(kface_idx_and_pose_path,
                                save_kface_attribtued_path,
                                accessories, 
                                luces, 
                                expressions):
    

    with open(kface_idx_and_pose_path, 'r') as f:
        test_lst = f.read().split('\n')[:-1]
    
    
    with open(save_kface_attribtued_path, 'w') as f:    
        count = 0
        for info in test_lst:                
            idpose1, idpose2, label = info.split(' ')
            id1, pose1 = idpose1.split('/')
            id2, pose2 = idpose2.split('/')
            
            a1, a2 = accessories[0], accessories[0]
            l1, l2 = luces[0], luces[0]
            e1, e2 = expressions[0], expressions[0]
            
            id1 = os.path.join(id1, a1, l1, e1, pose1)
            id2 = os.path.join(id2, a2, l2, e2, pose2)
            data = id1 + ' ' + id2 + ' ' + str(label) + '\n'                
            if count % 10 == 0:
                print('{} pair'.format(count + 1), data[:-1])
            f.write(data)
            count += 1        
        

def create_kface_idx_and_pose_pair(kface_idx_path, 
                                   save_kface_idx_and_pose_path,
                                   num_pos_img=25000, 
                                   num_neg_img=25000):
        
    poses = kface_pose_converter('c1~20')
    print('Pose        : ',poses)
    
    with open(kface_idx_path) as f:
        test_lst = f.read().split('\n')[:-1]    

    f = open(save_kface_idx_and_pose_path, 'w')
    pos_count = 0
    pos_label = 1
    pos_data_set = set()
    while(pos_count < num_pos_img):
        for idx in test_lst:
            
            while True:
                p1, p2 = random.sample(poses, 2)

                id1 = os.path.join(idx, p1) + '.jpg'
                id2 = os.path.join(idx, p2) + '.jpg'
                data = id1 + ' ' + id2 + ' ' + str(pos_label) + '\n'
                rdata = id2 + ' ' + id1 + ' ' + str(pos_label) + '\n' # reverse 
                if data not in pos_data_set and rdata not in pos_data_set:
                    pos_data_set.add(data)
                    pos_data_set.add(rdata)
                    break
            print('{} positive pair'.format(pos_count + 1), data[:-1])
            f.write(data)
            pos_count += 1
            if pos_count >= num_pos_img:
                break
    
    neg_count = 0
    neg_label = 0
    neg_data_set = set()
    while(neg_count < num_neg_img):
        idx1, idx2 = random.sample(test_lst, 2)
        while True:            
            p1, p2 = random.sample(poses, 2)

            id1 = os.path.join(idx1, p1) + '.jpg'
            id2 = os.path.join(idx2, p2) + '.jpg'
            data = id1 + ' ' + id2 + ' ' + str(neg_label) + '\n'
            rdata = id2 + ' ' + id1 + ' ' + str(neg_label) + '\n' # reverse 
            if data not in neg_data_set:
                neg_data_set.add(data)
                neg_data_set.add(rdata)
                break
        print('{} negative pair'.format(pos_count + neg_count + 1), data[:-1])
        f.write(data)
        neg_count += 1
    f.close()
            

def parser():    
    parser = argparse.ArgumentParser(description='Create kface test pair')    
    parser.add_argument('-a', '--accessories', type=str , default='s1', help='Accessory attribtutes')
    parser.add_argument('-l', '--luces', type=str , default='l1', help='Lux attribtutes')
    parser.add_argument('-e', '--expressions', type=str , default='e1', help='Expression attribtutes')

    parser.add_argument('--project_name', type=str, default='')
    parser.add_argument('--kface_idx_path', default='data/kface-test-identity30.txt')
    parser.add_argument('--save_kface_idx_and_pose_path', default='data/kface-testidx_and_pose-10K.txt')
    parser.add_argument('--save_kface_attribtued_name', default='pair10k.txt')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parser()

    wdir = Path('data/' + opt.project_name)
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    
    kface_idx_path = opt.kface_idx_path
    save_kface_idx_and_pose_path = opt.save_kface_idx_and_pose_path    
    save_kface_attribtued_path =  wdir / (opt.accessories + '-' + opt.luces + '-' + opt.expressions + '-' + opt.save_kface_attribtued_name)

    """ if not os.path.isfile(save_kface_idx_and_pose_path):
        create_kface_idx_and_pose_pair(kface_idx_path=kface_idx_path, 
                                    save_kface_idx_and_pose_path=save_kface_idx_and_pose_path, 
                                    num_pos_img=5000, 
                                    num_neg_img=5000)
    else:
        print('{} file already exist'.format(save_kface_idx_and_pose_path)) """

    accessories = kface_accessories_converter(opt.accessories)
    luces = kface_luces_converter(opt.luces) # except l7
    expressions = kface_expressions_converter(opt.expressions)

    if not os.path.isfile(save_kface_attribtued_path):    
        create_kface_condition_pair(kface_idx_and_pose_path=save_kface_idx_and_pose_path, 
                                    save_kface_attribtued_path=save_kface_attribtued_path,
                                    accessories=accessories,
                                    luces=luces,
                                    expressions=expressions)
    else:
        print('{} file already exist'.format(save_kface_attribtued_path))



