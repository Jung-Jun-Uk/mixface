import sys
import random
import os

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
from dataloader.utils import kface_accessories_converter, kface_expressions_converter, kface_luces_converter, kface_pose_converter

# We have already created a pair of kface in 'data/kface-test-pair.txt'
# Do not run this code. Just use 'data/kface-test-pair.txt'. That is our benchmark testset.

def att_sampler(attributes, num_sample):
    if len(attributes) > 1:
        return random.sample(attributes, num_sample)
    return attributes[0], attributes[0]
        

def create_kface_test_pair(test_idx_path, 
                           save_testpair_path,
                           num_pos_img=25000, 
                           num_neg_img=25000):
    
    accessories = kface_accessories_converter('s1')
    luces = kface_luces_converter('l1') # except l7
    expressions = kface_expressions_converter('e1')
    poses = kface_pose_converter('c4~10')

    """ accessories = kface_accessories_converter('s1~4')
    luces = kface_luces_converter('l1~3,8~9,12~13,16~17,19~20,22~23,25~26,28~29') # except l7
    expressions = kface_expressions_converter('e1~2')
    poses = kface_pose_converter('c1~13') """

    print('Aceessories : ',accessories)
    print('Lux         : ',luces)
    print('Expression  : ',expressions)
    print('Pose        : ',poses)
    
    with open(test_idx_path) as f:
        test_lst = f.read().split('\n')[:-1]    

    f = open(save_testpair_path, 'w')
    pos_count = 0
    pos_label = 1
    pos_data_set = set()
    while(pos_count < num_pos_img):
        for idx in test_lst:
            
            while True:
                a1, a2 = att_sampler(accessories, 2)
                l1, l2 = att_sampler(luces, 2)
                e1, e2 = att_sampler(expressions, 2)
                p1, p2 = att_sampler(poses, 2)

                id1 = os.path.join(idx, a1, l1, e1, p1) + '.jpg'
                id2 = os.path.join(idx, a2, l2, e2, p2) + '.jpg'
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
            a1, a2 = att_sampler(accessories, 2)
            l1, l2 = att_sampler(luces, 2)
            e1, e2 = att_sampler(expressions, 2)
            p1, p2 = att_sampler(poses, 2)

            id1 = os.path.join(idx1, a1, l1, e1, p1) + '.jpg'
            id2 = os.path.join(idx2, a2, l2, e2, p2) + '.jpg'
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
            

if __name__ == "__main__":
    test_idx_path = 'data/kface-test-identity30.txt'
    save_testpair_path = 'kface-test-pair-verysmall-10k.txt'
    create_kface_test_pair(test_idx_path, save_testpair_path, num_pos_img=500, num_neg_img=500)
    

