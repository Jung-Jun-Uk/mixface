import os
import json
import pickle
"""
    ###################################################################

    K-Face : Korean Facial Image AI Training Dataset
    url    : http://www.aihub.or.kr/aidata/73

    Directory structure : High-ID-Accessories-Lux-Emotion
    ID example          : '19062421' ... '19101513' len 400
    Accessories example : 'S001', 'S002' .. 'S006'  len 6
    Lux example         : 'L1', 'L2' .. 'L30'       len 30
    Emotion example     : 'E01', 'E02', 'E03'       len 3
    S001 - L1, every emotion folder contaions a information txt file
    (ex. bbox, facial landmark) 
    
    ###################################################################
"""


def read_test_pair_dataset(test_pair_path):
        pair_data = list()
        with open(test_pair_path) as f:
            data = f.readlines()
        for d in data:
            info = d[:-1].split(' ')
            pair_data.append(info)
                                           
        return pair_data


def create_face_dataset(data_path):
    """
    -data
        -id1
            -.jpg
        -id2
            -.jpg                      
    """
    information = dict()
    identities = sorted(os.listdir(data_path))
    for idx, identity in enumerate(identities):
        if idx % 1000 == 0:
            print(idx, "preprocessing...")              
        identity_path = os.path.join(data_path, identity)
        try:
            images = os.listdir(identity_path)
        except:
            continue
        if len(images) == 0:
            continue
        img_path_list = list()
        for img in images:
            img_path = os.path.join(identity_path, img)
            img_path_list.append(img_path)
        information[idx] = img_path_list
    
    return information
    


""" def create_face_dataset(data_path):
    information = list()
    identity_lst = sorted(os.listdir(data_path))
    
    for i, idx in enumerate(identity_lst):
        if (i+1) % 1000 == 0:
            print(i+1, "preprocessing...")
        num_labels = i+1
        identity_path = os.path.join(data_path, idx)
        if os.path.isdir(identity_path):
            for img in os.listdir(identity_path):
                image_path = os.path.join(identity_path, img)
                label = i
                information.append({'img_path' : image_path, 'label' : label})
    return information, num_labels """


def create_kface_dataset(data_path, test_idx_path, 
                         accessories, luces, expressions, poses, mode='train'):
        
    assert isinstance(accessories, list)
    assert isinstance(luces, list)
    assert isinstance(expressions, list)
    assert isinstance(poses, list)

    with open(test_idx_path) as f:
        except_lst = f.read().split('\n')[:-1]    

    identity_lst = sorted(os.listdir(data_path))
    information = list()
    
    if mode == 'train':
        identity_lst = list(set(identity_lst) - set(except_lst))
    elif mode == 'test':
        identity_lst = except_lst
    else:
        assert ValueError

    for i, idx in enumerate(identity_lst):
        num_labels = i+1
        #print(num_labels, "preprocessing..")
        for a in accessories:
            for l in luces:
                for e in expressions:
                    for p in poses:
                        image_path = os.path.join(idx, a, l, e, p) + '.jpg'
                        label = i
                        information.append({'img_path' : image_path, 'label' : label})
    return information, num_labels


def read_kface_mtcnn_bbox_data(kface_mtcnn_bbox_path):
    with open(kface_mtcnn_bbox_path) as j:
        kface_mtcnn_bbox = json.load(j)
    
    return kface_mtcnn_bbox

def kface_accessories_converter(accessories):
    """
        ex) 
        parameters  : S1~3,5
        return      : [S001,S002,S003,S005]
    """

    accessories = accessories.lower()
    assert 's' == accessories[0]
    
    alst = []
    accessries = accessories[1:].split(',')
    for acs in accessries:
        acs = acs.split('~')
        if len(acs) == 1:
            acs = ['S' + acs[0].zfill(3)]
        else:
            acs = ['S' + str(a).zfill(3) for a in range(int(acs[0]), int(acs[1])+1)]
        alst.extend(acs)
    return alst

def kface_luces_converter(luces):
    """
        ex) 
        parameters  : L1~7,10~15,20~30
        return      : [L1, ... , L7, L10, ... L15, L20, ... , L30]
    """
    luces = luces.lower()
    assert 'l' == luces[0]

    llst = []
    luces = luces[1:].split(',')
    for lux in luces:
        lux = lux.split('~')
        if len(lux) == 1:
            lux = ['L' + lux[0]]
        else:
            lux = ['L' + str(l) for l in range(int(lux[0]), int(lux[1])+1)]
        llst.extend(lux)
    return llst

def kface_expressions_converter(expressions):
    """
        ex) 
        parameters  : E1~3
        return      : [E01, E02, E03]
    """
    expressions = expressions.lower()
    assert 'e' == expressions[0]    
    elst = []
    expressions = expressions[1:].split(',')
    for eps in expressions:
        eps = eps.split('~')
        if len(eps) == 1:
            eps = ['E' + eps[0].zfill(2)]
        else:
            eps = ['E' + str(e).zfill(2) for e in range(int(eps[0]), int(eps[1])+1)]
        elst.extend(eps)
    return elst


def kface_pose_converter(poses):
    """
        ex) 
        parameters  : C1,3,5,10~20
        return      : [C1,C3,C5,C10, ..., C20]
    """
    poses = poses.lower()
    assert 'c' == poses[0]    

    plst = []
    poses = poses[1:].split(',')
    for pose in poses:
        pose = pose.split('~')
        if len(pose) == 1:
            pose = ['C' + pose[0]]
        else:
            pose = ['C' + str(p) for p in range(int(pose[0]), int(pose[1])+1)]
        plst.extend(pose)
    return plst


if __name__ == "__main__":
    """ acs = 'S1,3,5'
    lux = 'L1~5,6~7,11~30'
    eps = 'E1~3'
    poses = 'C1,3,5,7~20'

    a = kface_accessories_converter(acs)
    l = kface_luces_converter(lux)
    e = kface_expressions_converter(eps)
    p = kface_pose_converter(poses)

    print("Aceessories :",a)
    print("Lux         :",l)
    print("Expression  :",e)
    print("Pose        :",p) """

    create_face_dataset(name='ms1m-retinaface-t1', data_path='/home/work/jju/data/MS1M-RetinaFace')