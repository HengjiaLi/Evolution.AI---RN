# More-CLEVR without images, but state descriptions of ground truth.
import cv2
import os
import numpy as np
import random
#import cPickle as pickle
import pickle
import warnings
import argparse

parser = argparse.ArgumentParser(description='Sort-of-CLEVR dataset generator')
parser.add_argument('--seed', type=int, default=2021, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--t-subtype', type=int, default=-1,
                    help='Force ternary questions to be of a given type')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
print('2020')
train_size = 9800
test_size = 200
val_size = 128

img_size = 75
size = 5 #radius of object
question_size = 19 ## (6 for one-hot vector of color), 2 for question type, 5 for question subtype
q_type_idx = 12
sub_q_type_idx = 14
"""Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""

nb_questions = 15
dirs = './data'

colors = [
    (0,0,255),##r
    (0,255,0),##g
    (255,0,0),##b
    (0,156,255),##o
    (128,128,128),##k
    (0,255,255)##y
]


try:
    os.makedirs(dirs)
except:
    print('directory {} already exists'.format(dirs))

def center_generate(objects):
    while True:
        pas = True
        center = np.random.randint(0+size, img_size - size, 2)
        #a =  np.random.randint(0+size, img_size - size, 2)
        #center = np.asarray([a[1],a[0]])#flip x-y coordinate
        if len(objects) > 0:
            for obj in objects:
                if ((center - obj[6:8]) ** 2).sum() < ((size * 2) ** 2):
                    pas = False
        if pas:
            return center



def build_dataset():#for each image
    objects = []
    img = np.ones((img_size,img_size,3)) * 255
    for color_id,color in enumerate(colors):  
        center = center_generate(objects)
        color = np.zeros(6)
        color[color_id] = 1
        if random.random()<0.5:
            start = (center[0]-size, center[1]-size)
            end = (center[0]+size, center[1]+size)
            cv2.rectangle(img, start, end, colors[color_id], -1)#artificially generates recs
            shape = np.asarray([1,0])#rectangle
            objects.append(np.hstack((color,center,shape)))
        else:
            center_ = (center[0], center[1])
            cv2.circle(img, center_, size, colors[color_id], -1)#artificially generates circles
            shape = np.asarray([0,1])#circle
            objects.append(np.hstack((color,center,shape)))


    #ternary_questions = []
    binary_questions = []
    norel_questions = []
    #ternary_answers = []
    binary_answers = []
    norel_answers = []
    """Non-relational questions"""
    for _ in range(nb_questions):
        question = np.zeros((question_size))
        color = random.randint(0,5)
        question[color] = 1
        question[q_type_idx] = 1
        subtype = random.randint(0,2)
        question[subtype+sub_q_type_idx] = 1
        norel_questions.append(question)
        """Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""
        if subtype == 0:#14
            """query shape->rectangle/circle"""
            if objects[color][-2] == 1:
                answer = 2#rec
            else:
                answer = 3#circle

        elif subtype == 1:#15
            """query vertical position->yes/no"""
            if objects[color][6] < img_size / 2:#check row ind
                answer = 0
            else:
                answer = 1

        elif subtype == 2:#16
            """query horizontal position->yes/no"""
            if objects[color][7] < img_size / 2:#check col ind
                answer = 0
            else:
                answer = 1
        norel_answers.append(answer)
    
    """Binary Relational questions"""
    for _ in range(nb_questions):
        question = np.zeros((question_size))
        color = random.randint(0,5)
        question[color] = 1
        question[q_type_idx+1] = 1
        subtype = random.randint(0,4)
        question[subtype+sub_q_type_idx] = 1
        
        if subtype == 0:#14
            """closest-to->rectangle/circle"""
            my_obj = objects[color][6:8]# coordinate of object's centre
            dist_list = [((my_obj - obj[6:8]) ** 2).sum() for obj in objects]
            dist_list[dist_list.index(0)] = 999
            closest = dist_list.index(min(dist_list))
            if objects[closest][-2] == 1:
                answer = 2
            else:
                answer = 3
                
        elif subtype == 1:#15
            """furthest-from->rectangle/circle"""
            my_obj = objects[color][6:8]# coordinate of object's centre
            dist_list = [((my_obj - obj[6:8]) ** 2).sum() for obj in objects]
            furthest = dist_list.index(max(dist_list))
            if objects[furthest][-2] == 1:
                answer = 2
            else:
                answer = 3

        elif subtype == 2:#16
            """count->1~6"""
            my_obj = objects[color][-2]#my object's shape
            count = -1
            for obj in objects:
                if obj[-2] == my_obj:
                    count +=1 
            answer = count+4
        elif subtype == 3:#17
            ''' if current object is above object 2--->yes or no'''
            
            color2 = random.randint(0,5)#select second color
            question[color2+6] = 1
            obj_1 = objects[color]#reference object
            obj_2 = objects[color2]
            if obj_1[6]>obj_2[6]:#compare row ind
                answer = 0
            else:
                answer = 1
      
        elif subtype == 4:#18
            ''' if color2 is on the left to the current object --->yes or no'''
            color2 = random.randint(0,5)#select second color
            question[color2+6] = 1
            obj_1 = objects[color]#reference object
            obj_2 = objects[color2]
            if obj_1[7]<obj_2[7]:#compare col ind
                answer = 0
            else:
                answer = 1
        
        binary_questions.append(question)
        binary_answers.append(answer)
    binary_relations = (binary_questions, binary_answers)
    norelations = (norel_questions, norel_answers)
    img = img/255.
    #img_states = [item for sublist in objects for item in sublist]#flatten objects
    img_states = [item for sublist in objects for item in np.delete(sublist,[6,7])]#delete center coordinate
    dataset = (img_states, binary_relations, norelations,img)
    return dataset

#print('x-y flipped')
print('building test datasets...')
test_datasets = [build_dataset() for _ in range(test_size)]
print('building train datasets...')
train_datasets = [build_dataset() for _ in range(train_size)]
print('building validation datasets...')
val_datasets = [build_dataset() for _ in range(val_size)]

#img_count = 0
#cv2.imwrite(os.path.join(dirs,'{}.png'.format(img_count)), cv2.resize(train_datasets[0][0]*255, (512,512)))

print('saving datasets...')
filename = os.path.join(dirs,'more-clevr_state.pickle')
with  open(filename, 'wb') as f:
    pickle.dump((train_datasets, test_datasets, val_datasets), f)
print('datasets saved at {}'.format(filename))
