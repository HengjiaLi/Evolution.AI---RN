"""Generate the pixel version of the extended Sort-of-CLEVR dataset"""

# import packages
import cv2
import os
import numpy as np
import random
import pickle
import warnings
import argparse

parser = argparse.ArgumentParser(description='Sort-of-CLEVR dataset generator')
parser.add_argument('--seed', type=int, default=2020, metavar='S',
                    help='random seed (default: 1)')#random seed
parser.add_argument('--t-subtype', type=int, default=-1,
                    help='Force ternary questions to be of a given type')
args = parser.parse_args()

# define random seed
random.seed(args.seed)
np.random.seed(args.seed)

# define dataset size
train_size = 9800
test_size = 200
val_size = 128
# define image-question size
img_size = 75
size = 5 #radius of object
question_size = 19 ## (6 for one-hot vector of color), 2 for question type, 5 for question subtype
q_type_idx = 12
sub_q_type_idx = 14
"""Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""

nb_questions = 15# Each image contains 15 relational and 15 non-relational Qs
dirs = './data'# Directory of storing the dataset

# colors of the objects
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

def center_generate(objects):# generate random object centers
    while True:
        pas = True
        center = np.random.randint(0+size, img_size - size, 2)
        if len(objects) > 0:
            for name,c,shape in objects:
                if ((center - c) ** 2).sum() < ((size * 2) ** 2):# if objects overlap
                    pas = False
        if pas:
            return center



def build_dataset():# Generate one image
    '''
    Generate Objects
    '''
    objects = []# store all six objects
    img = np.ones((img_size,img_size,3)) * 255
    for color_id,color in enumerate(colors):  
        center = center_generate(objects)# generate a center
        if random.random()<0.5:# determine object's shape
            start = (center[0]-size, center[1]-size)
            end = (center[0]+size, center[1]+size)
            cv2.rectangle(img, start, end, color, -1)# rectangle
            objects.append((color_id,center,'r'))
        else:
            center_ = (center[0], center[1])
            cv2.circle(img, center_, size, color, -1)#Circle
            objects.append((color_id,center,'c'))
    '''
    Generate Questions
    '''
    binary_questions = []
    norel_questions = []
    binary_answers = []
    norel_answers = []
    """Non-relational questions"""
    for _ in range(nb_questions):
        question = np.zeros((question_size))
        color = random.randint(0,5)# Selcet object's color
        question[color] = 1
        question[q_type_idx] = 1
        subtype = random.randint(0,2)# generate reandom question
        question[subtype+sub_q_type_idx] = 1
        norel_questions.append(question)
        """Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""
        if subtype == 0:#14
            """query shape->rectangle/circle"""
            if objects[color][2] == 'r':
                answer = 2
            else:
                answer = 3

        elif subtype == 1:#15
            """query vertical position->yes/no"""
            if objects[color][1][0] < img_size / 2:#check row ind
                answer = 0
            else:
                answer = 1

        elif subtype == 2:#16
            """query horizontal position->yes/no"""
            if objects[color][1][1] < img_size / 2:#check col ind
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
        subtype = random.randint(0,4)# Generate random question
        question[subtype+sub_q_type_idx] = 1
        
        if subtype == 0:#14
            """closest-to->rectangle/circle"""
            my_obj = objects[color][1]# coordinate of object's centre
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]# compute all distances
            dist_list[dist_list.index(0)] = 999 # set the distance to self as 999
            closest = dist_list.index(min(dist_list))# find the closest object
            if objects[closest][2] == 'r':
                answer = 2
            else:
                answer = 3
                
        elif subtype == 1:#15
            """furthest-from->rectangle/circle"""
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]# compute all distances
            furthest = dist_list.index(max(dist_list))# find the farthest object
            if objects[furthest][2] == 'r':
                answer = 2
            else:
                answer = 3

        elif subtype == 2:#16
            """count->1~6"""
            my_obj = objects[color][2]# color of the selected object
            count = -1
            for obj in objects:
                if obj[2] == my_obj:
                    count +=1 
            answer = count+4 # possible answers: 0~5

        elif subtype == 3:#17
            ''' if current object is above object 2--->yes or no'''
            
            color2 = random.randint(0,5)#select second object'scolor
            question[color2+6] = 1
            obj_1 = objects[color][1]#reference object
            obj_2 = objects[color2][1]
            if obj_1[0]>obj_2[0]:#compare row ind
                answer = 0
            else:
                answer = 1
      
        elif subtype == 4:#18
            ''' if color2 is on the left to the current object --->yes or no'''
            color2 = random.randint(0,5)#select second object's color
            question[color2+6] = 1
            obj_1 = objects[color][1]#reference object
            obj_2 = objects[color2][1]
            if obj_1[1]<obj_2[1]:#compare col ind
                answer = 0
            else:
                answer = 1

        binary_questions.append(question)
        binary_answers.append(answer)
    binary_relations = (binary_questions, binary_answers)
    norelations = (norel_questions, norel_answers)
    
    img = img/255.
    dataset = (img, binary_relations, norelations)
    return dataset

# Create train/test/val datasets
print('building test datasets...')
test_datasets = [build_dataset() for _ in range(test_size)]
print('building train datasets...')
train_datasets = [build_dataset() for _ in range(train_size)]
print('building validation datasets...')
val_datasets = [build_dataset() for _ in range(val_size)]


print('saving datasets...')
filename = os.path.join(dirs,'more-clevr.pickle')
with  open(filename, 'wb') as f:
    pickle.dump((train_datasets, test_datasets, val_datasets), f)
print('datasets saved at {}'.format(filename))
