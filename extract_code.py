import sys, os
sys.path.insert(0, "/home/username/usr/caffe_ssd/python")
import caffe

import cv2
import numpy as np
from caffe.proto import caffe_pb2
from PIL import Image
import matplotlib.pyplot as plt
import time

if len(sys.argv) <= 5:
    print 'usage'
    print './main model_folder test_gt batch_size save_file'
    exit(-1)

model_file = sys.argv[1]
deploy_file = sys.argv[2]
test_file = sys.argv[3]
batch_size = int(sys.argv[4])
save_file = sys.argv[5]

model_path = model_file
deploy_path = deploy_file

model_folder = '../models/'
deploy_path_generate = os.path.join(model_folder, 'deploy_gen.prototxt')

f = open(deploy_path,'r')
lines = f.readlines()
f.close()
if 'input_dim' in lines[2]:
    lines[2] = 'input_dim: '+str(batch_size)+'\n'
else:
    lines[3] = '    dim: '+str(batch_size)+'\n'

f = open(deploy_path_generate,'w')
f.writelines(lines)
f.close()


# for line in open(sub_2_label_path):
#     cls, idx = line.strip().rsplit(' ', 1)
#     sub_list[int(idx)] = cls
#     sub_exist[cls] = 1
#     sub_map[cls] = int(idx)


caffe.set_device(0)
caffe.set_mode_gpu()

CONF=deploy_path_generate
MODEL=model_path
net = caffe.Net(CONF, MODEL, caffe.TEST)

def cal_prob(net, data):
    output = net.forward(**{"data":data})
    prob_fc5 = output["fc5"]

    return prob_fc5

def nimg_to_img(nimg):
    img = Image.fromarray(np.array(nimg.swapaxes(0,1).swapaxes(1,2), dtype=np.uint8))
    return img
def nimg_to_cvimg(nimg):
    cvimg = np.array(nimg.swapaxes(0,1).swapaxes(1,2), dtype=np.uint8)
    return cvimg
def cvimg_to_nimg(cvimg):
    nimg = np.array(cvimg.swapaxes(1,2).swapaxes(0,1), dtype=np.float32)
    return nimg
def img_to_nimg(img):
    nimg = np.array(img, dtype=np.float32)
    nimg = nimg.swapaxes(1,2).swapaxes(0,1)
    return nimg
def sub_local(nimg):
    a = nimg_to_cvimg(nimg)
    a = cv2.addWeighted(a, 1, cv2.GaussianBlur(a, (0,0), a.shape[1]/20), -1, 128)
    return cvimg_to_nimg(a)


img_idx = 0
data_list = []

resize_height=128
resize_width=64

img_list = []
gt_list = []

file_object = open(test_file)
list_of_all_the_lines = file_object.readlines( )

for line in list_of_all_the_lines:
    #print line
    temp = line.strip().split()
    #pos = temp.find('.png')
    img = temp[0].strip()
    label = temp[1].strip()

    cvimg = cv2.imread('../data/' + img)
    if cvimg is None:
        print 'read faild: ', line
        continue
    
    crop = cv2.resize(cvimg,(resize_width, resize_height))
    # cv2.imshow('', crop)
    # cv2.waitKey(0)

    data = np.array(crop, dtype=np.float32)
    #print data.shape
    #data = histeq(data)
    data = data.swapaxes(1,2).swapaxes(0,1)#C H W
    #print data.shape



    data -= 128.0
    data /= 100.0
    data = data.reshape((3, resize_height, resize_width))

    data_list.append(data)

    img_list.append(img)
    gt_list.append(label)

time_sum = 0
total_main_list = []
hash_results = []
for i in range(len(data_list)/batch_size):
    input_list = data_list[i*batch_size:(i+1)*batch_size]
    #print np.array(list_data[0]).shape
    #print np.array(data_list).shape
    time_begin1 = time.clock()
    prob_fc5_all = cal_prob(net, np.array(input_list))
    time_end1 = time.clock()
    time_sum += time_end1 - time_begin1
    #print prob.shape
            
    for j in range(batch_size):
        prob_fc5 = prob_fc5_all[j]
        tmp = ''
        for idx, k in enumerate(prob_fc5):
            if k >= 0: tmp += ' 1'
            else: tmp += ' -1'
            if idx == 0: tmp = tmp.replace(' ', '')
        gt = gt_list[i * batch_size + j]

        hash_results.append(tmp + ' ' + gt + '\n')

        img_idx += 1

fw = open(save_file, 'w')
fw.writelines(hash_results)
fw.close()

print 'batch_size:',batch_size
print 'time', time_sum*1.0/(len(data_list)/batch_size)*1000,'ms'
print 'img_idx', img_idx
