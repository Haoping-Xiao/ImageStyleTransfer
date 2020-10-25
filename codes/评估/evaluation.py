# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 02:32:24 2019

@author: anktkyo
"""
from PIL import Image
import time
import os
import string
import numpy as np
from skimage import measure
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
import cv2
from matplotlib import pyplot as plt

#加载图片并且预处理
def load_img(img_path, target_size=(300,400)):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


#随机
def rand_unit_vec(size, seed=None):
    if seed != None:
        np.random.seed(seed)
    temp = np.random.randn(size)
    norm_temp = temp / np.sqrt(np.sum(np.square(temp)))

    return norm_temp

def normal_kl(mu_1, std_1, mu_2, std_2):
    # mu_1, std_1 is style feature
    # mu_2, std_2 is result feature
    return (np.log(std_2 / std_1)) + ((std_1 ** 2 + ((mu_1 - mu_2) ** 2)) / (2 * (std_2 ** 2))) - (0.5)

def normal_dist(img_path, rand_vec, feature_extractor):
    img = load_img(img_path)
    feature_map = feature_extractor([img])
    result_0 = np.reshape(np.dot(feature_map[0], rand_vec[:64]), -1).tolist()
    result_1 = np.reshape(np.dot(feature_map[1], rand_vec[64:192]), -1).tolist()
    result_2 = np.reshape(np.dot(feature_map[2], rand_vec[192:448]), -1).tolist()
    result_3 = np.reshape(np.dot(feature_map[3], rand_vec[448:960]), -1).tolist()
    result_4 = np.reshape(np.dot(feature_map[4], rand_vec[960:]), -1).tolist()
    scalar_data = result_0 + result_1 + result_2 + result_3 + result_4
    mu = np.mean(scalar_data)
    std = np.std(scalar_data)
    return mu, std

#根据内容图片与生成图片计算SSIM
def calc_ssim(content_img_path, result_img_path):
    content_img = cv2.imread(content_img_path)
    result_img = cv2.imread(result_img_path)
    content_img = cv2.resize(content_img, (np.shape(result_img)[1], np.shape(result_img)[0]))
    ssim_val = measure.compare_ssim(content_img, result_img, multichannel=True)
    return ssim_val


def calc_kl(base_model,get_feature,style_img,result_img,seed):
    #base_model = VGG19(weights='imagenet')
    #feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    #get_feature = K.function([base_model.layers[0].input],[base_model.get_layer(name).output[0] for name in feature_layers])
    np.random.seed(seed)
    vec_array = np.array([rand_unit_vec(1472) for _ in range(0, 128)])
    kl_arr = []
    for rand_vec_idx in range(128):
        mu_1, std_1 = normal_dist(img_path=style_img, rand_vec=vec_array[rand_vec_idx],feature_extractor=get_feature)
        mu_2, std_2 = normal_dist(img_path=result_img, rand_vec=vec_array[rand_vec_idx],feature_extractor=get_feature)
        kl_div = normal_kl(mu_1, std_1, mu_2, std_2)
        kl_arr.append(kl_div)

    result = np.mean(kl_arr)
    result=(-1)*np.log(result)
    #print(result)

    return result

if __name__=='__main__':
    
    start_time=time.time()
    base_model = VGG19(include_top=False,weights='imagenet')
    feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    get_feature = K.function([base_model.layers[0].input],[base_model.get_layer(name).output[0] for name in feature_layers])
    end_time=time.time()
    print("-Used %ds" %(end_time-start_time))
    cpath="./image/content-img/content-1.jpg"
    spath="./image/style-img/style-1.jpg"
    rpath="./eval/result-16.png"
    ssim_score=calc_ssim(cpath,rpath)
    start_time=time.time()
    kl_score=calc_kl(base_model,get_feature,spath,rpath,7777)
    end_time=time.time()
    print("-Used %ds" %(end_time-start_time))
    print("SSIM：" ,ssim_score)
    print("KL：",kl_score)
    plt.figure()
    img = Image.open(rpath)
    plt.imshow(img)
    plt.title("SSIM-%.3f   KL-%.3f" %(ssim_score,kl_score))
    plt.savefig("./eval/eval-16.png")
    cpath="./image/content-img/content-1.jpg"
    spath="./image/style-img/style-2.jpg"
    rpath="./eval/result-17.png"
    ssim_score=calc_ssim(cpath,rpath)
    start_time=time.time()
    kl_score=calc_kl(base_model,get_feature,spath,rpath,7777)
    end_time=time.time()
    print("-Used %ds" %(end_time-start_time))
    print("SSIM：" ,ssim_score)
    print("KL：",kl_score)
    plt.figure()
    img = Image.open(rpath)
    plt.imshow(img)
    plt.title("SSIM-%.3f   KL-%.3f" %(ssim_score,kl_score))
    plt.savefig("./eval/eval-17.png")
    cpath="./image/content-img/content-2.jpg"
    spath="./image/style-img/style-4.jpg"
    rpath="./eval/result-18.png"
    ssim_score=calc_ssim(cpath,rpath)
    start_time=time.time()
    kl_score=calc_kl(base_model,get_feature,spath,rpath,7777)
    end_time=time.time()
    print("-Used %ds" %(end_time-start_time))
    print("SSIM：" ,ssim_score)
    print("KL：",kl_score)
    plt.figure()
    img = Image.open(rpath)
    plt.imshow(img)
    plt.title("SSIM-%.3f   KL-%.3f" %(ssim_score,kl_score))
    plt.savefig("./eval/eval-18.png")
    cpath="./image/content-img/content-2.jpg"
    spath="./image/style-img/style-5.jpg"
    rpath="./eval/result-19.png"
    ssim_score=calc_ssim(cpath,rpath)
    start_time=time.time()
    kl_score=calc_kl(base_model,get_feature,spath,rpath,7777)
    end_time=time.time()
    print("-Used %ds" %(end_time-start_time))
    print("SSIM：" ,ssim_score)
    print("KL：",kl_score)
    plt.figure()
    img = Image.open(rpath)
    plt.imshow(img)
    plt.title("SSIM-%.3f   KL-%.3f" %(ssim_score,kl_score))
    plt.savefig("./eval/eval-19.png")

# =============================================================================
#     rpath="./result-111/"   #结果图的文件夹路径
#     resultlist=os.listdir(rpath)
#     ssim_array=np.zeros((150),dtype=np.float)
#     kl_array=np.zeros((150),dtype=np.float)
#     for item in resultlist:
#         resultimgpath=rpath+item
#         print(resultimgpath+":")
#         ssim_score=calc_ssim(cpath,resultimgpath)
#         index=item[6:]
#         index=index[:-4]
#         index=int(index)-1
#         print(index)
#         ssim_array[index]=ssim_score
#         start_time=time.time()
#         kl_score=calc_kl(base_model,get_feature,spath,resultimgpath,7777)
#         kl_array[index]=kl_score
#         end_time=time.time()
#         print("-Used %ds" %(end_time-start_time))
#         print(ssim_score)
#         print(kl_score)
#         
#     plt.figure()
#     x=np.array(range(150))
#     y1=np.array(ssim_array)
#     y2=np.array(kl_array)
#     plt.plot(x,y1,color='blue')
#     plt.plot(x,y2,color='red')
#     plt.show()
#     plt.savefig("./evaluation.png")
# =============================================================================
