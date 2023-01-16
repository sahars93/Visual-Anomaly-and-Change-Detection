import numpy as np
import csv  
import math
from matplotlib import pyplot as plt  
import seaborn as sns  
import cv2
import json
sns.set()
import subprocess
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.projects import point_rend
from sklearn.cluster import KMeans, DBSCAN


colors = [ 'red']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])


def overlap_points(matched_points):
    match_p = matched_points
    min_x, max_x = int(min(match_p[:,0])),int(max(match_p[:,0]))
    min_y, max_y = int(min(match_p[:,1])),int(max(match_p[:,1]))
    return min_y,max_y,min_x,max_x
def segment(masked_im):
    img0_msk = masked_im
    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3 
    cfg.MODEL.WEIGHTS = "my_data/model_final_2d9806.pkl" #download from https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
    predictor = DefaultPredictor(cfg)
    outputs = predictor(img0_msk)
    return outputs, cfg
def point_comparison(mask,mp,nmp):
    match_p0 = mp
    not_match0 = nmp
    msk0 = mask
    msk1=(msk0 == True).nonzero()
    msk2=msk1.tolist()
    msk3 = np.array(msk2)
    msk=np.flip(msk3, axis=1).tolist()
    match_points = [i for i in match_p0 if i in msk]
    not_match_points = [i for i in not_match0 if i in msk]
    return match_points, not_match_points


def not_msked(mask,points):
    newpoint=[]
    IMM=mask
    Y0 = points
    for i,j in zip(Y0[:,0], Y0[:,1]):
        i = int(i)
        j = int(j)
        if IMM[j,i,0]==1.0:
            newpoint.append([i,j])
    return newpoint

def dbscan(not_match_points):
    points = not_match_points
    clsfr = DBSCAN(eps=15, min_samples=10)
    clsfr.fit(points)
    clusters = clsfr.labels_
 
    return clusters

subprocess.call(['python3', 'match_pairs.py', '--viz', '--show_keypoints',
                '--max_length','--input_pairs','images/images_pair.txt', 
                '1000', '--match_threshold', '0.4',
                '--keypoint_threshold','0.003','--input_dir','images/',
                 '--output_dir','matching_results/',])

f=open('images/images_pair.txt')
lines = f.readlines()
new_list = [s.replace("\n", "").replace(".jpg", "").split(' ') for s in lines]

data = np.load('matching_results/{}_{}_matches.npz'.format(new_list[0][0],new_list[0][1]))

img1 = cv2.imread('images/{}.jpg'.format(new_list[0][1]))
img0 = cv2.imread('images/{}.jpg'.format(new_list[0][0])) #anomaly


img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

img1 = cv2.resize(img1, (640, 480))
img0 = cv2.resize(img0, (640, 480))

final_mask = np.zeros((img0.shape[0], img0.shape[1]))
final_mask_all = np.zeros((img0.shape[0], img0.shape[1]))

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

indx_p1 = data['matches1']
indx_p0 = data['matches0']

kp11 = data['keypoints1']
kp0 = data['keypoints0']
ds1 = data['descriptors1']
ds0 = data['descriptors0']

classes = []
not_matched_points = []
cnt_p = 0
ext = []
ext_m = []
match_segment =[]

kp1=np.zeros(np.shape(kp0))
for o,d in enumerate(indx_p0):
    if d > -1:
        kp1[o]=kp11[d]

#matching
valid_mp0 = indx_p0 > -1
match_p0 = kp0[valid_mp0]
match_p1 = kp1[valid_mp0]

#notmatching
valid_np0 = indx_p0 == -1
not_match0 = kp0[valid_np0] 

valid_np1 = indx_p1 == -1
not_match1 = kp11[valid_np1]

not_match0_list = not_match0.tolist()
match_p0_list = match_p0.tolist()

if len(match_p0)>0:

    min_y0,max_y0,min_x0,max_x0 = overlap_points(match_p0)
    mask0 = np.zeros((img0.shape[0], img0.shape[1],3), dtype=np.uint8)
    mask0[min_y0:max_y0,min_x0:max_x0,:]=1
    img0_msk = img0 * mask0
    #####segment************
    outputs, cfg = segment(img0_msk)
    print(len(outputs["instances"].pred_masks))
    for mask_seg in range(len(outputs["instances"].pred_masks)):
        msk0=outputs["instances"].pred_masks[mask_seg]
        mps_points, nmps_points = point_comparison(msk0,match_p0_list,not_match0_list)
        nmps = len(nmps_points)
        mps = len(mps_points)

        mask_l_all = msk0.cpu()
        mask_l_arr_all = mask_l_all[...,None].numpy().reshape((img0.shape[0], img0.shape[1])).astype(int)  
        final_mask_all = final_mask_all + mask_l_arr_all

        if nmps - mps >1:
            
            mask_l = msk0.cpu()
            mask_l_arr = mask_l[...,None].numpy().reshape((img0.shape[0], img0.shape[1])).astype(int)  
            final_mask = final_mask + mask_l_arr

            clss = outputs["instances"].pred_classes[mask_seg].cpu()
            class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
            match_segment.append(mps_points)
            not_matched_points.append(nmps_points)



for c in range(len(not_matched_points)):
    ext.extend(not_matched_points[c])

remain_points = [i for i in not_match0_list if i not in ext]
remain_points_arr = np.array(remain_points) 
masked_points = not_msked(mask0,remain_points_arr)
masked_points_arr = np.array(masked_points)


print(len(masked_points_arr))
if len(masked_points_arr)>0:
    db_res = dbscan(masked_points_arr)

    clus=[]
    for i,c in enumerate(db_res):
        if c == -1:
            clus.append(i)
    final_nmp = np.delete(masked_points_arr, clus, axis=0)
    clusters = np.delete(db_res, clus)

color = np.array([255,0,0], dtype='uint8')


masked_img = np.where(final_mask[...,None], color, img0)
out = cv2.addWeighted(img0_msk, 0.8, masked_img, 0.4,0)
plt.imshow(out)
plt.axis('off')


plt.savefig('results_powerpoint/final_seg2.jpg',dpi=200,bbox_inches='tight',pad_inches=0)

plt.scatter(final_nmp[:,0],final_nmp[:,1],s=10,c='red')
plt.savefig('results_powerpoint/a_final.jpg',dpi=200,bbox_inches='tight',pad_inches=0)
plt.clf()

out2 = cv2.addWeighted(img0_msk, 0.8, img0, 0.4,0)
plt.imshow(out2)
plt.axis('off')
print(np.shape(mask0))
xg = [not_match0[i,0] for i in range(len(not_match0)) if mask0[int(not_match0[i,1]),int(not_match0[i,0]),1]==1]
yg = [not_match0[i,1] for i in range(len(not_match0)) if mask0[int(not_match0[i,1]),int(not_match0[i,0]),1]==1]
xgr = np.array(xg)
ygr = np.array(yg)

xr = [match_p0[i,0] for i in range(len(match_p0)) if mask0[int(match_p0[i,1]),int(match_p0[i,0]),1]==1]
yr = [match_p0[i,1] for i in range(len(match_p0)) if mask0[int(match_p0[i,1]),int(match_p0[i,0]),1]==1]
xrr = np.array(xr)
yrr = np.array(yr)

plt.scatter(xgr,ygr,s=2,c='red')
plt.scatter(xrr,yrr,s=2,c='red')
plt.savefig('results_powerpoint/overlapedPoints.jpg',dpi=200,bbox_inches='tight',pad_inches=0)
# plt.show()


plt.imshow(img0)
plt.axis('off')
plt.scatter(not_match0[:,0],not_match0[:,1],s=2,c='red')
plt.scatter(match_p0[:,0],match_p0[:,1],s=2,c='green')
plt.savefig('results_powerpoint/scatterPoints.jpg',dpi=200,bbox_inches='tight',pad_inches=0)
plt.clf()


masked_img_all = np.where(final_mask_all[...,None], color, img0)
out3 = cv2.addWeighted(img0_msk, 0.8, masked_img_all, 0.4,0)
plt.imshow(out3)
plt.axis('off')

plt.savefig('results_powerpoint/all_seg.jpg',dpi=200,bbox_inches='tight',pad_inches=0)
