import subprocess
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt 
from kneebow.rotor import Rotor
from kneed import DataGenerator, KneeLocator


plt.style.use('seaborn-poster')
count = 0
num_pairs = 6
max_step_k = 0.009
step_k= 0.001
max_step_m = 0.9
step_m= 0.1
num_cam = 3
length = (len(np.arange(0, max_step_k, step_k))) * (len(np.arange(0, max_step_m, step_m)))
points0 = np.zeros((len(np.arange(0, max_step_m, step_m)),num_pairs,2))
points1 = np.zeros((len(np.arange(0, max_step_m, step_m)),num_pairs,2))
points2 = np.zeros((len(np.arange(0, max_step_m, step_m)),num_pairs,2))
# list_k = list(np.arange(0, max_step_k, step_k))
list_m = list(np.arange(0, max_step_m, step_m))
list_k = [0.003]

print('total comparison for finding thresholds : {}'.format(length))


for i, val_k in enumerate(list_k):
    for j , val_m in enumerate(list_m):
        print('comparison for matching_thresh: {}, and keypoint_thresh: {}'.format(val_m,val_k))
        subprocess.call(['python3', 'superpoint_superglue/match_pairs.py', '--viz', '--input_pairs','calib.txt',
        '--input_dir','few_shots/', '--show_keypoints','--output_dir','calibration_results/',
        '--max_length' ,f'{num_cam*num_pairs}','--keypoint_threshold', f'{val_k}',
        '--match_threshold', f'{val_m}'])
        

        for cam, points in enumerate([points0, points1,points2]):

            for p in range(num_pairs):
                m = np.load('calibration_results/tresh{}{}0_tresh{}{}1_matches.npz'.format(cam,p,cam,p))
                b1 = m['matches1']
                b0 = m['matches0']

                k1 = m['keypoints1']
                k0 = m['keypoints0']

                d1 = m['descriptors1']
                d0 = m['descriptors0']

                valid0 = b0 > -1
                X = k0[valid0]
                valid1 = b1 > -1
                Y = k1[valid1]
                if len(X) != 0 and len(Y) != 0:
                    d=[]
                    for k in range(len(X)):
                        d.append(norm(X[k]-Y[k]))
                    dis=np.array(d) 

                    st=np.std(dis)
                    mean=np.mean(dis)
                    sm=st/mean

                    points[j,p, 0] = sm/len(X)
                    points[j,p, 1] = val_m
        # count = count + 1
    mini0 = np.min(points0, axis=1)
    maxi0 = np.max(points0, axis=1)
    mean0=np.mean(points0, axis=1)
    plt.fill_between(mean0[:,1], mini0[:,0], maxi0[:,0],alpha=.2, linewidth=0)
    plt.plot(mean0[:,1], mean0[:,0],linewidth=3,color='red',label="cam_0")
    kl0 = KneeLocator(mean0[:,1], mean0[:,0], curve="convex",direction='increasing',interp_method='interp1d', online=True)
    plt.vlines(round(kl0.knee,8), 0, 1, linestyles='--',colors='red', label='opt_0')
    print(round(kl0.knee,8))
    plt.ylim([0, 0.01])
    plt.xlabel("matching_tresh")
    plt.ylabel("m_cv/N")


    mini1 = np.min(points1, axis=1)
    maxi1 = np.max(points1, axis=1)
    mean1=np.mean(points1, axis=1)
    plt.fill_between(mean1[:,1], mini1[:,0], maxi1[:,0], alpha=.2, linewidth=0)
    plt.plot(mean1[:,1], mean1[:,0],linewidth=3,color='blue',label="cam_1")
    kl1 = KneeLocator(mean1[:,1], mean1[:,0], curve="convex",direction='increasing',interp_method='interp1d', online=True)
    plt.vlines(round(kl1.knee,8), 0, 1, linestyles='--',colors='blue', label='opt_1')
    print(round(kl1.knee,8))
    plt.ylim([0, 0.01])


    mini2 = np.min(points2, axis=1)
    maxi2 = np.max(points2, axis=1)
    mean2=np.mean(points2, axis=1)
    plt.fill_between(mean2[:,1], mini2[:,0], maxi2[:,0],alpha=.2, linewidth=0)
    plt.plot(mean2[:,1], mean2[:,0],linewidth=3,color='black',label="cam_2")
    kl2 = KneeLocator(mean2[:,1], mean2[:,0], curve="convex",direction='increasing',interp_method='interp1d', online=True)
    plt.vlines(round(kl2.knee,8), 0, 1, linestyles='--',colors='black', label='opt_2')
    print(round(kl2.knee,8))
    plt.ylim([0, 0.01])

    plt.legend()
    plt.xlabel("matching_tresh")
    plt.ylabel("cv")
    plt.savefig('calibration_results/cameras_key_thresh{}.png'.format(val_k))
    plt.clf() 

