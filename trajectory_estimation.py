import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

img1 = cv2.imread('./img1.png')
img2 = cv2.imread('./img2.png')
img3 = cv2.imread('./img3.png')

vr2d = np.load('./vr2d.npy')
vr3d = np.load('./vr3d.npy')

fund_mat2 = cv2.findFundamentalMat(vr3d,vr2d)

f_len = 100
cx, cy = [960,540]

camera_mat = np.array([[f_len,0,cx],
                      [0,f_len,cy],
                      [0,0,1]],np.float32)

distCoefs = np.zeros((4,1))

success, rotation_vector, translation_vector = cv2.solvePnP(objectPoints=vr3d, imagePoints=vr2d, cameraMatrix=camera_mat, distCoeffs= distCoefs)

rotM, _ = cv2.Rodrigues(rotation_vector)

Tmat = np.concatenate([np.concatenate([rotM,translation_vector], axis=1),np.array([0,0,0,1]).reshape(1,4)],axis=0)


def kps_to_np(keypoints):
    kps = [key.pt for key in keypoints]
    return np.array(kps)

def orb_features(img):

    orb = cv2.ORB_create(nfeatures=1500)
    keypoints_orb, descriptors = orb.detectAndCompute(img, None)

    return keypoints_orb, descriptors

def keypoints_new(im1,im2):

    lk_params = dict(winSize  = (10, 10),
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01))

    key1, desc1 = orb_features(im1)
    key1 = kps_to_np(key1)

    key2,st,err = cv2.calcOpticalFlowPyrLK(im1,im2,key1.astype(np.float32),None,**lk_params)

    st = st.flatten()

    good_key2 = key2[st == 1]
    good_key1 = key1[st == 1]

    return good_key1,good_key2

def euler_angles(R):
    roll = math.atan2(-R[2][1], R[2][2])
    pitch = math.asin(R[2][0])
    yaw = math.atan2(-R[1][0], R[0][0])

    return roll, pitch, yaw

def get_position(R,t):

    R = R.transpose()
    pos = np.matmul(-R,t)

    return pos

def main(im1,im2):

    if im1.ndim > 2 or im2.ndim > 2 :
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    key1, key2 = keypoints_new(im1,im2)

    E, mask = cv2.findEssentialMat(key1,key2,camera_mat,method=cv2.RANSAC,prob=0.999,threshold=1.0)

    points, rmat, tvec, mask = cv2.recoverPose(E, key1, key2)

    pos = get_position(rmat,tvec)
    roll, pitch, yaw = euler_angles(rmat)

    return pos, [roll, pitch, yaw], tvec, rmat

def visualize_orb(img):
    keypoints = orb_features(img)[0]

    plt.figure(figsize=(10, 10))
    img_ = cv2.drawKeypoints(img, keypoints, None)
    plt.imshow(img_)
    plt.show()

def bf_matcher(first_frame, second_frame):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(first_frame, None)
    kp2, des2 = orb.detectAndCompute(second_frame, None)

    # Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    matching_result = cv2.drawMatches(first_frame, kp1, second_frame, kp2, matches, None, flags=2)

    return matching_result

def matcher_visualize(image1, image2):
    mr = bf_matcher(image1, image2)
    plt.figure(figsize=(15,15))
    plt.imshow(mr)
    plt.show()

def visualize(t):
    plt.figure()
    plt.xlabel('Sides')
    plt.ylabel('Forward-Back')
    plt.plot(t[:,0],t[:,2])
    plt.savefig('./resulting_trajectory.png')
    plt.show()

if __name__ == '__main__':

    pos_1, angles_1, t_1, rmat_1 = main(img1,img2)
    pos_2, angles_2, t_2, rmat_2 = main(img1,img3)

    f = open("output.txt", "a")
    print('Parameters of the change between img1 and img2:  \n\n'
          'Rotation Matrix :   \n{}'.format(rmat_1),'\n\n'
          'Translation Vector :   \n{}'.format(t_1),'\n\n\n', file=f)

    print('Parameters of the change between img1 and img3:  \n\n'
          'Rotation Matrix :   \n{}'.format(rmat_2),'\n\n'
          'Translation Vector :   \n{}'.format(t_2),'\n\n\n', file=f)
    f.close()

    visualize(np.array([np.zeros_like(pos_1),pos_1,pos_2]))

    print('Results saved in the same directory.')


