# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import glob

import json

from collections import namedtuple

import cv2
import numpy as np

from PIL import Image
from camera import create_camera

import torch
from torch.utils.data import Dataset


from utils import smpl_to_openpose

Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'gender_gt', 'gender_pd'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)


def create_dataset(dataset='openpose', data_folder='data', **kwargs):
    if dataset.lower() == 'openpose':
        if kwargs['model_type'] == 'mano':
            return OpenPoseMano(data_folder, **kwargs)
        else:
            raise ValueError('Unknown model type: {}'.format(kwargs['model_type']))

    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def read_hand_keypoints(keypoint_fn, conf_thr=0.12):
    if osp.exists(keypoint_fn):
        with open(keypoint_fn) as keypoint_file:
            data = json.load(keypoint_file)

        keypoints = []

        for idx, person_data in enumerate(data['people']):
            right_hand_keyp = np.array(
                    person_data['hand_right_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])
            # make low confidence 0
            right_hand_keyp[:, 2] = right_hand_keyp[:, 2] * (right_hand_keyp[:, 2] > conf_thr)

            keypoints.append(right_hand_keyp)

        # just one person...
        return keypoints
    else:
        return [np.zeros((21,3))]
    
class OpenPoseMano(Dataset):

    def __init__(self, data_folder, 
                 joints_conf_thr=0.12,
                 dtype=torch.float32,
                 model_type='mano',
                 joints_to_ign=None,
                 openpose_format='coco25',
                 **kwargs):
        super(OpenPoseMano, self).__init__()

        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.openpose_format = openpose_format

        self.data_folder = data_folder
        self.img_path_list = sorted(glob.glob(osp.join(data_folder,  'cam_0', '*.jpg')))
        # self.keyp_path_list = sorted(glob.glob(osp.join(data_folder,  'cam_0_keypoints', '*_keypoints.json')))
        self.depth_folder = osp.join(data_folder, 'cam_0_depth')

        self.joints_conf_thr = joints_conf_thr
        self.num_joints = 21 
        cam_path = osp.join(data_folder, 'cam_params_final.json') 
        self.camera_list = self.get_camera_list(cam_path)
        self.num_cam = len(self.camera_list)

        self.cnt = 0

    def get_camera_list(self, cam_path):
        camera_list = []
        with open(cam_path, 'r') as f:
            cam_data = json.load(f)
        
        for cam_idx in sorted(cam_data.keys(), key=lambda x: int(x)):
            camera = create_camera()

            camera.focal_length_x = torch.full([1], cam_data[cam_idx]['fx'])
            camera.focal_length_y = torch.full([1], cam_data[cam_idx]['fy'])
            camera.center = torch.tensor([cam_data[cam_idx]['cx'], cam_data[cam_idx]['cy']]).unsqueeze(0)
            rotation, _ = cv2.Rodrigues(np.array(cam_data[cam_idx]['rvec'], dtype=np.float32))
            camera.rotation.data = torch.from_numpy(rotation).unsqueeze(0)
            camera.translation.data = torch.tensor(cam_data[cam_idx]['tvec']).unsqueeze(0) / 1000.
            camera.rotation.requires_grad = False
            camera.translation.requires_grad = False
            camera.name = str(cam_idx)

            camera_list.append(camera)
        
        return camera_list


    def get_model2data(self):
        return smpl_to_openpose(self.model_type, use_hands=True,
                                use_face=False,
                                use_face_contour=False,
                                openpose_format=self.openpose_format)

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints, dtype=np.float32)

        return torch.tensor(optim_weights, dtype=self.dtype)

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        base_img_path = self.img_path_list[idx]
        # base_keyp_path = self.keyp_path_list[idx]
        base_depth_path = osp.join(self.depth_folder, osp.basename(base_img_path.replace('jpg', 'png')))

        item = self.read_item(base_img_path)
        depth = np.asarray(Image.open(base_depth_path))
        item['depth'] = depth
        item['fn'] = base_img_path.split('_')[-1][:-4]  # 0000 ; frame idx

        return item

    def read_item(self, base_img_path):

        img_path_list = []
        keypoints_list = []
        img_list = []

        for cam_idx in range(0, self.num_cam):
            img_folder_name = f'cam_{cam_idx}'
            keyp_folder_name = f'cam_{cam_idx}_keypoints'
            
            # read images
            img_fn = osp.basename(base_img_path)
            img_fn = str(cam_idx) + img_fn[1:]  # 0_0000.jpg
            img_path = osp.join(self.data_folder, img_folder_name, img_fn)

            # read OpenPose hand joints
            keyp_fn = img_fn.replace('.jpg', '_keypoints.json')
            keyp_path = osp.join(self.data_folder, keyp_folder_name, keyp_fn)

            img_path_list.append(img_path)
        
            img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
            keypoints = read_hand_keypoints(keyp_path, conf_thr=self.joints_conf_thr)
            keypoints = np.stack(keypoints)

            img_list.append(img)
            keypoints_list.append(keypoints)

        output_dict = {
            'img_path_list': img_path_list,
            'keypoints_list': keypoints_list, 
            'img_list': img_list,
        }

        # hand mesh initialiation
        mano_folder_name = f'cam_0_handoccnet'
        img_fn = osp.basename(base_img_path)
        mano_fn = '0' + img_fn[1:-4] + '_3dmesh.json'  #
        mano_path = osp.join(self.data_folder, mano_folder_name, mano_fn)
        camera = self.camera_list[0]
        with open(mano_path, 'r') as f:
            mano_data = json.load(f)
        hand_scale = np.array(mano_data['hand_scale'], dtype=float)  # (1)
        hand_translation = np.array(mano_data['hand_translation'], dtype=float) # (3)
        mano_pose = np.array(mano_data['mano_pose'], dtype=float) # (48)
        mano_shape = np.array(mano_data['mano_shape'], dtype=float)  # (10)

        # get camera to world transformation
        rot = camera.rotation.data.cpu().numpy()[0, :, :]
        trans = camera.translation.data.cpu().numpy()[0, :]
        rot = rot.T
        trans = - rot @ trans 

        global_ori, _ = cv2.Rodrigues(mano_pose[:3])
        global_ori_aa, _ = cv2.Rodrigues(rot @ global_ori)
        mano_pose[:3] = global_ori_aa[:, 0]

        hand_translation = rot @ hand_translation + trans
        output_dict['handoccnet'] = {
            'hand_scale': hand_scale,
            'hand_translation': hand_translation,
            'mano_pose': mano_pose,
            'mano_shape': mano_shape
        }


        return output_dict

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= len(self.img_path_list):
            raise StopIteration

        self.cnt += 1

        return self.__getitem__(self.cnt-1)
