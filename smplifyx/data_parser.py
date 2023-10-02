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
            return OpenPose(data_folder, **kwargs)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    keypoints = []

    gender_pd = []
    gender_gt = []
    for idx, person_data in enumerate(data['people']):
        # body_keypoints = np.array(person_data['pose_keypoints_2d'],dtype=np.float32)

        # Custom
        body_keypoints = np.zeros((25,3), dtype=np.float32)

        body_keypoints = body_keypoints.reshape([-1, 3])
        if use_hands:
            # left_hand_keyp = np.array(person_data['hand_left_keypoints_2d'], dtype=np.float32).reshape([-1, 3])
            # Custom

            left_hand_keyp = np.zeros((21,3), dtype=np.float32)

            # index 46 to 66
            # 0,1,5,9,13,17
            right_hand_keyp = np.array(
                person_data['hand_right_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])

            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
        if use_face:
            # TODO: Make parameters, 17 is the offset for the eye brows,
            # etc. 51 is the total number of FLAME compatible landmarks
            face_keypoints = np.array(
                person_data['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

            contour_keyps = np.array(
                [], dtype=body_keypoints.dtype).reshape(0, 3)
            if use_face_contour:
                contour_keyps = np.array(
                    person_data['face_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])[:17, :]

            body_keypoints = np.concatenate(
                [body_keypoints, face_keypoints, contour_keyps], axis=0)

        if 'gender_pd' in person_data:
            gender_pd.append(person_data['gender_pd'])
        if 'gender_gt' in person_data:
            gender_gt.append(person_data['gender_gt'])

        keypoints.append(body_keypoints)

    return Keypoints(keypoints=keypoints, gender_pd=gender_pd,
                     gender_gt=gender_gt)


def read_hand_keypoints(keypoint_fn, conf_thr=0.12):
    if osp.exists(keypoint_fn):
        with open(keypoint_fn) as keypoint_file:
            data = json.load(keypoint_file)

        keypoints = []

        gender_pd = []
        gender_gt = []
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

class OpenPose(Dataset):

    NUM_BODY_JOINTS = 25
    NUM_HAND_JOINTS = 20

    def __init__(self, data_folder, img_folder='images',
                 keyp_folder='keypoints',
                 use_hands=False,
                 use_face=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 use_face_contour=False,
                 openpose_format='coco25',
                 **kwargs):
        super(OpenPose, self).__init__()

        self.use_hands = use_hands
        self.use_face = use_face
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = use_face_contour

        self.openpose_format = openpose_format

        self.num_joints = (self.NUM_BODY_JOINTS +2 * self.NUM_HAND_JOINTS * use_hands)

        self.img_folder = osp.join(data_folder, img_folder)
        self.keyp_folder = osp.join(data_folder, keyp_folder)

        self.img_paths = [osp.join(self.img_folder, img_fn)
                          for img_fn in os.listdir(self.img_folder)
                          if img_fn.endswith('.png') or
                          img_fn.endswith('.jpg') and
                          not img_fn.startswith('.')]
        self.img_paths = sorted(self.img_paths)
        self.cnt = 0

    def get_model2data(self):
        return smpl_to_openpose(self.model_type, use_hands=self.use_hands,
                                use_face=self.use_face,
                                use_face_contour=self.use_face_contour,
                                openpose_format=self.openpose_format)

    def get_left_shoulder(self):
        return 2

    def get_right_shoulder(self):
        return 5

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
                                self.use_face * 51 +
                                17 * self.use_face_contour,
                                dtype=np.float32)

        # Custom
        # optim_weights[25:46] = 0.

        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        return self.read_item(img_path)

    def read_item(self, img_path):
        img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
        img_fn = osp.split(img_path)[1]
        img_fn, _ = osp.splitext(osp.split(img_path)[1])

        keypoint_fn = osp.join(self.keyp_folder,
                               img_fn + '_keypoints.json')
        keyp_tuple = read_keypoints(keypoint_fn, use_hands=self.use_hands,
                                    use_face=self.use_face,
                                    use_face_contour=self.use_face_contour)

        if len(keyp_tuple.keypoints) < 1:
            return {}
        keypoints = np.stack(keyp_tuple.keypoints)

        output_dict = {'fn': img_fn,
                       'img_path': img_path,
                       'keypoints': keypoints, 'img': img}
        if keyp_tuple.gender_gt is not None:
            if len(keyp_tuple.gender_gt) > 0:
                output_dict['gender_gt'] = keyp_tuple.gender_gt
        if keyp_tuple.gender_pd is not None:
            if len(keyp_tuple.gender_pd) > 0:
                output_dict['gender_pd'] = keyp_tuple.gender_pd
        return output_dict

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= len(self.img_paths):
            raise StopIteration

        img_path = self.img_paths[self.cnt]
        self.cnt += 1

        return self.read_item(img_path)


class OpenPoseMano(Dataset):

    def __init__(self, data_folder, img_folder='images',
                 keyp_folder='keypoints',
                 use_hands=False,
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

        self.num_joints = 21 
        self.num_cam = 7
        self.data_folder = data_folder
        self.img_path_list = sorted(glob.glob(osp.join(data_folder,  'cam_0', '*.jpg')))
        self.keyp_folder_list = sorted(glob.glob(osp.join(data_folder,  'cam_0_keypoints', '*_keypoints.json')))
        # TEMP
        # self.img_path_list = self.img_path_list[65:]
        # self.keyp_folder_list = self.keyp_folder_list[65:]

        self.depth_folder = osp.join(data_folder, 'cam_0_depth')
        cam_path = '/home/hongsuk.c/Projects/MultiCamCalib/data/handnerf_calibration_0822/output/cam_params/cam_params_final.json'
        self.camera_list = self.get_camera_list(cam_path)

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
        base_keyp_path = self.keyp_folder_list[idx]
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
            
            img_fn = osp.basename(base_img_path)
            img_fn = str(cam_idx) + img_fn[1:]  # 0_0000.jpg
            img_path = osp.join(self.data_folder, img_folder_name, img_fn)
            keyp_fn = img_fn.replace('.jpg', '_keypoints.json')
            keyp_path = osp.join(self.data_folder, keyp_folder_name, keyp_fn)

            img_path_list.append(img_path)
        
            img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
            keypoints = read_hand_keypoints(keyp_path)
            keypoints = np.stack(keypoints)

            img_list.append(img)
            keypoints_list.append(keypoints)

        output_dict = {
            'img_path_list': img_path_list,
            'keypoints_list': keypoints_list, 
            'img_list': img_list,
        }

        #
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
