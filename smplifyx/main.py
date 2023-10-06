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

import time
import yaml
import torch

import smplx

from utils import JointMapper
from cmd_parser import parse_config
from data_parser import create_dataset
from fit_single_frame import fit_single_frame
from fit_multi_view import fit_multi_view

from camera import create_camera
from prior import create_prior

torch.backends.cudnn.enabled = False


def main(**args):
    output_folder = args.pop('output_folder')
    output_folder = osp.expandvars(output_folder)
    if not osp.exists(output_folder):
        os.makedirs(output_folder)

    # Store the arguments for the current experiment
    conf_fn = osp.join(output_folder, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(args, conf_file)

    result_folder = args.pop('result_folder', 'results')
    result_folder = osp.join(output_folder, result_folder)
    if not osp.exists(result_folder):
        os.makedirs(result_folder)

    float_dtype = args['float_dtype']
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float64
    else:
        print('Unknown float type {}, exiting!'.format(float_dtype))
        sys.exit(-1)

    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    dataset_obj = create_dataset(**args)

    start = time.time()

    input_gender = args.pop('gender', 'neutral')

    float_dtype = args.get('float_dtype', 'float32')
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))

    joint_mapper = JointMapper(dataset_obj.get_model2data())

    model_params = dict(model_path=args.get('model_folder'),
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        dtype=dtype,
                        **args)

    male_model = smplx.create(gender='male', **model_params)
    neutral_model = smplx.create(gender='neutral', **model_params)
    female_model = smplx.create(gender='female', **model_params)

    # get priors
    rhand_args = args.copy()
    rhand_args['num_gaussians'] = args.get('num_pca_comps')
    right_hand_prior = create_prior(
        prior_type=args.get('right_hand_prior_type'),
        dtype=dtype,
        use_right_hand=True, # doesn't matter
        **rhand_args)

    shape_prior = create_prior(
        prior_type=args.get('shape_prior_type', 'l2'),
        dtype=dtype, **args)


    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')

        female_model = female_model.to(device=device)
        male_model = male_model.to(device=device)
        neutral_model = neutral_model.to(device=device)
        shape_prior = shape_prior.to(device=device)

        right_hand_prior = right_hand_prior.to(device=device)
    else:
        device = torch.device('cpu')

    # A weight for every joint of the model
    joint_weights = dataset_obj.get_joint_weights().to(device=device, dtype=dtype)
    # Add a fake batch dimension for broadcasting
    joint_weights.unsqueeze_(dim=0)

    camera_list = [camera.to(device) for camera in dataset_obj.camera_list]
    for idx, data in enumerate(dataset_obj):

        img_list = data['img_list']
        keypoints_list = data['keypoints_list']
        fn = data['fn']  # frame index 0000
        init_handmesh = data['handoccnet']
        print('Processing: {}'.format(data['img_path_list']))

        curr_result_folder = osp.join(result_folder, fn)
        if not osp.exists(curr_result_folder):
            os.makedirs(curr_result_folder)
        curr_result_fn = osp.join(curr_result_folder, 'result.pkl')
        curr_mesh_fn = osp.join(curr_result_folder, 'mesh.obj')

        curr_img_folder = osp.join(curr_result_folder, 'images')
        if not osp.exists(curr_img_folder):
            os.makedirs(curr_img_folder)
        
        out_img_fn_list = [osp.join(curr_img_folder, f'cam_{cam_idx}.png') for cam_idx in range(dataset_obj.num_cam)]
     
        gender = input_gender
        if gender == 'neutral':
            hand_model = neutral_model
        elif gender == 'female':
            hand_model = female_model
        elif gender == 'male':
            hand_model = male_model

        
        fit_multi_view(
            init_handmesh,
            img_list,
            keypoints_list,
            camera_list,
            hand_model=hand_model, # mano model
            joint_weights=joint_weights,
            dtype=dtype,
            output_folder=output_folder,
            result_folder=curr_result_folder,
            out_img_fn_list=out_img_fn_list,
            result_fn=curr_result_fn,
            mesh_fn=curr_mesh_fn,
            right_hand_prior=right_hand_prior,  
            shape_prior=shape_prior,   
        **args)


    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))


if __name__ == "__main__":
    args = parse_config()
    main(**args)
