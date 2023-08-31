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


import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
import os
import os.path as osp
import copy

import numpy as np
import torch
from mesh_viewer import MeshViewer


from tqdm import tqdm

from collections import defaultdict

import cv2
import PIL.Image as pil_img

from optimizers import optim_factory

import fitting
from human_body_prior.tools.model_loader import load_vposer


def fit_multi_view(
                     depth,
                     img_list,
                     keypoints_list,
                     camera_list,
                     body_model,
                     joint_weights,
                     body_pose_prior,
                     jaw_prior,
                     left_hand_prior,
                     right_hand_prior,
                     shape_prior,
                     expr_prior,
                     angle_prior,
                     result_fn='out.pkl',
                     mesh_fn='out.obj',
                     out_img_fn_list=['overlay.png'],
                     loss_type='smplify',
                     use_cuda=True,
                     init_joints_idxs=(9, 12, 2, 5),
                     use_face=True,
                     use_hands=True,
                     data_weights=None,
                     body_pose_prior_weights=None,
                     hand_pose_prior_weights=None,
                     jaw_pose_prior_weights=None,
                     shape_weights=None,
                     expr_weights=None,
                     hand_joints_weights=None,
                     face_joints_weights=None,
                     depth_loss_weight=1e2,
                     interpenetration=True,
                     coll_loss_weights=None,
                     df_cone_height=0.5,
                     penalize_outside=True,
                     max_collisions=8,
                     point2plane=False,
                     part_segm_fn='',
                     focal_length=5000.,
                     side_view_thsh=25.,
                     rho=100,
                     vposer_latent_dim=32,
                     vposer_ckpt='',
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     save_meshes=True,
                     degrees=None,
                     batch_size=1,
                     dtype=torch.float32,
                     ign_part_pairs=None,
                     left_shoulder_idx=2,
                     right_shoulder_idx=5,
                     **kwargs):
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'

    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    if degrees is None:
        degrees = [0, 90, 180, 270]

    if data_weights is None:
        data_weights = [1, ] * 5

    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]

    msg = (
        'Number of Body pose prior weights {}'.format(
            len(body_pose_prior_weights)) +
        ' does not match the number of data term weights {}'.format(
            len(data_weights)))
    assert (len(data_weights) ==
            len(body_pose_prior_weights)), msg

    if use_hands:
        if hand_pose_prior_weights is None:
            hand_pose_prior_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of hand pose prior weights')
        assert (len(hand_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg
        if hand_joints_weights is None:
            hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
            msg = ('Number of Body pose prior weights does not match the' +
                   ' number of hand joint distance weights')
            assert (len(hand_joints_weights) ==
                    len(body_pose_prior_weights)), msg

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of Body pose prior weights = {} does not match the' +
           ' number of Shape prior weights = {}')
    assert (len(shape_weights) ==
            len(body_pose_prior_weights)), msg.format(
                len(shape_weights),
                len(body_pose_prior_weights))

    if coll_loss_weights is None:
        coll_loss_weights = [0.0] * len(body_pose_prior_weights)
    msg = ('Number of Body pose prior weights does not match the' +
           ' number of collision loss weights')
    assert (len(coll_loss_weights) ==
            len(body_pose_prior_weights)), msg


    view_num = len(camera_list)
    loss_list = list()
    gt_joints_list = list()
    joints_conf_list = list()

    assert(view_num > 0)
    for view_id in range(view_num):
        keypoint_data = torch.tensor(keypoints_list[view_id], dtype=dtype)
        gt_joints = keypoint_data[:, :, :2]
        if use_joints_conf:
            joints_conf = keypoint_data[:, :, 2].reshape(1, -1)

        # Transfer the data to the correct device
        gt_joints = gt_joints.to(device=device, dtype=dtype)
        gt_joints_list.append(gt_joints)
        if use_joints_conf:
            joints_conf = joints_conf.to(device=device, dtype=dtype)
            joints_conf_list.append(joints_conf)

        # Create the search tree
        search_tree = None
        pen_distance = None
        filter_faces = None
        if interpenetration:
            raise NotImplementedError(
                'The interpenetration constraint was removed!')

        # Weights used for the pose prior and the shape prior
        opt_weights_dict = {'data_weight': data_weights,
                            'body_pose_weight': body_pose_prior_weights,
                            'shape_weight': shape_weights}
        if use_face:
            opt_weights_dict['face_weight'] = face_joints_weights
            opt_weights_dict['expr_prior_weight'] = expr_weights
            opt_weights_dict['jaw_prior_weight'] = jaw_pose_prior_weights
        if use_hands:
            opt_weights_dict['hand_weight'] = hand_joints_weights
            opt_weights_dict['hand_prior_weight'] = hand_pose_prior_weights
        if interpenetration:
            opt_weights_dict['coll_loss_weight'] = coll_loss_weights

        keys = opt_weights_dict.keys()
        opt_weights = [dict(zip(keys, vals)) for vals in
                    zip(*(opt_weights_dict[k] for k in keys
                            if opt_weights_dict[k] is not None))]
        for weight_list in opt_weights:
            for key in weight_list:
                weight_list[key] = torch.tensor(weight_list[key],
                                                device=device,
                                                dtype=dtype)

      
        loss = fitting.create_loss(loss_type=loss_type,
                                joint_weights=joint_weights,
                                rho=rho,
                                use_joints_conf=use_joints_conf,
                                use_face=use_face, use_hands=use_hands,
                                body_pose_prior=body_pose_prior,
                                shape_prior=shape_prior,
                                angle_prior=angle_prior,
                                expr_prior=expr_prior,
                                left_hand_prior=left_hand_prior,
                                right_hand_prior=right_hand_prior,
                                jaw_prior=jaw_prior,
                                interpenetration=interpenetration,
                                pen_distance=pen_distance,
                                search_tree=search_tree,
                                tri_filtering_module=filter_faces,
                                dtype=dtype,
                                **kwargs)
        loss = loss.to(device=device)
        loss_list.append(loss)

    # potentially use depth info from the wrist cam
    hand_scale = torch.tensor([1.0 / 1.0], dtype=dtype, device=device,requires_grad=True)
    global_hand_translation = torch.tensor([0, 0, 0], dtype=dtype, device=device,requires_grad=True)
    
    with fitting.FittingMonitor(
            batch_size=batch_size, visualize=visualize, **kwargs) as monitor:


        data_weight = 2.
     
        try_both_orient = False

        orient = body_model.global_orient.detach().cpu().numpy()

        # store here the final error for both orientations,
        # and pick the orientation resulting in the lowest error


        # Step 2: Optimize the full model
        final_loss_val = 0
        opt_start = time.time()

        # new_params = defaultdict(global_orient=orient,
        #                          body_pose=body_mean_pose)
        new_params = defaultdict(global_orient=orient)
        body_model.reset_params(**new_params) # if not designated, reset to zreo

        
        for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):
            body_params = list(body_model.parameters())

            final_params = list(
                filter(lambda x: x.requires_grad, body_params))
            final_params.append(global_hand_translation)
            final_params.append(hand_scale)

            body_optimizer, body_create_graph = optim_factory.create_optimizer(
                final_params,
                **kwargs)
            body_optimizer.zero_grad()

            curr_weights['data_weight'] = data_weight
            curr_weights['bending_prior_weight'] = (
                3.17 * curr_weights['body_pose_weight'])
            if use_hands:
                joint_weights[:, 25:67] = curr_weights['hand_weight']
            if use_face:
                joint_weights[:, 67:] = curr_weights['face_weight']
            for i in range(len(loss_list)):
                loss_list[i].reset_loss_weights(curr_weights)

            closure = monitor.create_fitting_closure_multiview(
                body_optimizer, body_model,
                camera_list=camera_list, global_body_translation=global_hand_translation,
                body_model_scale=hand_scale,
                gt_joints_list=gt_joints_list,
                joints_conf_list=joints_conf_list,
                joint_weights=joint_weights,
                loss_list=loss_list, create_graph=body_create_graph,
                return_verts=True, return_full_pose=True)
                
            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                stage_start = time.time()
            final_loss_val = monitor.run_fitting(
                body_optimizer,
                closure, final_params,
                body_model
            )

            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - stage_start
                if interactive:
                    tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
                        opt_idx, elapsed))
                    

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - opt_start
            tqdm.write(
                'Body fitting Orientation done after {:.4f} seconds'.format(elapsed))
            tqdm.write('Body final loss val = {:.5f}'.format(
                final_loss_val))

        # Get the result of the fitting process
        # Store in it the errors list in order to compare multiple
        # orientations, if they exist
        result = {}
        result.update({key: val.detach().cpu().numpy()
                        for key, val in body_model.named_parameters()})
        result['global_hand_translation'] = global_hand_translation.detach().cpu().numpy()
        result['hand_scale'] = hand_scale.detach().cpu().numpy()
        result['loss'] = final_loss_val

        with open(result_fn, 'wb') as result_file:            
            pickle.dump(result, result_file, protocol=2)

    if save_meshes or visualize:
        model_output = body_model(return_verts=True, body_pose=None)
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()

        # test projection
        global_trans = global_hand_translation.detach().cpu().numpy().squeeze()
        body_scale = hand_scale.detach().cpu().numpy().squeeze()

        # project smpl vertices onto images for debugging
        for i, (camera, img, out_img_fn) in enumerate(zip(camera_list, img_list, out_img_fn_list)):
            cam_fx = camera.focal_length_x.detach().cpu().numpy().squeeze()
            cam_fy = camera.focal_length_y.detach().cpu().numpy().squeeze()
            cam_c = camera.center.detach().cpu().numpy().squeeze()
            cam_trans = camera.translation.detach().cpu().numpy().squeeze()
            cam_rotation = camera.rotation.detach().cpu().numpy().squeeze()

            vertices_proj = vertices * body_scale + global_trans
            vertices_proj = np.dot(vertices_proj, cam_rotation.transpose())
            vertices_proj += np.expand_dims(cam_trans, axis=0)
            vertices_proj[:, 0] = vertices_proj[:, 0] * \
                cam_fx / vertices_proj[:, 2] + cam_c[0]
            vertices_proj[:, 1] = vertices_proj[:, 1] * \
                cam_fy / vertices_proj[:, 2] + cam_c[1]
            img_proj = np.copy(img)
            for v in vertices_proj:
                v = np.int32(np.round(v))
                v[0] = np.clip(v[0], 0, img_proj.shape[1]-1)
                v[1] = np.clip(v[1], 0, img_proj.shape[0]-1)
                img_proj[v[1], v[0], :] = np.asarray(
                    [0, 0, 1], dtype=np.float32)
            img_proj = np.uint8(img_proj*255)
            cv2.imwrite(out_img_fn, img_proj[:, :, ::-1])
        import pdb
        pdb.set_trace()

        import trimesh
        out_mesh = trimesh.Trimesh(
            vertices * body_scale + global_trans, body_model.faces)
        out_mesh.export(mesh_fn)
