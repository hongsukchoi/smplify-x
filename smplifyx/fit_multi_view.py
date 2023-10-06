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
from utils import to_tensor




def fit_multi_view(
        handoccnet_result,
                     img_list,
                     keypoints_list,
                     camera_list,
                     hand_model,
                     joint_weights,
                     right_hand_prior,
                     shape_prior,
                     fit_hand_scale=False,
                     result_fn='out.pkl',
                     mesh_fn='out.obj',
                     out_img_fn_list=['overlay.png'],
                     loss_type='smplify',
                     use_cuda=True,
                     data_weights=None,
                     hand_pose_prior_weights=None,
                     shape_weights=None,
                     hand_joints_weights=None,
                     interpenetration=True,
                     coll_loss_weights=None,
                     rho=100,
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     save_meshes=True,
                     degrees=None,
                     batch_size=1,
                     dtype=torch.float32,
                     **kwargs):
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    if degrees is None:
        degrees = [0, 90, 180, 270]

    if data_weights is None:
        data_weights = [1, ] * 5


    if hand_pose_prior_weights is None:
        hand_pose_prior_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of hand pose prior weights does not match the' +
            ' number of data term weights')
    assert (len(hand_pose_prior_weights) ==
            len(data_weights)), msg
    if hand_joints_weights is None:
        hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of hand joint distance weights does not match the' +
                ' number of data term weights')
        assert (len(hand_joints_weights) ==
                len(data_weights)), msg

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of shape prior weights = {} does not match the' +
           ' number of data term weights = {}')
    assert (len(shape_weights) ==
            len(data_weights)), msg.format(
                len(shape_weights),
                len(data_weights))

    if coll_loss_weights is None:
        coll_loss_weights = [0.0] * len(data_weights)
    msg = ('Number of collision loss weights does not match the' +
           ' number of data term weights')
    assert (len(coll_loss_weights) ==
            len(data_weights)), msg

    view_num = len(camera_list)
    loss_list = list()
    target_joints_list = list() 
    joints_conf_list = list()

    assert(view_num > 0)
    for view_id in range(view_num):
        keypoint_data = torch.tensor(keypoints_list[view_id], dtype=dtype)
        target_joints = keypoint_data[:, :, :2]
        if use_joints_conf:
            joints_conf = keypoint_data[:, :, 2].reshape(1, -1)

        # Transfer the data to the correct device
        target_joints = target_joints.to(device=device, dtype=dtype)
        target_joints_list.append(target_joints)
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
                            'shape_weight': shape_weights,
                            'hand_weight': hand_joints_weights,
                            'hand_prior_weight': hand_pose_prior_weights
                        }
        
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
                                shape_prior=shape_prior,
                                right_hand_prior=right_hand_prior,
                                interpenetration=interpenetration,
                                pen_distance=pen_distance,
                                search_tree=search_tree,
                                tri_filtering_module=filter_faces,
                                dtype=dtype,
                                **kwargs)
        loss = loss.to(device=device)
        loss_list.append(loss)

    # potentially use depth info from the wrist cam
    hand_scale = torch.tensor([1.0 / 1.0], dtype=dtype, device=device,requires_grad=fit_hand_scale)
    global_hand_translation = torch.tensor([0, 0, 0], dtype=dtype, device=device,requires_grad=True)
    
    with fitting.FittingMonitor(
            batch_size=batch_size, visualize=visualize, **kwargs) as monitor:

        data_weight = 2.
     
        orient = hand_model.global_orient.detach().cpu().numpy()

        # Step 2: Optimize the full model
        final_loss_val = 0
        opt_start = time.time()

        # initialize pose here.
        # new_params = defaultdict(global_orient=orient,
        #                          hand_pose=hand_mean_pose)
        use_handoccnet = True
        if use_handoccnet:
            orient = torch.tensor(handoccnet_result['mano_pose'][:3], dtype=dtype, device=device).reshape(1,3)
            mano_pose = torch.tensor(handoccnet_result['mano_pose'][3:], dtype=dtype, device=device).reshape(1,45)
            mano_shape = torch.tensor(handoccnet_result['mano_shape'][:], dtype=dtype, device=device).reshape(1,10)
            hand_scale = torch.tensor(handoccnet_result['hand_scale'][:], dtype=dtype, device=device, requires_grad=False)
            global_hand_translation = torch.tensor(handoccnet_result['hand_translation'], dtype=dtype, device=device, requires_grad=True)
        new_params = defaultdict(global_orient=orient, hand_pose=mano_pose, betas=mano_shape)
        hand_model.reset_params(**new_params) # if not designated, reset to zreo

        for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):
            hand_params = list(hand_model.parameters())

            final_params = list(
                filter(lambda x: x.requires_grad, hand_params))
            final_params.append(global_hand_translation)
            final_params.append(hand_scale)

            hand_optimizer, hand_create_graph = optim_factory.create_optimizer(
                final_params,
                **kwargs)
            hand_optimizer.zero_grad()

            curr_weights['data_weight'] = data_weight
            for i in range(len(loss_list)):
                loss_list[i].reset_loss_weights(curr_weights)

            closure = monitor.create_fitting_closure_multiview(
                hand_optimizer, hand_model,
                camera_list=camera_list, global_hand_translation=global_hand_translation,
                hand_model_scale=hand_scale,
                target_joints_list=target_joints_list,
                joints_conf_list=joints_conf_list,
                joint_weights=joint_weights,
                loss_list=loss_list, create_graph=hand_create_graph,
                return_verts=True, return_full_pose=True)
                
            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                stage_start = time.time()
            final_loss_val = monitor.run_fitting(
                hand_optimizer,
                closure, final_params,
                hand_model
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
                'Hand fitting Orientation done after {:.4f} seconds'.format(elapsed))
            tqdm.write('Hand final loss val = {:.5f}'.format(
                final_loss_val))

        # Get the result of the fitting process
        # Store in it the errors list in order to compare multiple
        # orientations, if they exist
        result = {}
        result.update({key: val.detach().cpu().numpy()
                        for key, val in hand_model.named_parameters()})
        result['global_hand_translation'] = global_hand_translation.detach().cpu().numpy()
        result['hand_scale'] = hand_scale.detach().cpu().numpy()
        result['loss'] = final_loss_val

        with open(result_fn, 'wb') as result_file:            
            pickle.dump(result, result_file, protocol=2)

    return 
    mv = MeshViewer()
    if save_meshes or visualize:
        import pyrender
        import trimesh

        model_output = hand_model(return_verts=True, hand_pose=None)
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()

        # update translation and scale
        global_trans = global_hand_translation.detach().cpu().numpy().squeeze()
        hand_scale = hand_scale.detach().cpu().numpy().squeeze()
        vertices = vertices * hand_scale + global_trans


        out_mesh = trimesh.Trimesh(vertices, hand_model.faces, process=False)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        # out_mesh.apply_transform(rot)
        out_mesh.export(mesh_fn)
        # project smpl vertices onto images for debugging
        for i, (camera, img, out_img_fn) in enumerate(zip(camera_list, img_list, out_img_fn_list)):
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='OPAQUE',
                baseColorFactor=(1.0, 1.0, 0.9, 1.0))

            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                ambient_light=(0.3, 0.3, 0.3))
            

            cam_fx = camera.focal_length_x.detach().cpu().numpy().squeeze()
            cam_fy = camera.focal_length_y.detach().cpu().numpy().squeeze()
            cam_c = camera.center.detach().cpu().numpy().squeeze()
            cam_trans = camera.translation.detach().cpu().numpy().squeeze()
            cam_rotation = camera.rotation.detach().cpu().numpy().squeeze()

            cam_mesh = copy.deepcopy(out_mesh)
            transform = np.eye(4)
            transform[:3, :3] = cam_rotation
            transform[:3, 3] = cam_trans
            cam_mesh.apply_transform(transform)
            cam_mesh.apply_transform(rot)
            mesh = pyrender.Mesh.from_trimesh(
                cam_mesh,
                material=material)
            scene.add(mesh, 'mesh')


            camera_pose = np.eye(4)
            # camera_pose[:3, :3] = cam_rotation
            # camera_pose[:3, 3] = cam_trans
            # camera_pose[:, 1:3] = -camera_pose[:, 1:3]
            # camera_pose = np.linalg.inv(camera_pose)

            camera = pyrender.camera.IntrinsicsCamera(
                fx=cam_fx, fy=cam_fy, cx=cam_c[0], cy=cam_c[1])
            scene.add(camera, pose=camera_pose)

            # custom
            # # Get the lights from the viewer
            light_nodes = mv.viewer._create_raymond_lights()
            for node in light_nodes:
                scene.add_node(node)
            mv.close_viewer()

            H, W = img.shape[:2]
            r = pyrender.OffscreenRenderer(viewport_width=W,
                                        viewport_height=H,
                                        point_size=1.0)
            color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
            color = color.astype(np.float32) / 255.0

            valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
            output_img = (color[:, :, :-1] * valid_mask +
                        (1 - valid_mask) * img)

            img = pil_img.fromarray((output_img * 255).astype(np.uint8))
            img.save(out_img_fn)

            # cam_fx = camera.focal_length_x.detach().cpu().numpy().squeeze()
            # cam_fy = camera.focal_length_y.detach().cpu().numpy().squeeze()
            # cam_c = camera.center.detach().cpu().numpy().squeeze()
            # cam_trans = camera.translation.detach().cpu().numpy().squeeze()
            # cam_rotation = camera.rotation.detach().cpu().numpy().squeeze()

            # vertices_proj = vertices * hand_scale + global_trans
            # vertices_proj = np.dot(vertices_proj, cam_rotation.transpose())
            # vertices_proj += np.expand_dims(cam_trans, axis=0)
            # vertices_proj[:, 0] = vertices_proj[:, 0] * \
            #     cam_fx / vertices_proj[:, 2] + cam_c[0]
            # vertices_proj[:, 1] = vertices_proj[:, 1] * \
            #     cam_fy / vertices_proj[:, 2] + cam_c[1]
            # img_proj = np.copy(img)
            # for v in vertices_proj:
            #     v = np.int32(np.round(v))
            #     v[0] = np.clip(v[0], 0, img_proj.shape[1]-1)
            #     v[1] = np.clip(v[1], 0, img_proj.shape[0]-1)
            #     img_proj[v[1], v[0], :] = np.asarray(
            #         [0, 0, 1], dtype=np.float32)
            # img_proj = np.uint8(img_proj*255)
            # cv2.imwrite(out_img_fn, img_proj[:, :, ::-1])

        out_mesh = trimesh.Trimesh(
            vertices * hand_scale + global_trans, hand_model.faces)
        out_mesh.export(mesh_fn)
