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

import time

import numpy as np

import torch
import torch.nn as nn

from mesh_viewer import MeshViewer
import utils
from vis import vis_3d_skeleton


class FittingMonitor(object):
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=100, ftol=2e-09, gtol=1e-05,
                 hand_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.hand_color = hand_color
        self.model_type = model_type

    def __enter__(self):
        self.steps = 0
        if self.visualize:
            self.mv = MeshViewer(body_color=self.hand_color)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize:
            self.mv.close_viewer()

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    def run_fitting(self, optimizer, closure, params, hand_model,
                    use_vposer=True, pose_embedding=None, vposer=None,
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                hand_model: nn.Module
                    The hand model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''
        append_wrists = self.model_type == 'smpl' and use_vposer
        prev_loss = None
        for n in range(self.maxiters):
            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break

            if self.visualize and n % self.summary_steps == 0:
                hand_pose = None
                model_output = hand_model(
                    return_verts=True, hand_pose=hand_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    hand_model.faces)

            prev_loss = loss.item()

        return prev_loss

    def create_fitting_closure_multiview(self,
                                            optimizer, hand_model,
                                            camera_list=None, global_hand_translation=None,
                                            hand_model_scale=None,
                                            target_joints_list=None, loss_list=None,
                                            joints_conf_list=None,
                                            joint_weights=None,
                                            return_verts=True, return_full_pose=False,
                                            use_vposer=False, vposer=None,
                                            pose_embedding=None,
                                            create_graph=False,
                                            **kwargs):
        faces_tensor = hand_model.faces_tensor.view(-1)

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            hand_pose = None

            hand_model_output = hand_model(return_verts=return_verts,
                                           hand_pose=hand_pose,
                                           return_full_pose=return_full_pose)
            total_loss = 0

            for i in range(len(camera_list)):
                loss = loss_list[i]
                total_loss += loss(hand_model_output, camera=camera_list[i],
                                   global_hand_translation=global_hand_translation,
                                   hand_model_scale=hand_model_scale,
                                   target_joints=target_joints_list[i],
                                   hand_model_faces=faces_tensor,
                                   joints_conf=joints_conf_list[i],
                                   joint_weights=joint_weights,
                                   **kwargs)

            if backward:
                total_loss.backward(create_graph=create_graph)

            self.steps += 1
            if self.visualize and self.steps % self.summary_steps == 0:
                model_output = hand_model(return_verts=True,
                                          hand_pose=hand_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    hand_model.faces)

            return total_loss

        return fitting_func
    

def create_loss(loss_type='smplify', **kwargs):
    if loss_type == 'smplify':
        return SMPLifyLoss(**kwargs)
    elif loss_type == 'camera_init':
        raise ValueError('Unknown loss type: {}'.format(loss_type))
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))


class SMPLifyLoss(nn.Module):

    def __init__(self, search_tree=None,
                 pen_distance=None, tri_filtering_module=None,
                 rho=100,
                 shape_prior=None,
                 use_joints_conf=True,
                 right_hand_prior=None,
                 interpenetration=True, dtype=torch.float32,
                 data_weight=1.0,
                 shape_weight=0.0,
                 bending_prior_weight=0.0,
                 hand_prior_weight=0.0,
                 coll_loss_weight=0.0,
                 reduction='sum',
                 **kwargs):

        super(SMPLifyLoss, self).__init__()

        self.model_type = kwargs['model_type']
        self.use_joints_conf = use_joints_conf

        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho


        self.shape_prior = shape_prior

        self.interpenetration = interpenetration
        if self.interpenetration:
            self.search_tree = search_tree
            self.tri_filtering_module = tri_filtering_module
            self.pen_distance = pen_distance

        self.right_hand_prior = right_hand_prior

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('shape_weight',
                             torch.tensor(shape_weight, dtype=dtype))
        self.register_buffer('bending_prior_weight',
                             torch.tensor(bending_prior_weight, dtype=dtype))
        self.register_buffer('hand_prior_weight',
                                torch.tensor(hand_prior_weight, dtype=dtype))
        if self.interpenetration:
            self.register_buffer('coll_loss_weight',
                                 torch.tensor(coll_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, hand_model_output, camera, 
                global_hand_translation, hand_model_scale,
                target_joints, joints_conf,
                hand_model_faces, joint_weights,
                **kwargs):
        projected_joints = camera(
            hand_model_scale * hand_model_output.joints + global_hand_translation) 
        # Calculate the weights for each joints
        weights = (joint_weights * joints_conf
                   if self.use_joints_conf else
                   joint_weights).unsqueeze(dim=-1)

        # Calculate the distance of the projected joints from
        # the ground truth 2D detections
        joint_diff = self.robustifier(target_joints - projected_joints)
        joint_loss = (torch.sum(weights ** 2 * joint_diff) *
                      self.data_weight ** 2)

        shape_loss = torch.sum(self.shape_prior(
            hand_model_output.betas)) * self.shape_weight ** 2


        # Apply the prior on the pose space of the hand
        right_hand_prior_loss = 0.0
        right_hand_prior_loss = torch.sum(
            self.right_hand_prior(
                hand_model_output.hand_pose)) * \
            self.hand_prior_weight ** 2
    
        pen_loss = 0.0
        # Calculate the loss due to interpenetration
        if (self.interpenetration and self.coll_loss_weight.item() > 0):
            batch_size = projected_joints.shape[0]
            triangles = torch.index_select(
                hand_model_output.vertices, 1,
                hand_model_faces).view(batch_size, -1, 3, 3)

            with torch.no_grad():
                collision_idxs = self.search_tree(triangles)

            # Remove unwanted collisions
            if self.tri_filtering_module is not None:
                collision_idxs = self.tri_filtering_module(collision_idxs)

            if collision_idxs.ge(0).sum().item() > 0:
                pen_loss = torch.sum(
                    self.coll_loss_weight *
                    self.pen_distance(triangles, collision_idxs))

        total_loss = (joint_loss + shape_loss + pen_loss + right_hand_prior_loss)
        return total_loss

