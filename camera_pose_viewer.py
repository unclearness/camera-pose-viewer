#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

from numpy import linspace

import json

def inverse_homogeneoux_matrix(M):
    R = M[0:3, 0:3]
    T = M[0:3, 3]
    M_inv = np.identity(4)
    M_inv[0:3, 0:3] = R.T
    M_inv[0:3, 3] = -(R.T).dot(T)

    return M_inv


def transform_to_matplotlib_frame(cMo, X, inverse=False):
    M = np.identity(4)
    #M[1, 1] = 0
    #M[1, 2] = 1
    #M[2, 1] = -1
    #M[2, 2] = 0

    if inverse:
        return M.dot(inverse_homogeneoux_matrix(cMo).dot(X))
    else:
        return M.dot(cMo.dot(X))


def create_camera_model(camera_matrix, width, height, scale_focal, draw_frame_axis=False):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    focal = 2 / (fx + fy)
    f_scale = scale_focal * focal

    # draw image plane
    X_img_plane = np.ones((4, 5))
    X_img_plane[0:3, 0] = [-width, height, f_scale]
    X_img_plane[0:3, 1] = [width, height, f_scale]
    X_img_plane[0:3, 2] = [width, -height, f_scale]
    X_img_plane[0:3, 3] = [-width, -height, f_scale]
    X_img_plane[0:3, 4] = [-width, height, f_scale]

    # draw triangle above the image plane
    X_triangle = np.ones((4, 3))
    X_triangle[0:3, 0] = [-width, -height, f_scale]
    X_triangle[0:3, 1] = [0, -2*height, f_scale]
    X_triangle[0:3, 2] = [width, -height, f_scale]

    # draw camera
    X_center1 = np.ones((4, 2))
    X_center1[0:3, 0] = [0, 0, 0]
    X_center1[0:3, 1] = [-width, height, f_scale]

    X_center2 = np.ones((4, 2))
    X_center2[0:3, 0] = [0, 0, 0]
    X_center2[0:3, 1] = [width, height, f_scale]

    X_center3 = np.ones((4, 2))
    X_center3[0:3, 0] = [0, 0, 0]
    X_center3[0:3, 1] = [width, -height, f_scale]

    X_center4 = np.ones((4, 2))
    X_center4[0:3, 0] = [0, 0, 0]
    X_center4[0:3, 1] = [-width, -height, f_scale]

    # draw camera frame axis
    X_frame1 = np.ones((4, 2))
    X_frame1[0:3, 0] = [0, 0, 0]
    X_frame1[0:3, 1] = [f_scale/2, 0, 0]

    X_frame2 = np.ones((4, 2))
    X_frame2[0:3, 0] = [0, 0, 0]
    X_frame2[0:3, 1] = [0, f_scale/2, 0]

    X_frame3 = np.ones((4, 2))
    X_frame3[0:3, 0] = [0, 0, 0]
    X_frame3[0:3, 1] = [0, 0, f_scale/2]

    if draw_frame_axis:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4, X_frame1, X_frame2, X_frame3]
    else:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4]



def create_origin_model(length):
    # draw world axis
    X_frame1 = np.ones((4, 2))
    X_frame1[0:3, 0] = [0, 0, 0]
    X_frame1[0:3, 1] = [length, 0, 0]

    X_frame2 = np.ones((4, 2))
    X_frame2[0:3, 0] = [0, 0, 0]
    X_frame2[0:3, 1] = [0, length, 0]

    X_frame3 = np.ones((4, 2))
    X_frame3[0:3, 0] = [0, 0, 0]
    X_frame3[0:3, 1] = [0, 0, length]

    return [X_frame1, X_frame2, X_frame3]


def draw_objects(ax, objs, min_values, max_values, color, w2c=np.eye(4)):
    for i in range(len(objs)):
        X = np.zeros(objs[i].shape)
        for j in range(objs[i].shape[1]):
            X[:, j] = transform_to_matplotlib_frame(
                w2c, objs[i][:, j])
        ax.plot3D(X[0, :], X[1, :], X[2, :], color=color)
        min_values = np.minimum(min_values, X[0:3, :].min(1))
        max_values = np.maximum(max_values, X[0:3, :].max(1))
    return min_values, max_values

def draw_cameras(ax, camera_parameters, cam_width, cam_height, scale_focal, draw_origin=True):
    from matplotlib import cm

    min_values = np.zeros((3, 1))
    min_values = np.inf
    max_values = np.zeros((3, 1))
    max_values = -np.inf
    #X_ca = create_camera_model(camera_matrix, cam_width, cam_height, scale_focal, True)

    cm_subsection = linspace(0.0, 1.0, len(camera_parameters))
    colors = [cm.jet(x) for x in cm_subsection]

    if draw_origin:
        origin = create_origin_model((cam_width + cam_height)*0.5)
        min_values, max_values = draw_objects(ax, [origin[0]],
                                    min_values, max_values, 'r')
        min_values, max_values = draw_objects(ax, [origin[1]],
                                    min_values, max_values, 'g')
        min_values, max_values = draw_objects(ax, [origin[2]],
                                    min_values, max_values, 'b')

        arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)
        from matplotlib.patches import FancyArrowPatch
        from mpl_toolkits.mplot3d import proj3d
        class Arrow3D(FancyArrowPatch):
            def __init__(self, xs, ys, zs, *args, **kwargs):
                FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
                self._verts3d = xs, ys, zs
            def draw(self, renderer):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
                self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
                FancyArrowPatch.draw(self, renderer)
        a = Arrow3D([0, 1], [0, 0], [0, 0], **arrow_prop_dict, color='r')
        ax.add_artist(a)
        a = Arrow3D([0, 0], [0, 1], [0, 0], **arrow_prop_dict, color='g')
        ax.add_artist(a)
        a = Arrow3D([0, 0], [0, 0], [0, 1], **arrow_prop_dict, color='b')
        ax.add_artist(a)
        #max_values = np.ones((3, 1))

    for i, cam_param in enumerate(camera_parameters):
        K = np.array(cam_param['K'])
        camera = create_camera_model(K, cam_width,
                                    cam_height, scale_focal, True)
        w2c = np.eye(4)
        w2c[:3, :] = np.array(cam_param['world2cam'])
        c2w = np.linalg.inv(w2c)
        #print(w2c, c2w)
        min_values, max_values =\
             draw_objects(ax, camera, min_values, max_values, colors[i], c2w)
        label = cam_param['name']
        x, y, z = c2w[0, 3], c2w[1, 3], c2w[2, 3]
        zdir = c2w[:3, 2]
        #print(x, y, z, zdir)
        ax.text(x, y, z, label, 'x', color=colors[i])
    # for i in range(len(X_static)):
    #     X = np.zeros(X_static[i].shape)
    #     for j in range(X_static[i].shape[1]):
    #         X[:, j] = transform_to_matplotlib_frame(
    #             np.eye(4), X_static[i][:, j])
    #     ax.plot3D(X[0, :], X[1, :], X[2, :], color='r')
    #     min_values = np.minimum(min_values, X[0:3, :].min(1))
    #     max_values = np.maximum(max_values, X[0:3, :].max(1))

    # for idx in range(extrinsics.shape[0]):
    #     R, _ = cv.Rodrigues(extrinsics[idx, 0:3])
    #     cMo = np.eye(4, 4)
    #     cMo[0:3, 0:3] = R
    #     cMo[0:3, 3] = extrinsics[idx, 3:6]
    #     for i in range(len(X_moving)):
    #         X = np.zeros(X_moving[i].shape)
    #         for j in range(X_moving[i].shape[1]):
    #             X[0:4, j] = transform_to_matplotlib_frame(
    #                 cMo, X_moving[i][0:4, j], patternCentric)
    #         ax.plot3D(X[0, :], X[1, :], X[2, :], color=colors[idx])
    #         min_values = np.minimum(min_values, X[0:3, :].min(1))
    #         max_values = np.maximum(max_values, X[0:3, :].max(1))

    return min_values, max_values


def load_json(path):
    with open(path) as f:
        return json.load(f)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Plot camera calibration extrinsics.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--camera_parameters', type=str, default='camera_data.json',
                        help='camera calibration file.')
    parser.add_argument('--cam_width', type=float, default=0.064*5,
                        help='Width/2 of the displayed camera.')
    parser.add_argument('--cam_height', type=float, default=0.048*5,
                        help='Height/2 of the displayed camera.')
    parser.add_argument('--scale_focal', type=float, default=400,
                        help='Value to scale the focal length.')
    # parser.add_argument('--patternCentric', action='store_true',
    #                    help='The calibration board is static and the camera is moving.')
    args = parser.parse_args()

    #fs = cv.FileStorage(cv.samples.findFile(
    #    args.calibration), cv.FILE_STORAGE_READ)
    #board_width = int(fs.getNode('board_width').real())
    #board_height = int(fs.getNode('board_height').real())
    #square_size = fs.getNode('square_size').real()
    #camera_matrix = fs.getNode('camera_matrix').mat()
    #extrinsics = fs.getNode('extrinsic_parameters').mat()

    camera_matrix_list = []
    c2w_list = []

    camera_parameters = load_json(args.camera_parameters)
    # for cam in j:
    #     camera_matrix = cam['K']
    #    w2c = 'world2cam'

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-variable

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("auto")

    cam_width = args.cam_width
    cam_height = args.cam_height
    scale_focal = args.scale_focal
    min_values, max_values = draw_cameras(ax, camera_parameters, cam_width, cam_height,
                                                scale_focal)

    X_min = min_values[0]
    X_max = max_values[0]
    Y_min = min_values[1]
    Y_max = max_values[1]
    Z_min = min_values[2]
    Z_max = max_values[2]
    max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

    mid_x = (X_max+X_min) * 0.5
    mid_y = (Y_max+Y_min) * 0.5
    mid_z = (Z_max+Z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Extrinsic Parameters Visualization')

    plt.show()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
