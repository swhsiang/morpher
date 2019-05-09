#!/usr/bin/env python3.7

import numpy as np
import os
from scipy.spatial import Delaunay
from scipy import interpolate
import imageio
from matplotlib import path


class Morpher:
    """
    h = np.linalg.solve(A, b)
    """
    leftImage: 'np.ndarray[np.uint8]'
    leftTriangles: 'List[Triangle]'
    rightImage: 'np.ndarray[np.uint8]'
    rightTriangles: 'List[Triangle]'

    def __init__(self, leftImage: np.ndarray, leftTriangles,
                 rightImage: np.ndarray, rightTriangles):
        if (not isinstance(leftImage, np.ndarray)) or (not isinstance(
                rightImage, np.ndarray)):
            raise TypeError("Invalid input arguments")
        if (leftImage.dtype != "uint8") or (rightImage.dtype != "uint8"):
            raise TypeError("Invalid input arguments (uint8)")

        if any(
                map(lambda x: not isinstance(x, list),
                    [leftTriangles, rightTriangles])):
            raise TypeError("Invalid input arguments, List")

        if any(
                map(
                    lambda x: any(
                        map(lambda y: not isinstance(y, Triangle), x)),
                    [leftTriangles, rightTriangles])):
            raise TypeError("Invalid input arguments, (triangles)")

        self.leftImage = leftImage
        self.leftTriangles = leftTriangles
        self.rightImage = rightImage
        self.rightTriangles = rightTriangles

    def getImageAtAlpha(self, alpha: float) -> np.ndarray:
        height, width = self.leftImage.shape

        result = np.zeros((height, width), dtype=np.uint8)

        xx = range(0, height)
        yy = range(0, width)

        alphaed_left_img = np.multiply(
            self.leftImage, (1 - alpha), casting='unsafe')
        alphaed_right_img = np.multiply(
            self.rightImage, alpha, casting='unsafe')

        f_left_new = interpolate.RectBivariateSpline(xx, yy, alphaed_left_img)
        f_right_new = interpolate.RectBivariateSpline(xx, yy,
                                                      alphaed_right_img)

        for l_tri, r_tri in zip(self.leftTriangles, self.rightTriangles):

            # Find intermediate triangle
            temp = (1 - alpha) * l_tri.vertices + alpha * r_tri.vertices
            final_t = Triangle(temp)

            final_p_in_polygon = final_t.getPoints()

            # Left triange to final triangle
            inv_final_to_left_trnasf = calculate_transfor_and_inverse(
                l_tri.vertices, final_t.vertices)

            # Right triangle to final triangle
            inv_final_to_right_trnasf = calculate_transfor_and_inverse(
                r_tri.vertices, final_t.vertices)

            extra_ones = np.ones((final_p_in_polygon.shape[0], 1))

            transpose_inv_h_final_to_left = inv_final_to_left_trnasf.T[:, :2]
            transpose_inv_h_final_to_right = inv_final_to_right_trnasf.T[:, :2]
            points = np.hstack((final_p_in_polygon, extra_ones))

            # NOTE Find corresponding points of final triangle on left/right triangles
            np_left_p_in_polygon = points.dot(
                transpose_inv_h_final_to_left)[:, ::-1]
            np_right_p_in_polygon = points.dot(
                transpose_inv_h_final_to_right)[:, ::-1]

            left_p_color = f_left_new.ev(np_left_p_in_polygon[:, 0],
                                         np_left_p_in_polygon[:, 1])
            right_p_color = f_right_new.ev(np_right_p_in_polygon[:, 0],
                                           np_right_p_in_polygon[:, 1])

            result[final_p_in_polygon[:, 1],
                   final_p_in_polygon[:, 0]] = np.add(left_p_color,
                                                      right_p_color)

        return result

    def saveVideo(self,
                  targetFilePath: str,
                  frameCount: int,
                  frameRate: int,
                  includeReversed=True):
        writer = imageio.get_writer(targetFilePath, fps=frameRate)

        pngs = []

        writer.append_data(self.getImageAtAlpha(0))
        for i in range(1, frameCount - 1):
            temp = self.getImageAtAlpha( i / (frameCount - 1))
            if includeReversed:
                pngs.append(temp)
            writer.append_data(temp)

        while pngs:
            writer.append_data(pngs.pop(-1))
        else:
            writer.append_data(self.getImageAtAlpha(0))

        writer.close()


class Triangle:
    vertices: 'np.ndarray'

    def __init__(self, vertices: np.ndarray):
        if (not isinstance(vertices,
                           np.ndarray)) or (vertices.dtype != 'float64'):
            raise ValueError('Contain unexpected types of data')

        if vertices.shape != (3, 2):
            raise ValueError('Does not meet the expected dimension')

        self.vertices = vertices

    def getPoints(self) -> np.ndarray:
        x_max = int(max(self.vertices[:, 0]) + 1)
        x_min = int(min(self.vertices[:, 0]))
        y_max = int(max(self.vertices[:, 1]) + 1)
        y_min = int(min(self.vertices[:, 1]))

        xx, yy = np.meshgrid(range(x_min, x_max), range(y_min, y_max))
        xy = np.dstack((xx, yy))
        xy_flat = xy.reshape((-1, 2))
        mpath = path.Path(self.vertices)
        mask_flat = mpath.contains_points(xy_flat)

        return xy_flat[mask_flat]

    def __str__(self):
        return str(self.vertices)


class ColorMorpher(Morpher):
    def __init__(self, leftImage: np.ndarray, leftTriangles,
                 rightImage: np.ndarray, rightTriangles):
        super().__init__(leftImage, leftTriangles, rightImage, rightTriangles)

    def getImageAtAlpha(self, alpha: float) -> np.ndarray:
        height, width, dim = self.leftImage.shape

        result = np.zeros((height, width, dim), dtype=np.uint8)

        xx = range(0, height, 1)
        yy = range(0, width, 1)

        alphaed_left_img = np.multiply(
            self.leftImage, (1 - alpha), casting='unsafe')
        alphaed_right_img = np.multiply(
            self.rightImage, alpha, casting='unsafe')

        f_left_new_0 = interpolate.RectBivariateSpline(
            xx, yy, alphaed_left_img[:, :, 0])
        f_left_new_1 = interpolate.RectBivariateSpline(
            xx, yy, alphaed_left_img[:, :, 1])
        f_left_new_2 = interpolate.RectBivariateSpline(
            xx, yy, alphaed_left_img[:, :, 2])

        f_right_new_0 = interpolate.RectBivariateSpline(
            xx, yy, alphaed_right_img[:, :, 0])
        f_right_new_1 = interpolate.RectBivariateSpline(
            xx, yy, alphaed_right_img[:, :, 1])
        f_right_new_2 = interpolate.RectBivariateSpline(
            xx, yy, alphaed_right_img[:, :, 2])

        for l_tri, r_tri in zip(self.leftTriangles, self.rightTriangles):
            # Find intermediate triangle
            temp = (1 - alpha) * l_tri.vertices + alpha * r_tri.vertices
            final_t = Triangle(temp)

            final_p_in_polygon = final_t.getPoints()

            # Left triange to final triangle
            inv_final_to_left_trnasf = calculate_transfor_and_inverse(
                l_tri.vertices, final_t.vertices)

            # Right triangle to final triangle
            inv_final_to_right_trnasf = calculate_transfor_and_inverse(
                r_tri.vertices, final_t.vertices)

            extra_ones = np.ones((final_p_in_polygon.shape[0], 1))

            transpose_inv_h_final_to_left = inv_final_to_left_trnasf.T[:, :2]
            transpose_inv_h_final_to_right = inv_final_to_right_trnasf.T[:, :2]
            points = np.hstack((final_p_in_polygon, extra_ones))

            # NOTE Find corresponding points of final triangle on left/right triangles
            np_left_p_in_polygon = points.dot(
                transpose_inv_h_final_to_left)[:, ::-1]
            np_right_p_in_polygon = points.dot(
                transpose_inv_h_final_to_right)[:, ::-1]

            left_p_color_0 = f_left_new_0.ev(np_left_p_in_polygon[:, 0],
                                             np_left_p_in_polygon[:, 1])
            right_p_color_0 = f_right_new_0.ev(np_right_p_in_polygon[:, 0],
                                               np_right_p_in_polygon[:, 1])

            left_p_color_1 = f_left_new_1.ev(np_left_p_in_polygon[:, 0],
                                             np_left_p_in_polygon[:, 1])
            right_p_color_1 = f_right_new_1.ev(np_right_p_in_polygon[:, 0],
                                               np_right_p_in_polygon[:, 1])

            left_p_color_2 = f_left_new_2.ev(np_left_p_in_polygon[:, 0],
                                             np_left_p_in_polygon[:, 1])
            right_p_color_2 = f_right_new_2.ev(np_right_p_in_polygon[:, 0],
                                               np_right_p_in_polygon[:, 1])

            layer_0 = np.add(left_p_color_0, right_p_color_0)
            layer_1 = np.add(left_p_color_1, right_p_color_1)
            layer_2 = np.add(left_p_color_2, right_p_color_2)

            result[final_p_in_polygon[:, 1],
                   final_p_in_polygon[:, 0]] = np.stack(
                       (layer_0, layer_1, layer_2), axis=1)

        return result


def loadTriangles(leftPointFilePath: str, rightPointFilePath: str
                  ) -> '(list[Triangle], list[Triangle])':

    l_narray = np.loadtxt(leftPointFilePath, dtype=np.float64)
    r_narray = np.loadtxt(rightPointFilePath, dtype=np.float64)

    l_tri_indecies = Delaunay(l_narray)

    to_triangle = lambda n_array: Triangle(n_array)

    left_triangles = list(map(to_triangle, l_narray[l_tri_indecies.simplices]))
    right_triangles = list(
        map(to_triangle, r_narray[l_tri_indecies.simplices]))

    return (left_triangles, right_triangles)

def generate_tris(left_arr: list, right_arr: list) -> '(list[Triangle], list[Triangle])':
    l_tri_indecies = Delaunay(left_arr)

    to_triangle = lambda arr: Triangle(arr)

    left_triangles = list(map(to_triangle, left_arr[l_tri_indecies.simplices]))
    right_triangles = list(map(to_triangle, right_arr[l_tri_indecies.simplices]))

    return (left_triangles, right_triangles)

def calculate_transfor_and_inverse(
        origin_tri: np.ndarray,
        target_tri: np.ndarray) -> (np.ndarray, np.ndarray):
    temp_a = np.vstack(
        (np.hstack((origin_tri[0].copy(), np.ones(1, dtype=np.float64),
                    np.zeros(3, dtype=np.float64))),
         np.hstack((np.zeros(3, dtype=np.float64), origin_tri[0].copy(),
                    np.ones(1, dtype=np.float64))),
         np.hstack((origin_tri[1].copy(), np.ones(1, dtype=np.float64),
                    np.zeros(3, dtype=np.float64))),
         np.hstack((np.zeros(3, dtype=np.float64), origin_tri[1].copy(),
                    np.ones(1, dtype=np.float64))),
         np.hstack((origin_tri[2].copy(), np.ones(1, dtype=np.float64),
                    np.zeros(3, dtype=np.float64))),
         np.hstack((np.zeros(3, dtype=np.float64), origin_tri[2].copy(),
                    np.ones(1, dtype=np.float64)))))

    temp_b = target_tri.reshape((6, 1))
    t = np.linalg.solve(temp_a, temp_b).reshape((2, 3))

    h = np.stack((*t, [0, 0, 1]))

    return np.linalg.inv(h)


if __name__ == "__main__":
    LEFT_POINT_FILE = "TestData/points.left.txt"
    RIGHT_POINT_FILE = "TestData/points.right.txt"
    left_tris, right_tris = loadTriangles(LEFT_POINT_FILE, RIGHT_POINT_FILE)

    # Gray-Scale Image Blending
    # LEFT_GRAY_IMAGE_FILE = "TestData/LeftGray.png"
    # RIGHT_GRAY_IMAGE_FILE = "TestData/RightGray.png"
    # left_image = imageio.imread(LEFT_GRAY_IMAGE_FILE)
    # right_image = imageio.imread(RIGHT_GRAY_IMAGE_FILE)
    # mor = Morpher(left_image, left_tris, right_image, right_tris)
    # imageio.imwrite("TestData/GrayBlending.png", mor.getImageAtAlpha(0.5))
    # mor.saveVideo("TestData/GrayBlending.mp4", 10, True, 5)

    # Colorful Image Blending
    LEFT_COLOR_IMAGE_FILE = "TestData/LeftColor.png"
    RIGHT_COLOR_IMAGE_FILE = "TestData/RightColor.png"
    left_image = imageio.imread(LEFT_COLOR_IMAGE_FILE)
    right_image = imageio.imread(RIGHT_COLOR_IMAGE_FILE)
    mor = ColorMorpher(left_image, left_tris, right_image, right_tris)
    imageio.imwrite("TestData/ColorfulBlending.png", mor.getImageAtAlpha(0.5))
    # Transition Sequence Generation
    # mor.saveVideo("TestData/ColorfulBlending.mp4", 16, True, 8)
