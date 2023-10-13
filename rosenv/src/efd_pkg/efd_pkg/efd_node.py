from interfaces.srv import ComputePointEFD
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy
from geometry_msgs.msg import Polygon, Point32
import rclpy
from rclpy.node import Node

import numpy as np
import cv2
import math
import pyefd


class EFDService(Node):

    def __init__(self):
        super().__init__('efd_service')
        # Subscribes to the point cloud topic
        # The topic contains segmented object point cloud
        self.pc_subscription = self.create_subscription(
            PointCloud2,
            'downsampled_points',
            self.pc_callback,
            10)
        self.pc_subscription  # prevent unused variable warning

        # Service that given `t`, computes `x(t)`, `y(t)`, `Tx(t)`, `Ty(t)`,
        # `Nx(t)`, `Ny(t)`
        self.srv = self.create_service(ComputePointEFD, 'compute_efd',
                                       self.compute_efd_callback)
        self.n = 20  # order
        self.coefs = []
        self.a0 = 0
        self.c0 = 0
        self.T = 0

        # "Image" size 1000x1000
        self.im_sz = (1000, 1000)

        # Visualization publisher
        self.vis_publisher = self.create_publisher(Polygon, 'efd/vis', 10)

    def pc_callback(self, msg):
        # Take x and y coordinates
        obj_pc = read_points_numpy(msg)[:, [0, 1]]

        # Shifting and scaling [-1, 1] to [0, 1000]
        obj_pc = 500*obj_pc + 500
        obj_pc = obj_pc.astype(int)

        # Project points to an "image"
        silhouette = np.zeros(self.im_sz, dtype=np.ubyte)
        silhouette[obj_pc[:, 0], obj_pc[:, 1]] = 255
        contours, hierarchy = cv2.findContours(
            silhouette, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imwrite('silhouette.jpg', silhouette)

        # Fit EFD
        contour = np.squeeze(contours[0])
        self.coefs = pyefd.elliptic_fourier_descriptors(contour, order=self.n)
        self.a0, self.c0 = pyefd.calculate_dc_coefficients(
                np.squeeze(contours[0]))
        dxy = np.diff(contour, axis=0)
        dt = np.sqrt((dxy ** 2).sum(axis=1))
        tmp = np.concatenate([([0.0]), np.cumsum(dt)])
        self.T = tmp[-1]

        # # Visualize
        # vis_contour = pyefd.reconstruct_contour(
        #         self.coefs,
        #         locus=(self.a0, self.c0)).astype(int)
        # vis_img = np.zeros(self.im_sz, dtype=np.ubyte)
        # vis_img[vis_contour[:, 1], vis_contour[:, 0]] = 255
        #
        # # Convert back to original unit
        # vis_contour = (vis_contour.astype(float) - 500.0) / 500.0
        # vis_msg = Polygon()
        # for p in vis_contour:
        #     pt = Point32()
        #     pt.x = p[0]
        #     pt.y = p[1]
        #     pt.z = 0.0
        #     vis_msg.points.append(pt)
        # self.vis_publisher.publish(vis_msg)

    def compute_efd_callback(self, request, response):
        t = request.t
        # Calculate T
        omega = 2*self.n*math.pi/self.T
        c = math.cos(omega*t)
        s = math.sin(omega*t)
        cs = np.array([[c], [s]])
        sc = np.array([[-s], [c]])
        # Calculate x, y, Tx, Ty, Nx, Ny
        response.x = self.a0 + np.sum(self.coefs[:, [0, 1]] @ cs)
        response.y = self.c0 + np.sum(self.coefs[:, [2, 3]] @ cs)
        response.tx = omega * np.sum(self.coefs[:, [0, 1]] @ sc)
        response.ty = omega * np.sum(self.coefs[:, [2, 3]] @ sc)
        response.nx = -omega * omega * np.sum(self.coefs[:, [0, 1]] @ cs)
        response.ny = -omega * omega * np.sum(self.coefs[:, [2, 3]] @ cs)

        return response


def main(args=None):
    rclpy.init(args=args)

    efd_service = EFDService()

    rclpy.spin(efd_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
