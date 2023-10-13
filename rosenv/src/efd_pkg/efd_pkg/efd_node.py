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
            'pcd',
            self.pc_callback,
            10)
        self.pc_subscription  # prevent unused variable warning

        # Service that given `t`, computes `x(t)`, `y(t)`, `Tx(t)`, `Ty(t)`,
        # `Nx(t)`, `Ny(t)`
        self.srv = self.create_service(ComputePointEFD, 'compute_efd',
                                       self.compute_efd_callback)
        self.n = 10  # order
        self.coefs = []
        self.a0 = 0
        self.c0 = 0

        # Visualization publisher
        self.vis_publisher = self.create_publisher(Polygon, 'efd/vis', 10)

    def pc_callback(self, msg):
        # Take x and y coordinates
        obj_pc = read_points_numpy(msg)[:, [0, 1]]
        print(obj_pc)
        # max_dim = np.max(obj_pc, axis=0)
        # print(max_dim)

        # Project points to the image
        silhouette = np.zeros(max_dim[0], max_dim[1])
        silhouette[obj_pc.astype(int)] = 1
        contours, hierarchy = cv2.findContours(
            silhouette, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Fit EFD
        contour = np.squeeze(contours[0])
        self.coefs = pyefd.elliptic_fourier_descriptors(contour, order=self.n)
        self.a0, self.c0 = pyefd.calculate_dc_coefficients(
                np.squeeze(contours[0]))
        dxy = np.diff(contour, axis=0)
        dt = np.sqrt((dxy ** 2).sum(axis=1))
        tmp = np.concatenate([([0.0]), np.cumsum(dt)])
        self.T = tmp[-1]

        # Visualize
        vis_contour = pyefd.reconstruct_contour(self.coefs)
        vis_msg = Polygon()
        vis_msg.points = [Point32(p[0], p[1], 0) for p in vis_contour]
        self.vis_publisher.publish(vis_msg)


    def compute_efd_callback(self, request, response):
        t = request.t
        # Calculate T
        c = math.cos(2*self.n*math.pi*t/self.T)
        s = math.sin(2*self.n*math.pi*t/self.T)
        cs = np.array([[c], [s]])
        # Calculate x, y, Tx, Ty, Nx, Ny
        response.x = self.a0 + np.sum(self.coefs[:, [0, 1]] @ cs)
        response.y = self.c0 + np.sum(self.coefs[:, [2, 3]] @ cs)

        return response


def main(args=None):
    rclpy.init(args=args)

    efd_service = EFDService()

    rclpy.spin(efd_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
