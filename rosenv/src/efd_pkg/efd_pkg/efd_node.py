from interfaces.srv import ComputePointEFD
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py.point_cloud2 import read_points_numpy
from geometry_msgs.msg import Polygon, Point32
import rclpy
from rclpy.node import Node

import numpy as np
import cv2
from cv_bridge import CvBridge
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

        self.br = CvBridge()

        self.n = 20  # order
        self.coefs = []
        self.a0 = 0
        self.c0 = 0
        self.T = 0
        
        self.obj_center = (0.0, 0.0)

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
        silhouette[obj_pc[:, 1], obj_pc[:, 0]] = 255
        self.obj_center = (np.mean(obj_pc[:, 0]), np.mean(obj_pc[:, 1]))
        contours, hierarchy = cv2.findContours(
            silhouette, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imwrite('silhouette.jpg', silhouette)

        # Fit EFD
        contour = np.squeeze(contours[0])
        self.coefs = pyefd.elliptic_fourier_descriptors(contour, order=self.n)
        self.a0, self.c0 = pyefd.calculate_dc_coefficients(
                np.squeeze(contours[0]))


    def compute_efd_callback(self, request, response):
        print('Service requested')
        t = request.t
        # Calculate x, y, Tx, Ty, Nx, Ny

        t = np.array([request.t])
        # Append extra dimension to enable element-wise broadcasted multiplication
        coeffs = self.coefs.reshape(self.coefs.shape[0], self.coefs.shape[1], 1)

        orders = coeffs.shape[0]
        orders = np.arange(1, orders + 1).reshape(-1, 1)
        omegas = 2 * orders * np.pi
        order_phases = omegas * t.reshape(1, -1)

        x = coeffs[:, 0] * np.cos(order_phases) + \
            coeffs[:, 1] * np.sin(order_phases)
        y = coeffs[:, 2] * np.cos(order_phases) + \
            coeffs[:, 3] * np.sin(order_phases)

        tx = -omegas * coeffs[:, 0] * np.sin(order_phases) + \
            omegas * coeffs[:, 1] * np.cos(order_phases)
        ty = -omegas * coeffs[:, 2] * np.sin(order_phases) + \
            omegas * coeffs[:, 3] * np.sin(order_phases)

        nx = -omegas*omegas * coeffs[:, 0] * np.cos(order_phases) - \
            omegas*omegas * coeffs[:, 1] * np.sin(order_phases)
        ny = -omegas*omegas * coeffs[:, 2] * np.cos(order_phases) - \
            omegas*omegas * coeffs[:, 3] * np.sin(order_phases)

        dnx = (omegas**3) * coeffs[:, 0] * np.sin(order_phases) - \
            (omegas**3) * coeffs[:, 1] * np.cos(order_phases)
        dny = (omegas**3) * coeffs[:, 2] * np.sin(order_phases) - \
            (omegas**3) * coeffs[:, 3] * np.cos(order_phases)

        ddnx = (omegas**4) * coeffs[:, 0] * np.cos(order_phases) + \
            (omegas**4) * coeffs[:, 1] * np.sin(order_phases)
        ddny = (omegas**4) * coeffs[:, 2] * np.cos(order_phases) + \
            (omegas**4) * coeffs[:, 3] * np.sin(order_phases)

        x = x.sum(axis=0)
        y = y.sum(axis=0)
        x = x + np.ones((1,)) * self.a0
        y = y + np.ones((1,)) * self.c0

        tx = tx.sum(axis=0)
        ty = ty.sum(axis=0)

        nx = nx.sum(axis=0)[0]
        ny = ny.sum(axis=0)[0]

        dnx = dnx.sum(axis=0)[0]
        dny = dny.sum(axis=0)[0]
        ddnx = ddnx.sum(axis=0)[0]
        ddny = ddny.sum(axis=0)[0]

        response.obj_x = self.obj_center[0]
        response.obj_y = self.obj_center[1]
        response.x = x[0]
        response.y = y[0]
        response.tx = tx[0]
        response.ty = ty[0]
        response.nx = nx
        response.ny = ny
        response.c = math.sqrt(nx**2 + ny**2)
        response.dc = (nx*dnx + ny*dny)/response.c
        response.ddc = ((dnx**2 + nx*ddnx + dny**2 + ny*ddny)*response.c - (nx*dnx + ny*dny)*response.dc)/(response.c**2)

        # Visualize
        vis_contour = pyefd.reconstruct_contour(
                self.coefs,
                locus=(self.a0, self.c0)).astype(int)
        vis_img = np.zeros(self.im_sz, dtype=np.ubyte)
        vis_img[vis_contour[:, 1], vis_contour[:, 0]] = 255
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
        response.img = self.br.cv2_to_imgmsg(vis_img, encoding='bgr8')

        return response


def main(args=None):
    rclpy.init(args=args)

    efd_service = EFDService()

    rclpy.spin(efd_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
