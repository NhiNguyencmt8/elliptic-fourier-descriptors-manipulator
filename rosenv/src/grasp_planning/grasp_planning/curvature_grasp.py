import time
import numpy as np

from interfaces.srv import ComputePointEFD
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class CurvatureGrasp(Node):

    def __init__(self):
        super().__init__('curvature_grasp')
        self.cli = self.create_client(ComputePointEFD, 'compute_efd')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.vis_publisher = self.create_publisher(
                Image, 'curvature_grasp_vis', 10)
        self.req = ComputePointEFD.Request()

        self.br = CvBridge()

    def send_request(self, t):
        self.req.t = t
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    def run(self):
        while rclpy.ok():
            # Search for maximum curvature grasp
            # A grasp will contain of 2 contact points with phase difference of pi
            step = 0.01
            max_curvature = -1.0
            best_grasp = (0, 0, 0, 0)  # (x1, y1, x2, y2)
            best_img = None
            for t in np.arange(0.0, 0.5, step):
                t_opposite = t + 0.5
                resp_t = self.send_request(t)
                resp_t_opposite = self.send_request(t_opposite)
                curvature = resp_t.c + resp_t_opposite.c
                if curvature > max_curvature:
                    max_curvature = curvature
                    best_grasp = (int(resp_t.x), int(resp_t.y),
                                  int(resp_t_opposite.x), int(resp_t_opposite.y))
                    best_img = resp_t.img
                # print(f'Response:\tx = {response.x}\ty = {response.y}\n' +
                #       f'Tx = {response.tx}\tTy = {response.ty}\n' +
                #       f'Nx = {response.nx}\tNy = {response.ny}\n' +
                #       f'obj_x = {response.obj_x}\tobj_y = {response.obj_y}\n' +
                #       f'c = {response.c}\tdc = {response.dc}\tddc = {response.ddc}')

            # Visualize the most stable grasp
            img = self.br.imgmsg_to_cv2(best_img,
                                        desired_encoding='passthrough')
            img = cv2.circle(img, (best_grasp[0], best_grasp[1]),
                             3, (0, 0, 255), -1)
            img = cv2.circle(img, (best_grasp[2], best_grasp[3]),
                             3, (0, 255, 0), -1)
            img_msg = self.br.cv2_to_imgmsg(img)
            self.vis_publisher.publish(img_msg)

            time.sleep(0.1)


def main():
    rclpy.init()

    curvature_grasp = CurvatureGrasp()

    curvature_grasp.run()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
