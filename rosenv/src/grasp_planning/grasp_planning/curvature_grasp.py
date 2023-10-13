import time
import math

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
            response = self.send_request(1.5)
            print(f'Response:\tx = {response.x}\ty = {response.y}\n' +
                  f'Tx = {response.tx}\tTy = {response.ty}\n' +
                  f'Nx = {response.nx}\tNy = {response.ny}\n')
            img = self.br.imgmsg_to_cv2(response.img,
                                        desired_encoding='passthrough')
            img = cv2.circle(img, (int(response.x), int(response.y)),
                             3, (0, 0, 255), -1)
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
