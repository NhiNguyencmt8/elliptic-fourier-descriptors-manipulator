import math
import numpy as np
import rclpy # Python library for ROS 2
import time
import interfaces
import cv2
from rclpy.node import Node # Handles the creation of nodes
from std_msgs.msg import String
from interfaces.srv import ComputePointEFD
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge

import sys

timer_period = 1  # seconds = 100Hz
   
# class PointVisualize:
#     def __init__(self) :
#         self.img_subscription = self.create_subscription(
#         Image, 
#         '/camera1/image_raw', 
#         self.img_listener_callback, 
#         10)
#         self.img_subscription # prevent unused variable warning
#                 # Create the publisher. This publisher will publish an Image
#         # to the video_frames topic. The queue size is 10 messages.
#         self.img_publisher_ = self.create_publisher(Image, 'output_image', 10)
#         # Used to convert between ROS and OpenCV images
#         self.br = CvBridge()
     
#     def img_listener_callback(self, data):
#         # Display the message on the console
#         self.get_logger().info('Receiving video frame')
    
#         # Convert ROS Image message to OpenCV image
#         current_frame = self.br.imgmsg_to_cv2(data)

#         # Find the contact locations
#         #metric, final_locations = self.quality_min_singular(self.object_center, self.contact_locations)
        
#         #Drawing the locations on the image
#         self.set_of_points = np.zeros(shape=(1, 7))
#         # Publish the image.
#         # The 'cv2_to_imgmsg' method converts an OpenCV
#         # image to a ROS 2 image message
#         self.publisher_.publish(self.br.cv2_to_imgmsg(current_frame, encoding="bgr8")) 
        

class GraspPlanning(Node):

    def __init__(self):
        # Initiate the Node class's constructor and give it a name
        super().__init__('grasp_planning')
        self.object_subscription = self.cli = self.create_client(ComputePointEFD, 'compute_efd')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = ComputePointEFD.Request()
        self.br = CvBridge()
        self.img_publisher_ = self.create_publisher(Image, 'output_image', 10)
        self.set_of_points = []

    def spin(self):
        while rclpy.ok():
            for i in range(0,100,1): 
                response = self.send_request(i/100)
                self.set_of_points.append([response.x, response.y, response.tx, response.ty, response.nx, response.ny])
            

            img = self.br.imgmsg_to_cv2(response.img,
                                        desired_encoding='passthrough')
            print(self.set_of_points)
            img = cv2.circle(img, (int(response.x), int(response.y)),
                             3, (0, 0, 255), -1)
            #cv2.circle(current_frame,(self.set_of_points[0][0],self.set_of_points[0][1]), 63, (0,255,0), -1)
            self.img_publisher_.publish(self.br.cv2_to_imgmsg(img, encoding="bgr8")) 
            print("Done one loop \n")
            time.sleep(timer_period)

    

    def send_request(self, t):
        #print("Sending request t = %f", t)
        self.req.t = t
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        self.response = self.future.result()
        # self.get_logger().info(
        # 'Result from EFD: x %f y %f Tx %f Ty %f Nx %f Nx %f' %
        # (self.response.x, self.response.y, self.response.tx, self.response.ty, self.response.nx, self.response.ny))
        #self.set_of_points[index] = np.array([self.response.x, self.response.y, self.response.tx, self.response.ty, self.response.nx, self.response.ny])
       
        return self.response

    @staticmethod
    def grasp_matrix(object_center, contact_location):
        """
        """
        object_angle = math.atan2(object_center[2],object_center[1])
        theta = object_angle; #Since we are rotating from the base frame to the contact point frame
        R = np.array([math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)])
        ax = object_center[1] - contact_location[1]
        ay = object_center[2] - contact_location[2]
        G = np.array([[np.matmul(R,np.eye(2, dtype=int)) , np.matmul(R,np.array([[ax],[ay]]))], [0, 0, 1]])
        return G
    
    def quality_min_singular(object_center, contact_locations):

        G = np.eye(3,dtype=int)
        min_Q_MSV = 1000000000
        
        final_locations = np.zeros(shape=(3, 2))
        #Find a combination of 4 contact points out of the set
        #TODO: Find a better way for this -> there's iteratool in python
        for i in range(len(contact_locations)):
            Gi = GraspPlanning.grasp_matrix(object_center, contact_locations[i])
            for j in range(len(contact_locations)):
                Gj = GraspPlanning.grasp_matrix(object_center, contact_locations[j])
                #Append to the large matrix
                G = np.eye(3,dtype=int) # R
                G = np.append(G,Gi,axis=0)
                G = np.append(G,Gj,axis=0)
                #Grasp metrics calculation
                P, D, Q = np.linalg.svd(np.transpose(G), full_matrices=False)
                G_svg = np.matmul(np.matmul(P, np.diag(D)), Q)
                Q_MSV = np.min(G_svg)
                #Comparision
                if (Q_MSV < min_Q_MSV):
                    #Swap
                    min_Q_MSV = Q_MSV
                    #Record locations
                    final_locations[0] = contact_locations[i]
                    final_locations[1] = contact_locations[j]
        return min_Q_MSV, final_locations
    

    

def main():
  
  # Initialize the rclpy library
  rclpy.init()
  
  # Create the node
  grasp = GraspPlanning()
  print("Grasp planning node created")
  grasp.spin()
  
if __name__ == '__main__':
  main()    
    
                        


                        

            
            


        
