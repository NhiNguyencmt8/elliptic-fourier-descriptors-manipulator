import math
import numpy as np

class GraspPlanning:

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
                for k in range(len(contact_locations)):
                    Gk = GraspPlanning.grasp_matrix(object_center, contact_locations[k])
                    for x in range(len(contact_locations)):
                        #Calculate new grasp matrix
                        Gx = GraspPlanning.grasp_matrix(object_center, contact_locations[x])
                        #Append to the large matrix
                        G = np.eye(3,dtype=int) # Reset
                        G = np.append(G,Gi,axis=0)
                        G = np.append(G,Gj,axis=0)
                        G = np.append(G,Gk,axis=0)
                        G = np.append(G,Gx,axis=0)
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
                            final_locations[2] = contact_locations[k]
                            final_locations[3] = contact_locations[x]

        
        return min_Q_MSV, final_locations
    
    def quality_shape_of_grasp_polygon(object_center, contact_locations):

        G = np.eye(3,dtype=int)
        max_Q_SGP = 0
        
        final_locations = np.zeros(shape=(3, 2))
        #Find a combination of 4 contact points out of the set
        #TODO: Find a better way for this -> there's iteratool in python
        for i in range(len(contact_locations)):
            for j in range(len(contact_locations)):
                for k in range(len(contact_locations)):
                    for x in range(len(contact_locations)):
                        #Calculating angles
                        theta = np.zeros(shape=(4, 1))
                        theta[0] = math.atan2(contact_locations[i],contact_locations[j])
                        theta[1] = math.atan2(contact_locations[j],contact_locations[k])
                        theta[2] = math.atan2(contact_locations[k],contact_locations[x])
                        theta[3] = math.atan2(contact_locations[x],contact_locations[i])

                        #Calculate quality metrics
                        n = 4 # 4 points
                        theta_ = 180*(n-2)/n
                        theta_max = (n - 2)* (180 - theta_) + 2*theta_
                        Q_SGP = (1/theta_max)*(abs(theta[0] - theta[1]) + abs(theta[1] - theta[2]) + abs(theta[2] - theta[3]) + abs(theta[3] - theta[0]))

                        #Comparisionabs(theta[0] - theta[1])
                        if (Q_SGP > max_Q_SGP):
                            #Swap
                            max_Q_SGP = Q_SGP
                            #Record locations
                            final_locations[0] = contact_locations[i]
                            final_locations[1] = contact_locations[j]
                            final_locations[2] = contact_locations[k]
                            final_locations[3] = contact_locations[x]

        
        return max_Q_SGP, final_locations
    
    
    
                        


                        

            
            


        
