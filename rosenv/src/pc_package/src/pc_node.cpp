#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

class PointCloudProcessingNode : public rclcpp::Node
{
public:
    PointCloudProcessingNode() : Node("pointcloud_processing_node")
    {
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("downsampled_points", 10);
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/realsense/points", 10, std::bind(&PointCloudProcessingNode::listener_callback, this, std::placeholders::_1));
    }

private:
    void listener_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        std::cout << "got pcd" << std::endl;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        pcl::ExtractIndices<pcl::PointXYZ> extract;

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(1000);
        seg.setDistanceThreshold(0.001);  // Adjust this value as needed
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);


        // sor.setInputCloud(cloud);
        // sor.setLeafSize(0.1f, 0.1f, 0.1f);
        // sor.filter(*cloud_filtered);
      
        // Filter out the table
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*cloud_filtered);

        if (inliers->indices.size() == 0)
        {
            std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
        }

        // std::cerr << "Model coefficients: " << coefficients->values[0] << " " 
        //           << coefficients->values[1] << " "
        //           << coefficients->values[2] << " " 
        //           << coefficients->values[3] << std::endl;

        // std::cerr << "Model inliers: " << inliers->indices.size () << std::endl;
        // for (const auto& idx: inliers->indices)
        //     std::cerr << idx << "    " << cloud->points[idx].x << " "
        //                        << cloud->points[idx].y << " "
        //                        << cloud->points[idx].z << std::endl;

        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*cloud_filtered, output);


        publisher_->publish(output);
    }

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudProcessingNode>());
    rclcpp::shutdown();
    return 0;
}
