#include <ros/ros.h>
#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chain.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <memory>

class CartesianVelocityConverter {
public:
    CartesianVelocityConverter() {
        // Load robot model
        std::string robot_desc_string;
        nh_.param("robot_description", robot_desc_string, std::string());
        if (!kdl_parser::treeFromString(robot_desc_string, kdl_tree_)) {
            ROS_ERROR("Failed to construct KDL tree");
            return;
        }

        // KDL chain from base to tool0
        if (!kdl_tree_.getChain("base_link", "tool0", kdl_chain_)) {
            ROS_ERROR("Failed to get KDL chain from base_link to tool0");
            return;
        }

        jac_solver_.reset(new KDL::ChainJntToJacSolver(kdl_chain_));

        // UR3e joint names in order
        ur_joint_names_ = {"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                           "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};

        // Subscribers & publisher
        twist_sub_ = nh_.subscribe("/cartesian_velocity_command", 1, &CartesianVelocityConverter::twistCallback, this);
        joint_state_sub_ = nh_.subscribe("/joint_states", 1, &CartesianVelocityConverter::jointStateCallback, this);
        joint_velocity_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("/ur3_joint_velocity_controller/command", 1);
    }

private:
    void twistCallback(const geometry_msgs::Twist::ConstPtr& msg) {
        twist_command_ = *msg;
        has_twist_command_ = true;
    }

    void jointStateCallback(const sensor_msgs::JointState::ConstPtr& msg) {
        current_joint_states_ = *msg;
        if (!has_twist_command_) return;

        // Fill KDL joint array from current joint states
        KDL::JntArray q_in(kdl_chain_.getNrOfJoints());
        for (size_t i = 0; i < ur_joint_names_.size(); ++i) {
            auto it = std::find(current_joint_states_.name.begin(), current_joint_states_.name.end(), ur_joint_names_[i]);
            if (it != current_joint_states_.name.end()) {
                size_t index = std::distance(current_joint_states_.name.begin(), it);
                q_in(i) = current_joint_states_.position[index];
            } else {
                ROS_WARN("Joint %s not found in /joint_states", ur_joint_names_[i].c_str());
                q_in(i) = 0.0;
            }
        }

        // Compute Jacobian
        KDL::Jacobian jacobian(kdl_chain_.getNrOfJoints());
        if (jac_solver_->JntToJac(q_in, jacobian) < 0) {
            ROS_ERROR("Failed to compute Jacobian");
            return;
        }

        // Convert Jacobian to Eigen
        Eigen::MatrixXd jac_eigen = jacobian.data;
        Eigen::MatrixXd jac_pinv = jac_eigen.completeOrthogonalDecomposition().pseudoInverse();

        // Convert Twist to Eigen vector
        Eigen::Matrix<double, 6, 1> twist_eigen;
        twist_eigen << twist_command_.linear.x, twist_command_.linear.y, twist_command_.linear.z,
                       twist_command_.angular.x, twist_command_.angular.y, twist_command_.angular.z;

        // Calculate joint velocities
        Eigen::VectorXd q_dot_eigen = jac_pinv * twist_eigen;

        // Publish joint velocities
        std_msgs::Float64MultiArray joint_velocities;
        joint_velocities.data.resize(ur_joint_names_.size());
        for (size_t i = 0; i < ur_joint_names_.size(); ++i) {
            joint_velocities.data[i] = q_dot_eigen(i);
        }
        joint_velocity_pub_.publish(joint_velocities);
    }

    ros::NodeHandle nh_;
    ros::Subscriber twist_sub_;
    ros::Subscriber joint_state_sub_;
    ros::Publisher joint_velocity_pub_;

    KDL::Tree kdl_tree_;
    KDL::Chain kdl_chain_;
    std::unique_ptr<KDL::ChainJntToJacSolver> jac_solver_;

    geometry_msgs::Twist twist_command_;
    sensor_msgs::JointState current_joint_states_;
    bool has_twist_command_ = false;

    std::vector<std::string> ur_joint_names_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "cartesian_velocity_converter");
    CartesianVelocityConverter converter;
    ros::spin();
    return 0;
}
