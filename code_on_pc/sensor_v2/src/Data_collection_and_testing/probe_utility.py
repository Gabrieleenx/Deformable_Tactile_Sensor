import rospy
import random
import numpy as np
from geometry_msgs.msg import Pose
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix, quaternion_from_matrix
from sensor_v2.msg import PoseStampedWithLimits
import copy
from scipy.spatial.transform import Rotation as R
import os
import sys
import subprocess
import math


def pose_to_hmat(p):
    T = quaternion_matrix([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
    T[0:3, 3] = [p.position.x, p.position.y, p.position.z]
    return T

def hmat_to_pose(T):
    p = Pose()
    p.position.x, p.position.y, p.position.z = T[0:3, 3]
    q = quaternion_from_matrix(T)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
    return p

def normalize_angle(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi  # [-pi, pi]

def angle_diff(a, b):
    """Signed difference a-b in (-π, π]."""
    d = (a - b + np.pi) % (2*np.pi) - np.pi
    return d

def publish_pose(pub, pose, time, f_max = 2.5, t_max = 0.1):
    msg = PoseStampedWithLimits()
    msg.pose.header.seq = 0
    msg.pose.header.stamp = rospy.Time(0)
    msg.pose.header.frame_id = ""

    msg.pose.pose = pose

    msg.enable_lead_control = False
    msg.max_force = f_max
    msg.max_torque = t_max
    msg.trajectory_time = time

    pub.publish(msg)
    rospy.loginfo("Going to pose and message published!")


def interpolate_yaw_path(start_pose, target_pose, reference_yaw, step_size=0.05):
    # Extract start yaw
    q_start = start_pose.orientation
    _, _, yaw_start = euler_from_quaternion([q_start.x, q_start.y, q_start.z, q_start.w])
    yaw_start = normalize_angle(yaw_start)

    # Extract target yaw
    q_target = target_pose.orientation
    _, _, yaw_target = euler_from_quaternion([q_target.x, q_target.y, q_target.z, q_target.w])
    yaw_target = normalize_angle(yaw_target)

    # Normalize reference
    ref = normalize_angle(reference_yaw)


    # Compute signed difference from start to target
    print(yaw_target, yaw_start)
    delta = yaw_target - yaw_start

    # Interpolate
    
    n_steps = max(math.ceil(abs(delta) / step_size), 1)

    
    yaw_vals = [normalize_angle(yaw_start + delta * i / n_steps) for i in range(n_steps + 1)]
    print("delta", delta, "step_size", step_size, "n_steps", yaw_vals)
    poses = []
    for yaw in yaw_vals:
        pose = Pose()
        pose.position = target_pose.position  # keep position
        q = quaternion_from_euler(np.pi, 0, yaw)  # z down
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
        poses.append(pose)

    return poses


def apply_delta(pose_, dx, dy, dz, dtx, dty, dtz):
    # Apply translations
    pose_.position.x += dx
    pose_.position.y += dy
    pose_.position.z += dz

    # Convert the existing quaternion to scipy Rotation
    q_orig = [pose_.orientation.x,
            pose_.orientation.y,
            pose_.orientation.z,
            pose_.orientation.w]
    R_orig = R.from_quat(q_orig)

    # Build delta rotation from euler angles (roll, pitch, yaw)
    R_delta = R.from_euler('xyz', [dtx, dty, dtz])

    # Compose rotations
    R_new = R_delta * R_orig  # apply delta first, then original

    # Convert back to quaternion
    q_new = R_new.as_quat()

    pose_.orientation.x = q_new[0]
    pose_.orientation.y = q_new[1]
    pose_.orientation.z = q_new[2]
    pose_.orientation.w = q_new[3]
    return pose_


class TargetShape:
    def __init__(self, name, x_min, x_max, y_min, y_max, max_z_delta, max_xy_delta, max_tz_delta, max_txy_delta, f_max, t_max):
        self.name = name
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.max_z_delta = max_z_delta
        self.max_xy_delta = max_xy_delta
        self.max_txy_delta = max_txy_delta
        self.max_tz_delta = max_tz_delta
        self.f_max = f_max
        self.t_max = t_max

def pose_in_base(object_pose, target_pose_in_object):
    pose_target_in_base = hmat_to_pose(pose_to_hmat(object_pose) @ pose_to_hmat(target_pose_in_object))
    return pose_target_in_base
    

class Prope:
    def __init__(self, z_offset):
        #shapes to include
        # nr probes per object 
        self.rot_offset = 0.0
        self.z_offset = z_offset 
        self.traj_time = 2.0
        self.epsilon_ = 0.1

    def gen_pos(self, target_shape, object_pose):
        # Define ranges
        ranges = [(target_shape.x_min, target_shape.x_max), 
                  (target_shape.y_min, target_shape.y_max), 
                  (-np.pi+self.epsilon_ + self.rot_offset, np.pi-self.epsilon_ + self.rot_offset)]

        # Generate random values
        x = random.uniform(*ranges[0])
        y = random.uniform(*ranges[1])
        yaw = random.uniform(*ranges[2])

        # Build Pose
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = self.z_offset

        # Orientation:
        #  - yaw around Z
        #  - plus 180° around X so tool Z points downward
        q = quaternion_from_euler(np.pi, 0, yaw)  
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q

        pose_target_in_base = hmat_to_pose(pose_to_hmat(object_pose) @ pose_to_hmat(pose))


        return pose_target_in_base
    
    def explore_torque_and_tangential(self, current_pose, base_height, height_steps, force_list, pub, dxy=0.004, dtz=0.3, nr_tangential=5):
        pose_reset = copy.deepcopy(current_pose)
        i = 0
        for height in height_steps:
            force = force_list[i]
            i += 0

            pose_reset.position.z = height + base_height
            if rospy.is_shutdown(): 
                return
            # go to height 
            print("height: ", height)
            pose_ = copy.deepcopy(pose_reset)
            publish_pose(pub, pose_, 2.0, f_max = force_list[i], t_max = 0.5)
            rospy.sleep(2.5)
            # rotate 
            if rospy.is_shutdown(): 
                return
            
            pose_ = apply_delta(pose_, dx=0, dy=0, dz=0, dtx=0, dty=0, dtz=dtz)
            publish_pose(pub, pose_, 2.0, f_max = force, t_max = 0.5)
            rospy.sleep(2.5)
            if rospy.is_shutdown(): 
                return

            pose_ = copy.deepcopy(pose_reset)
            publish_pose(pub, pose_, 2.0, f_max = force, t_max = 0.5)
            rospy.sleep(2.5)
            if rospy.is_shutdown(): 
                return

            pose_ = apply_delta(pose_, dx=0, dy=0, dz=0, dtx=0, dty=0, dtz=-dtz)
            publish_pose(pub, pose_, 2.0, f_max = force, t_max = 0.5)
            rospy.sleep(2.5)
            if rospy.is_shutdown(): 
                return
        
            pose_ = copy.deepcopy(pose_reset)
            publish_pose(pub, pose_, 2.0, f_max = force, t_max = 0.5)
            rospy.sleep(2.5)
            if rospy.is_shutdown(): 
                return

            # tangential 

            for i in range(nr_tangential):
                if rospy.is_shutdown(): 
                    return
                angle = random.uniform(-np.pi, np.pi)
                dx = np.sin(angle)*dxy
                dy = np.cos(angle)*dxy

                pose_ = apply_delta(pose_, dx=dx, dy=dy, dz=0, dtx=0, dty=0, dtz=0)
                publish_pose(pub, pose_, 2.0, f_max = force, t_max = 0.5)
                rospy.sleep(2.5)

                pose_ = copy.deepcopy(pose_reset)
                publish_pose(pub, pose_, 2.0, f_max = force, t_max = 0.5)
                rospy.sleep(2.5)
            


    def gen_delta_probe(self, target_shape, pose_in_contact, pub):
        dx = random.uniform(-target_shape.max_xy_delta, target_shape.max_xy_delta)
        dy = random.uniform(-target_shape.max_xy_delta, target_shape.max_xy_delta)
        dz = random.uniform(-target_shape.max_z_delta, 0)
        dtz = random.uniform(-target_shape.max_tz_delta, target_shape.max_tz_delta)
        dty = random.uniform(-target_shape.max_txy_delta, target_shape.max_txy_delta)
        dtx = random.uniform(-target_shape.max_txy_delta, target_shape.max_txy_delta)
        pose_ = copy.deepcopy(pose_in_contact)
        # Apply translations
        pose_.position.x += dx
        pose_.position.y += dy
        pose_.position.z += dz

        # Convert the existing quaternion to scipy Rotation
        q_orig = [pose_.orientation.x,
                pose_.orientation.y,
                pose_.orientation.z,
                pose_.orientation.w]
        R_orig = R.from_quat(q_orig)

        # Build delta rotation from euler angles (roll, pitch, yaw)
        R_delta = R.from_euler('xyz', [dtx, dty, dtz])

        # Compose rotations
        R_new = R_delta * R_orig  # apply delta first, then original

        # Convert back to quaternion
        q_new = R_new.as_quat()

        pose_.orientation.x = q_new[0]
        pose_.orientation.y = q_new[1]
        pose_.orientation.z = q_new[2]
        pose_.orientation.w = q_new[3]

        publish_pose(pub, pose_, 5.0, f_max = target_shape.f_max, t_max = target_shape.t_max)
        rospy.sleep(5.5)



    def in_contact_force_servoing(self, target_ft, target_pose):
        # just send it to the controller
        pass

    def go_to_height(self, target_shape, target_pose, object_pose, height, pub, time_):
        pose_ = copy.deepcopy(target_pose)  
        pose_.position.z = object_pose.position.z + height
        publish_pose(pub, pose_, time_, f_max=target_shape.f_max, t_max=target_shape.t_max)
        return pose_



    def move_to_target(self, target_pose, pose_msg_, pub):
        # Move to above target, make sure orientation is not violated by controller. 
        # move down to target until contact has been established and then move to keep contact to minimum. 
        print(pose_msg_)
        waypoints = interpolate_yaw_path(pose_msg_, target_pose, self.rot_offset, step_size=2.5)
        for p in waypoints:
            # Send p to robot
            publish_pose(pub, p, self.traj_time)
            rospy.sleep(self.traj_time+0.3)
            


def start_rosbag(topics, bag_name="recording", directory="."):
    """
    Start recording a rosbag with the specified topics and save it to the specified directory.
    
    :param topics: List of topics to record.
    :param bag_name: Name of the bag file.
    :param directory: Directory where the bag file will be saved.
    :return: The subprocess.Popen object for the rosbag recording process.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    bag_path = os.path.join(directory, bag_name)
    command = ['rosbag', 'record', '-O', bag_path] + topics
    process = subprocess.Popen(command)
    rospy.sleep(2)
    return process

def stop_rosbag(process):
    """
    Stop the rosbag recording process.
    
    :param process: The subprocess.Popen object for the rosbag recording process.
    """
    process.terminate()
    process.wait()




