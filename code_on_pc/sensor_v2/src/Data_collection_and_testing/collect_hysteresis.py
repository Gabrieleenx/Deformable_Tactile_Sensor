#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped, Pose
from visualization_msgs.msg import Marker
from sensor_v2.msg import PoseStampedWithLimits
import numpy as np
from scipy.spatial.transform import Rotation as R
from std_srvs.srv import Empty
import tf2_ros
import geometry_msgs.msg
import probe_utility
from tf.transformations import quaternion_from_euler
import copy
from std_srvs.srv import Trigger

static_broadcaster = tf2_ros.StaticTransformBroadcaster()
recorded_poses = []

# === EDIT THIS: Object feature points in object frame ===
object_points = np.array([
    [0.05, -0.075, 0.014],   # feature 1
    [-0.05, -0.075, 0.014],  # feature 2
    [0.0, 0.075, 0.014]      # feature 3
])

def get_frame_pose(frame_id="tactile_calibration", ref_frame="base_link", tf_buffer=None):
    try:
        trans = tf_buffer.lookup_transform(ref_frame,
                                           frame_id,
                                           rospy.Time(0),
                                           rospy.Duration(1.0))  # timeout 1s
        pose = PoseStamped()
        pose.header = trans.header
        pose.pose.position.x = trans.transform.translation.x
        pose.pose.position.y = trans.transform.translation.y
        pose.pose.position.z = trans.transform.translation.z
        pose.pose.orientation = trans.transform.rotation
        return pose
    except (tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as e:
        rospy.logwarn(f"TF lookup failed: {e}")
        return None

def bias_netft(sensor_nr=1):
    service_name = f"/netft_bias{sensor_nr}"

    rospy.loginfo(f"Waiting for service {service_name}...")
    rospy.wait_for_service(service_name)

    try:
        bias_srv = rospy.ServiceProxy(service_name, Empty)
        bias_srv()  # call the service
        rospy.loginfo("NetFT sensor biased successfully!")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")


def bias_tactile():
    service_name = "/tactile_reader/bias_sensor"

    rospy.loginfo(f"Waiting for service {service_name}...")
    rospy.wait_for_service(service_name)

    try:
        bias_srv = rospy.ServiceProxy(service_name, Trigger)
        resp = bias_srv()  # call the Trigger service
        if resp.success:
            rospy.loginfo(f"Tactile sensor biased successfully: {resp.message}")
        else:
            rospy.logwarn(f"Tactile bias failed: {resp.message}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")



def pose_callback(msg):
    global recorded_poses
    recorded_poses.append(msg)
    rospy.loginfo(f"Recorded pose {len(recorded_poses)}: {msg.pose}")

def wait_for_user_input(prompt):
    while True:
        choice = input(prompt + " [y/n]: ").strip().lower()
        if choice in ["y", "n"]:
            return choice
        print("Please type y or n.")

def publish_leadthrough(pub):
    msg = PoseStampedWithLimits()
    msg.pose.header.seq = 0
    msg.pose.header.stamp = rospy.Time(0)
    msg.pose.header.frame_id = ""

    msg.pose.pose.position.x = -0.4
    msg.pose.pose.position.y = -0.2
    msg.pose.pose.position.z = 0.0

    msg.pose.pose.orientation.x = 1.0
    msg.pose.pose.orientation.y = 0.0
    msg.pose.pose.orientation.z = 0.0
    msg.pose.pose.orientation.w = 0.0

    msg.enable_lead_control = True
    msg.max_force = 2.5
    msg.max_torque = 0.1
    msg.trajectory_time = 2.0

    pub.publish(msg)
    rospy.loginfo("Lead-through mode enabled and message published!")

def compute_rigid_transform(A, B):
    """Compute R, t such that A*R^T + t ≈ B"""
    assert A.shape == B.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R_opt = Vt.T @ U.T

    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = Vt.T @ U.T

    t_opt = centroid_B - R_opt @ centroid_A
    return R_opt, t_opt

def compute_object_pose(poses):
    measured_points = np.array([[p.pose.position.x,
                                 p.pose.position.y,
                                 p.pose.position.z] for p in poses])

    # NOTE: swapped order so we get object->base_link
    R_opt, t_opt = compute_rigid_transform(object_points, measured_points)

    quat = R.from_matrix(R_opt).as_quat()  # [x, y, z, w]

    obj_pose = PoseStamped()
    obj_pose.header.stamp = rospy.Time.now()
    obj_pose.header.frame_id = poses[0].header.frame_id
    obj_pose.pose.position.x = t_opt[0]
    obj_pose.pose.position.y = t_opt[1]
    obj_pose.pose.position.z = t_opt[2]
    obj_pose.pose.orientation.x = quat[0]
    obj_pose.pose.orientation.y = quat[1]
    obj_pose.pose.orientation.z = quat[2]
    obj_pose.pose.orientation.w = quat[3]

    rospy.loginfo(f"Estimated object pose:\n{obj_pose}")
    return obj_pose

def publish_mesh(pose, pub_mesh):
    marker = Marker()
    marker.header.frame_id = "base_link"  # adjust if needed
    marker.header.stamp = rospy.Time.now()
    marker.ns = "object"
    marker.id = 0
    marker.type = Marker.MESH_RESOURCE
    marker.action = Marker.ADD

    marker.pose = pose

    # scale depending on STL units (set to 0.001 if STL is in mm)
    marker.scale.x = 0.001
    marker.scale.y = 0.001
    marker.scale.z = 0.001

    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    marker.mesh_resource = "package://sensor_v2/data_creation/mesh_data/sensor_playground_mouse_flat.stl"
    marker.mesh_use_embedded_materials = False

    pub_mesh.publish(marker)
    rospy.loginfo("Published STL mesh to RViz")

def main():
    rospy.init_node("Hysteresis")
    probe_bool = False
    nr_pokes = 50

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    pub_desired = rospy.Publisher("/tactile/desired_pose",
                                  PoseStampedWithLimits,
                                  queue_size=10)

    pub_mesh = rospy.Publisher("/visualization_marker",
                               Marker,
                               queue_size=10)

    if wait_for_user_input("Enable lead-through mode and bias ft sensor?") == "y":
        bias_netft(sensor_nr=1)
        publish_leadthrough(pub_desired)
    else:
        rospy.loginfo("Lead-through mode skipped.")

    rospy.loginfo("Ready to record 3 poses.")
    while len(recorded_poses) < 3 and not rospy.is_shutdown():
        if wait_for_user_input(f"Record pose {len(recorded_poses)+1}?") == "y":
            rospy.loginfo("Waiting for next /tactile/current_pose message...")
            #msg = rospy.wait_for_message("/tactile/current_pose", PoseStamped)
            #pose_callback(msg)
            pose_msg = get_frame_pose("tactile_calibration", "base_link", tf_buffer)
            if pose_msg:
                pose_callback(pose_msg)
            else:
                rospy.logwarn("Failed to get tactile_calibration pose from TF")
            
        else:
            rospy.loginfo("Skipping pose recording.")
            break;

    if len(recorded_poses) == 3:
        obj_pose_stamped = compute_object_pose(recorded_poses)

        # Publish mesh in RViz
        publish_mesh(obj_pose_stamped.pose, pub_mesh)

            # Build static transform
        static_tf = geometry_msgs.msg.TransformStamped()
        static_tf.header.stamp = rospy.Time.now()
        static_tf.header.frame_id = "base_link"          # parent frame
        static_tf.child_frame_id = "object_frame"    # new frame name
        static_tf.transform.translation.x = obj_pose_stamped.pose.position.x
        static_tf.transform.translation.y = obj_pose_stamped.pose.position.y
        static_tf.transform.translation.z = obj_pose_stamped.pose.position.z
        static_tf.transform.rotation = obj_pose_stamped.pose.orientation

        # Broadcast once (static transform)
        static_broadcaster.sendTransform(static_tf)
    else:
        rospy.logwarn("Not enough poses recorded to compute object pose.")


    if wait_for_user_input("Start probing") == "y":
        probe_bool = True
        pose_reset = Pose()
        pose_reset.position = copy.deepcopy(obj_pose_stamped.pose.position)
        pose_reset.position.z += 0.05 
        q = quaternion_from_euler(np.pi, 0, 0)  # z down
        pose_reset.orientation.x, pose_reset.orientation.y, pose_reset.orientation.z, pose_reset.orientation.w = q
        probe_utility.publish_pose(pub_desired, pose_reset, time=5.0)
        rospy.sleep(5.0)
        bias_netft(sensor_nr=1)
        bias_tactile()
        rospy.sleep(1.0)
    else:
        rospy.loginfo("Probe mode skipped.")
    
    if probe_bool:

        bag_directory = "" # Set path to where you want bag saved
        topics_to_record = ["/tf", "/tf_static", "/netft_data1", "/tactile_reader/tactile_sensor", "/tactile/current_force", "/tactile_mapping/wrench"] 

        # Build Pose
        pose_ = Pose()
        pose_.position.x = 0.0
        pose_.position.y = -0.0
        pose_.position.z = 0.01

        q = quaternion_from_euler(np.pi, 0, 0)  
        pose_.orientation.x, pose_.orientation.y, pose_.orientation.z, pose_.orientation.w = q
        
        pose_t = probe_utility.pose_in_base(object_pose=obj_pose_stamped.pose, target_pose_in_object=pose_)
        probe_utility.publish_pose(pub_desired, pose_t, time=4.0)
        rospy.sleep(4.5)
        rospy.sleep(0.1)
        bias_netft(sensor_nr=1)
        bias_tactile()
        rospy.sleep(0.1)
        
        bag_process = probe_utility.start_rosbag(topics_to_record, "Hysteresis", bag_directory)
        rospy.sleep(2)
        

        for i in range(nr_pokes):
            if rospy.is_shutdown(): 
                break
            pose_.position.z = 0.0025
            pose_t = probe_utility.pose_in_base(object_pose=obj_pose_stamped.pose, target_pose_in_object=pose_)
            probe_utility.publish_pose(pub_desired, pose_t, f_max=10, time=4.5)
            rospy.sleep(5)

            pose_.position.z = 0.01
            pose_t = probe_utility.pose_in_base(object_pose=obj_pose_stamped.pose, target_pose_in_object=pose_)
            probe_utility.publish_pose(pub_desired, pose_t, f_max=10, time=4.5)
            rospy.sleep(5)

        probe_utility.stop_rosbag(bag_process)
        if rospy.is_shutdown(): 
            return
        rospy.sleep(2)

        pose_.position.z = 0.04
        q = quaternion_from_euler(np.pi, 0, 0.0)  
        pose_.orientation.x, pose_.orientation.y, pose_.orientation.z, pose_.orientation.w = q
        pose_t = probe_utility.pose_in_base(object_pose=obj_pose_stamped.pose, target_pose_in_object=pose_)
        probe_utility.publish_pose(pub_desired, pose_t, time=2.0)
        rospy.sleep(2)
    #rospy.spin()

if __name__ == "__main__":
    main()














