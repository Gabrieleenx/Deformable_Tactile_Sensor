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

    marker.mesh_resource = "package://sensor_v2/data_creation/mesh_data/sensor_playground v3.stl"
    marker.mesh_use_embedded_materials = False

    pub_mesh.publish(marker)
    rospy.loginfo("Published STL mesh to RViz")

def main():
    rospy.init_node("interactive_pose_recorder")
    probe_bool = False # dont touch
    nr_pokes = 30 # for training
    nr_tangential = 3

    # for test or validation 
    #nr_pokes = 5
    #nr_tangential = 1

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
        rospy.sleep(6.0)
        bias_netft(sensor_nr=1)
        bias_tactile()
        rospy.sleep(1.0)
    else:
        rospy.loginfo("Probe mode skipped.")
    
    if probe_bool:
        probe = probe_utility.Prope(0.05)
        calibration_obj = probe_utility.TargetShape("calibrate_height", -0.0251,-0.025, 0.025,0.0251, 0.00001, max_xy_delta=0.000001, max_tz_delta=0.0001, max_txy_delta=0.00001, f_max=5.0, t_max=0.3)
        target_square = probe_utility.TargetShape("square", -0.105,-0.065, 0.05,0.115, 0.003, max_xy_delta=0.003, max_tz_delta=0.3, max_txy_delta=0.1, f_max=15.0, t_max=0.5)
        target_cylinder = probe_utility.TargetShape(name="cylinder_flat_45", x_min=-0.03, x_max=-0.02, y_min=0.02, y_max=0.055, max_z_delta=0.006, max_xy_delta=0.003, max_tz_delta=0.3, max_txy_delta=0.1, f_max=15.0, t_max=0.6)
        target_cylinder_curve_100 = probe_utility.TargetShape(name="100mm_cylinder_curve", x_min=-0.115, x_max=-0.07, y_min=-0.02, y_max=0.02, max_z_delta=0.006, max_xy_delta=0.003, max_tz_delta=0.3, max_txy_delta=0.1, f_max=15.0, t_max=0.5)
        target_cylinder_curve_25 = probe_utility.TargetShape(name="25mm_cylinder_curve", x_min=-0.09, x_max=-0.06, y_min=-0.115, y_max=-0.07, max_z_delta=0.006, max_xy_delta=0.003, max_tz_delta=0.3, max_txy_delta=0.1, f_max=15.0, t_max=0.5)
        target_cylinder_flat_25 = probe_utility.TargetShape(name="cylinder_flat_25", x_min=-0.03, x_max=-0.02, y_min=-0.055, y_max=-0.02, max_z_delta=0.006, max_xy_delta=0.003, max_tz_delta=0.3, max_txy_delta=0.1, f_max=10.0, t_max=0.3)
        target_peg_curve = probe_utility.TargetShape(name="peg_curve", x_min=-0.005, x_max=0.005, y_min=-0.09, y_max=-0.06, max_z_delta=0.006, max_xy_delta=0.003, max_tz_delta=0.3, max_txy_delta=0.1, f_max=6.0, t_max=0.3)
        target_line_1mm = probe_utility.TargetShape(name="line_1mm", x_min=0.07, x_max=0.08, y_min=-0.11, y_max=-0.06, max_z_delta=0.006, max_xy_delta=0.003, max_tz_delta=0.3, max_txy_delta=0.1, f_max=10.0, t_max=0.3)
        dome_50 = probe_utility.TargetShape(name="dome_50", x_min=0.02, x_max=0.034, y_min=-0.06, y_max=-0.02, max_z_delta=0.006, max_xy_delta=0.003, max_tz_delta=0.3, max_txy_delta=0.1, f_max=10.0, t_max=0.3)
        peg_flat = probe_utility.TargetShape(name="peg_flat", x_min=0.07, x_max=0.09, y_min=-0.01, y_max=0.01, max_z_delta=0.006, max_xy_delta=0.003, max_tz_delta=0.3, max_txy_delta=0.1, f_max=6.0, t_max=0.3)
        dome_100 = probe_utility.TargetShape(name="dome_100", x_min=0.02, x_max=0.045, y_min=0.02, y_max=0.035, max_z_delta=0.006, max_xy_delta=0.003, max_tz_delta=0.3, max_txy_delta=0.1, f_max=15.0, t_max=0.5)
        target_line_4mm = probe_utility.TargetShape(name="line_4mm", x_min=0.07, x_max=0.115, y_min=0.06, y_max=0.09, max_z_delta=0.006, max_xy_delta=0.003, max_tz_delta=0.3, max_txy_delta=0.1, f_max=10.0, t_max=0.3)
        target_line_4mm_rot = probe_utility.TargetShape(name="line_4mm_rot", x_min=0.075, x_max=0.076, y_min=0.065, y_max=0.085, max_z_delta=0.006, max_xy_delta=0.003, max_tz_delta=0.3, max_txy_delta=0.0, f_max=10.0, t_max=0.3)
        
        
        target_objects = [target_square, 
                          target_cylinder, 
                          target_cylinder_curve_100, 
                          target_cylinder_curve_25, 
                          target_cylinder_flat_25, 
                          target_peg_curve, 
                          target_line_1mm,
                          dome_50,
                          peg_flat,
                          dome_100,
                          target_line_4mm,
                          target_line_4mm_rot]
        

        bag_directory = "" # Set where you want the bag stored
        topics_to_record = ["/tf", "/tf_static", "/netft_data1", "/tactile_reader/tactile_sensor", "/tactile/current_force"]
        
        
        # Calibraite height bag
        bag_process = probe_utility.start_rosbag(topics_to_record, calibration_obj.name, bag_directory)
        pose_cal = probe.gen_pos(calibration_obj, obj_pose_stamped.pose)
        probe_utility.publish_pose(pub_desired, pose_cal, time=2.0)
        rospy.sleep(2.1)
        bias_netft(sensor_nr=1)
        bias_tactile()
        rospy.sleep(0.1)
        pose_in_contact = probe.go_to_height(calibration_obj, pose_cal, obj_pose_stamped.pose, 0.020, pub_desired, 2.0)
        rospy.sleep(3.0)
        probe.go_to_height(calibration_obj, pose_cal, obj_pose_stamped.pose, 0.025, pub_desired, 2.0)
        rospy.sleep(2.5)
        
        probe_utility.stop_rosbag(bag_process)
        if rospy.is_shutdown(): 
            return
 
        base_height = obj_pose_stamped.pose.position.z
        bag_process = probe_utility.start_rosbag(topics_to_record, "torque_and_xy", bag_directory)
        current_pose = get_frame_pose("tactile_sensor", "base_link", tf_buffer).pose
        height_steps = [0.0245, 0.024, 0.0235, 0.023, 0.0225, 0.022]
        force_list = [2, 4, 8, 12, 16, 20]
        probe.explore_torque_and_tangential(current_pose, base_height=base_height, height_steps=height_steps, force_list=force_list, pub=pub_desired, dxy=0.004, dtz=0.3, nr_tangential=nr_tangential)
        probe_utility.stop_rosbag(bag_process)
        if rospy.is_shutdown(): 
            return
        
        probe.go_to_height(calibration_obj, pose_cal, obj_pose_stamped.pose, 0.05, pub_desired, 2.0)
        rospy.sleep(2.5)

        # run the probing 
        current_pose = get_frame_pose("tactile_sensor", "base_link", tf_buffer).pose
        for target_obj in target_objects: 
            bag_process = probe_utility.start_rosbag(topics_to_record, target_obj.name, bag_directory)
            if rospy.is_shutdown(): 
                break
            for i in range(nr_pokes):
                pose_ = probe.gen_pos(target_obj, obj_pose_stamped.pose)
                 
                probe.move_to_target(pose_, current_pose, pub_desired)
                rospy.sleep(0.1)
                bias_netft(sensor_nr=1)
                bias_tactile()
                rospy.sleep(0.1)
                if rospy.is_shutdown(): 
                    break
                pose_in_contact = probe.go_to_height(target_obj, pose_, obj_pose_stamped.pose, 0.023, pub_desired, 2.0)
                rospy.sleep(3.0)
                probe.gen_delta_probe(target_obj, pose_in_contact, pub_desired)

                probe.go_to_height(target_obj, pose_, obj_pose_stamped.pose, 0.05, pub_desired, 2.0)
                rospy.sleep(2.0)
                current_pose = copy.deepcopy(pose_)
                if rospy.is_shutdown(): 
                    break
            probe_utility.stop_rosbag(bag_process)
    
    #rospy.spin()

if __name__ == "__main__":
    main()
