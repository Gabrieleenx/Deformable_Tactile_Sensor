#!/usr/bin/env python3
import rospy
import numpy as np
import tf
from sensor_v2.msg import PoseStampedWithLimits
from geometry_msgs.msg import Twist, PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState
from tf.transformations import quaternion_slerp, quaternion_matrix, quaternion_from_matrix

def ensure_quat_hemisphere(q1, q2):
    """Flip q2 if it's in the opposite hemisphere of q1."""
    if np.dot(q1, q2) < 0.0:
        print("flipp")
        return q1, -np.array(q2)
    return q1, np.array(q2)


def rotmat_to_rotvec(R):
    """Return rotation vector (axis * angle) from a 3x3 rotation matrix R."""
    # numerical safety
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(tr)

    if angle < 1e-6:
        return np.zeros(3)

    # For angles near pi, standard formula is ill-conditioned. Handle both regimes.
    if np.pi - angle > 1e-3:
        # regular case
        skew = (R - R.T) / (2.0 * np.sin(angle))
        axis = np.array([skew[2,1], skew[0,2], skew[1,0]])
    else:
        # angle ~ pi: extract axis from diagonal
        A = (R + np.eye(3)) / 2.0
        axis = np.array([np.sqrt(max(A[0,0], 0)),
                         np.sqrt(max(A[1,1], 0)),
                         np.sqrt(max(A[2,2], 0))])
        # pick signs from off-diagonals
        axis[0] = np.copysign(axis[0], R[2,1] - R[1,2])
        axis[1] = np.copysign(axis[1], R[0,2] - R[2,0])
        axis[2] = np.copysign(axis[2], R[1,0] - R[0,1])
        if np.linalg.norm(axis) > 1e-9:
            axis = axis / np.linalg.norm(axis)
        else:
            axis = np.array([1.0, 0.0, 0.0])  # fallback

    return axis * angle



def quat_to_rotvec(q_err):
    """Convert quaternion error to rotation vector (axis * angle)."""
    q = np.array(q_err) / np.linalg.norm(q_err)
    w = q[3]
    xyz = q[0:3]

    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    if angle > np.pi:
        angle -= 2*np.pi
    if np.abs(angle) < 1e-6:
        return np.zeros(3)
    axis = xyz / np.sin(angle/2.0)
    return axis * angle

class TactileController:
    def __init__(self):
        rospy.init_node("tactile_controller")

        # Parameters
        self.kp_pos = rospy.get_param("~kp_pos", 1.0)
        self.kp_ori = rospy.get_param("~kp_ori", 1.0)

        self.max_lin_vel = rospy.get_param("~max_lin_vel", 0.1)   # m/s
        self.max_ang_vel = rospy.get_param("~max_ang_vel", 0.5)    # rad/s
        self.max_force = rospy.get_param("~max_force", 10.0)       # N
        self.max_torque = rospy.get_param("~max_torque", 2.0)      # Nm
        self.force_gain = rospy.get_param("~force_gain", 0.1)     # velocity per N
        self.torque_gain = rospy.get_param("~torque_gain", 10)   # velocity per Nm
        self.control_rate = rospy.get_param("~control_rate", 200.0)  # Hz
        self.tactile_frame = rospy.get_param("~tactile_frame", "tactile_sensor")
        self.base_frame = rospy.get_param("~base_frame", "base_link") 
        
        self.alpha_filter = 0.05  # smoothing factor, smaller = smoother
        
        
        # TF listener
        self.tf_listener = tf.TransformListener()


        # Tool0 -> FT and Tool0 -> Tactile transforms
        #self.tf_tool0_to_ft = rospy.get_param("~tool0_to_ft", [0,0,0,0,0,0,1])
        #self.tf_tool0_to_tactile = rospy.get_param("~tool0_to_tactile", [0,0,0,0,0,0,1])
        self.tf_tool0_to_ft = self.get_tf_transform("tool0", "ft_sensor")
        self.tf_tool0_to_tactile = self.get_tf_transform("tool0", self.tactile_frame)
       

        # States
        self.desired_pose = None
        self.ft_wrench = None
        self.start_pose = None
        self.lead_control = False
        self.traj_duration = 2.0
        self.traj_start_time = rospy.get_time()
        self.excess_force_filtered = np.zeros(3)
        self.excess_torque_filtered = np.zeros(3)
        # Subscribers
        rospy.Subscriber("/tactile/desired_pose", PoseStampedWithLimits, self.desired_pose_cb, tcp_nodelay=True)
        rospy.Subscriber("/netft_data1", WrenchStamped, self.ft_cb)
        rospy.Subscriber("/joint_states", JointState, self.joint_states_cb)

        # Publishers
        self.pub_vel = rospy.Publisher("/cartesian_velocity_command", Twist, queue_size=1)
        self.pub_tactile_pose = rospy.Publisher("/tactile/current_pose", PoseStamped, queue_size=1)
        self.pub_tactile_vel = rospy.Publisher("/tactile/current_velocity", Twist, queue_size=1)
        self.pub_tactile_force = rospy.Publisher("/tactile/current_force", WrenchStamped, queue_size=1)

        self.rate = rospy.Rate(self.control_rate)
        print("ready")
        self.run_controller()


    def get_tf_transform(self, parent_frame, child_frame, timeout=5.0):
        """
        Wait until the transform from parent_frame -> child_frame is available,
        then return it as [x, y, z, qx, qy, qz, qw].
        Returns None if timeout expires.
        """
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            try:
                t = self.tf_listener.getLatestCommonTime(parent_frame, child_frame)
                trans, rot = self.tf_listener.lookupTransform(parent_frame, child_frame, t)
                return list(trans) + list(rot)
            except (tf.Exception, tf.LookupException, tf.ConnectivityException):
                if (rospy.Time.now() - start_time).to_sec() > timeout:
                    rospy.logwarn(f"Timeout: TF from {parent_frame} to {child_frame} not available")
                    return None
                rate.sleep()

    # Callbacks
    def desired_pose_cb(self, msg):
        self.desired_pose = msg.pose
        self.lead_control = msg.enable_lead_control
        self.max_force = msg.max_force       # N
        self.max_torque = msg.max_torque      # Nm
        self.traj_duration = msg.trajectory_time
        self.traj_start_time = rospy.get_time()
        if self.start_pose is not None:
            self.start_pose = self.current_pose


    def ft_cb(self, msg):
        self.ft_wrench = msg

    def joint_states_cb(self, msg):
        pass

    # TF utilities
    def get_current_pose(self):
        try:
            t = self.tf_listener.getLatestCommonTime(self.base_frame, self.tactile_frame)
            trans, rot = self.tf_listener.lookupTransform(self.base_frame, self.tactile_frame, t)
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = self.base_frame
            pose.pose.position.x = trans[0]
            pose.pose.position.y = trans[1]
            pose.pose.position.z = trans[2]
            pose.pose.orientation.x = rot[0]
            pose.pose.orientation.y = rot[1]
            pose.pose.orientation.z = rot[2]
            pose.pose.orientation.w = rot[3]
            return pose
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            return None

    def clamp_velocities(self, lin_vel, ang_vel):
        lin_norm = np.linalg.norm(lin_vel)
        if lin_norm > self.max_lin_vel:
            lin_vel = lin_vel / lin_norm * self.max_lin_vel
        ang_norm = np.linalg.norm(ang_vel)
        if ang_norm > self.max_ang_vel:
            ang_vel = ang_vel / ang_norm * self.max_ang_vel
        return lin_vel, ang_vel

    def project_force_to_tactile(self, ft_wrench):
        ft_mat = quaternion_matrix(self.tf_tool0_to_ft[3:])
        ft_mat[0:3,3] = self.tf_tool0_to_ft[0:3]
        tactile_mat = quaternion_matrix(self.tf_tool0_to_tactile[3:])
        tactile_mat[0:3,3] = self.tf_tool0_to_tactile[0:3]

        force_vec = np.array([ft_wrench.wrench.force.x,
                              ft_wrench.wrench.force.y,
                              ft_wrench.wrench.force.z])
        torque_vec = np.array([ft_wrench.wrench.torque.x,
                               ft_wrench.wrench.torque.y,
                               ft_wrench.wrench.torque.z])

        R_ft = ft_mat[0:3,0:3]
        R_tactile = tactile_mat[0:3,0:3]

        force_tactile = R_tactile.T @ R_ft @ force_vec
        torque_tactile = R_tactile.T @ R_ft @ torque_vec

        wrench = WrenchStamped()
        wrench.header.stamp = rospy.Time.now()
        wrench.header.frame_id = self.tactile_frame
        wrench.wrench.force.x = force_tactile[0]
        wrench.wrench.force.y = force_tactile[1]
        wrench.wrench.force.z = force_tactile[2]
        wrench.wrench.torque.x = torque_tactile[0]
        wrench.wrench.torque.y = torque_tactile[1]
        wrench.wrench.torque.z = torque_tactile[2]
        return wrench

    # Main control loop
    def run_controller(self):
        
        
        

        while not rospy.is_shutdown():
            # Force-based velocity correction
            tactile_wrench = None
            if self.ft_wrench is not None:
                tactile_wrench = self.project_force_to_tactile(self.ft_wrench)
                force_vec = np.array([tactile_wrench.wrench.force.x,
                                      tactile_wrench.wrench.force.y,
                                      tactile_wrench.wrench.force.z])
                torque_vec = np.array([tactile_wrench.wrench.torque.x,
                                       tactile_wrench.wrench.torque.y,
                                       tactile_wrench.wrench.torque.z])

            # Publish tactile pose & wrench
            if tactile_wrench is not None:
                self.pub_tactile_force.publish(tactile_wrench)
            
            
            # Update current pose from TF if available

            try:
                (trans, rot) = self.tf_listener.lookupTransform(self.base_frame, self.tactile_frame, rospy.Time(0))
                self.current_pose = PoseStamped()
                self.current_pose.header.stamp = rospy.Time.now()
                self.current_pose.header.frame_id = self.base_frame
                self.current_pose.pose.position.x = trans[0]
                self.current_pose.pose.position.y = trans[1]
                self.current_pose.pose.position.z = trans[2]
                self.current_pose.pose.orientation.x = rot[0]
                self.current_pose.pose.orientation.y = rot[1]
                self.current_pose.pose.orientation.z = rot[2]
                self.current_pose.pose.orientation.w = rot[3]
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn_throttle(5, "TF for tactile not found")
                self.rate.sleep()
                continue

            

            self.pub_tactile_pose.publish(self.current_pose)



            if self.desired_pose is None:
                self.desired_pose = self.current_pose

            if self.start_pose is None:
                self.start_pose = self.current_pose

            t = rospy.get_time() - self.traj_start_time
            alpha = min(t / self.traj_duration, 1.0)

            # Trajectory interpolation
            pos_start = np.array([self.start_pose.pose.position.x,
                                  self.start_pose.pose.position.y,
                                  self.start_pose.pose.position.z])
            pos_goal = np.array([self.desired_pose.pose.position.x,
                                 self.desired_pose.pose.position.y,
                                 self.desired_pose.pose.position.z])
            pos_traj = pos_start * (1 - alpha) + pos_goal * alpha

            quat_start = [self.start_pose.pose.orientation.x,
                          self.start_pose.pose.orientation.y,
                          self.start_pose.pose.orientation.z,
                          self.start_pose.pose.orientation.w]
            quat_goal = [self.desired_pose.pose.orientation.x,
                         self.desired_pose.pose.orientation.y,
                         self.desired_pose.pose.orientation.z,
                         self.desired_pose.pose.orientation.w]
            quat_traj = quaternion_slerp(quat_start, quat_goal, alpha)




            # Pose velocity
            pos_curr = np.array([self.current_pose.pose.position.x,
                                 self.current_pose.pose.position.y,
                                 self.current_pose.pose.position.z])
            lin_error = pos_traj - pos_curr

            lin_vel_cmd = self.kp_pos * lin_error
            
            # --- Orientation error in BASE frame (no hemisphere issues) ---
            quat_curr = np.array([self.current_pose.pose.orientation.x,
                                self.current_pose.pose.orientation.y,
                                self.current_pose.pose.orientation.z,
                                self.current_pose.pose.orientation.w])
            quat_des  = np.array([quat_traj[0], quat_traj[1], quat_traj[2], quat_traj[3]])

            # Build rotation matrices: base_R_tactile
            R_b_curr = tf.transformations.quaternion_matrix(quat_curr)[0:3, 0:3]
            R_b_des  = tf.transformations.quaternion_matrix(quat_des )[0:3, 0:3]

            # Relative rotation in BASE frame
            R_err = R_b_des.dot(R_b_curr.T)

            # Rotation vector IN BASE FRAME directly
            ang_error_base = rotmat_to_rotvec(R_err)

            # Angular velocity command (base frame)
            ang_vel_cmd = self.kp_ori * ang_error_base

            if self.lead_control:

                ang_vel_cmd = np.zeros(3)
                lin_vel_cmd = np.zeros(3)
                self.desired_pose = self.current_pose
                self.start_pose = self.current_pose



            # Force-based velocity correction
            tactile_wrench = None
            # Force-based velocity correction with hard limits
            if self.ft_wrench is not None:
                tactile_wrench = self.project_force_to_tactile(self.ft_wrench)
                force_vec = np.array([tactile_wrench.wrench.force.x,
                                    tactile_wrench.wrench.force.y,
                                    tactile_wrench.wrench.force.z])
                torque_vec = np.array([tactile_wrench.wrench.torque.x,
                                    tactile_wrench.wrench.torque.y,
                                    tactile_wrench.wrench.torque.z])

                # Compute excess over max limits
                excess_force = np.zeros(3)
                excess_torque = np.zeros(3)

                force_mag = np.linalg.norm(force_vec)
                if force_mag > self.max_force:
                    excess_force = (force_vec / force_mag) * (force_mag - self.max_force)

                torque_mag = np.linalg.norm(torque_vec)
                if torque_mag > self.max_torque:
                    excess_torque = (torque_vec / torque_mag) * (torque_mag - self.max_torque)


                # inside control loop, after computing excess_force/excess_torque
                self.excess_force_filtered = self.alpha_filter  * excess_force + (1-self.alpha_filter ) * self.excess_force_filtered
                self.excess_torque_filtered = self.alpha_filter  * excess_torque + (1-self.alpha_filter ) * self.excess_torque_filtered

                try:
                    # Rotation tactile -> base
                    (trans, rot) = self.tf_listener.lookupTransform(self.base_frame,
                                                                    self.tactile_frame,
                                                                    rospy.Time(0))
                    R_base_tactile = tf.transformations.quaternion_matrix(rot)[0:3, 0:3]

                    # Transform excess into base frame
                    excess_force_base = R_base_tactile @ self.excess_force_filtered 
                    excess_torque_base = R_base_tactile @ self.excess_torque_filtered

                    # Apply corrections **with negative sign for resistance**
                    lin_vel_cmd += self.force_gain * excess_force_base
                    ang_vel_cmd += self.torque_gain * excess_torque_base

                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    rospy.logwarn_throttle(5, "TF transform tactile->base unavailable for force correction")

            # Clamp velocities
            lin_vel_cmd, ang_vel_cmd = self.clamp_velocities(lin_vel_cmd, ang_vel_cmd)

            # Publish velocity
            twist_msg = Twist()
            twist_msg.linear.x = lin_vel_cmd[0]
            twist_msg.linear.y = lin_vel_cmd[1]
            twist_msg.linear.z = lin_vel_cmd[2]
            twist_msg.angular.x = ang_vel_cmd[0]
            twist_msg.angular.y = ang_vel_cmd[1]
            twist_msg.angular.z = ang_vel_cmd[2]
            self.pub_vel.publish(twist_msg)


            self.rate.sleep()


if __name__ == "__main__":
    try:
        TactileController()
    except rospy.ROSInterruptException:
        pass