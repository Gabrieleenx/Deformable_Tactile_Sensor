#!/usr/bin/env python3
import rospy
import torch
import numpy as np
from torch import nn
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from sensor_v2.msg import TactileSensor 
from sensor_v2.msg import TactileVelSensor 
from geometry_msgs.msg import WrenchStamped, Vector3 
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def build_matrix(theta1, theta2, r1, r2):
    """
    Returns the 4x3 matrix:
        [ Rz^T(theta1)   Rz^T(theta1) p_perp(r1) ]
        [ Rz^T(theta2)   Rz^T(theta2) p_perp(r2) ]

    r1, r2 are 2-element vectors/lists/arrays like [x, y]
    """
    # Rotation transpose in 2D
    def RzT(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[ c,  s],
                         [-s,  c]])

    # p_perp([x, y]) = [-y, x]
    def p_perp(r):
        x, y = r
        return np.array([-y, x])

    R1 = RzT(theta1)
    R2 = RzT(theta2)

    p1 = p_perp(r1)
    p2 = p_perp(r2)

    # Each block: 2x2 and 2x1 → stack into 4x3
    upper = np.hstack([R1, (R1 @ p1).reshape(2,1)])
    lower = np.hstack([R2, (R2 @ p2).reshape(2,1)])

    return np.vstack([upper, lower])


def make_circular_mask(size=32):
    """Return a (size*size,) mask with 1 inside circle, 0 outside."""
    y, x = np.ogrid[:size, :size]
    center = (size - 1) / 2
    radius = size / 2
    dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    mask = (dist <= radius).astype(np.float32)
    return mask

# ------------------- Model Definition -------------------
class MultiOutputMLP(nn.Module):
    def __init__(self, in_dim=36, hidden_dim=256, thickness_dim=(32,32), scalar_dim=6, hall13_dim=3):
        super().__init__()
        self.thickness_shape = thickness_dim  
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.thickness_head = nn.Sequential(
            nn.Linear(hidden_dim, 64 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),          # (batch, 64, 4, 4)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # → 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # → 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),   # → 32x32
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),                       # (batch, 1, 32, 32)
        )
        # Scalars + hall13
        self.scalars_head = nn.Linear(hidden_dim, scalar_dim)
        self.hall13_head = nn.Linear(hidden_dim, hall13_dim)

    def forward(self, x):
        z = self.encoder(x)

        # Thickness prediction
        t_pred = self.thickness_head(z)           
        t_pred = t_pred.view(z.size(0), -1)      
        

        # Scalars and hall13
        s_pred = self.scalars_head(z)
        h13_pred = self.hall13_head(z)

        return t_pred, s_pred, h13_pred


class SensorConfig:
    def __init__(self):
        self.cpmm1 = 206.9 # counts per mm
        self.cpmm2 = 201.9 
        self.px1 = -7.68
        self.py1 = 0.46
        self.omega1 = 1.54
        self.px2 = 8.64
        self.py2 = -0.04
        self.omega2 = -1.57
        

class VelocityMapping:
    def __init__(self, sensor_config, pub, sub_sample=1):
        self.last_t = rospy.Time.now().to_sec()
        self.delta_sum = np.array([[0], [0], [0], [0]])
        self.sub_sample = sub_sample
        self.cpmm = np.array([[sensor_config.cpmm1], [sensor_config.cpmm1], [sensor_config.cpmm2], [sensor_config.cpmm2]])
        self.mapping = build_matrix(theta1=sensor_config.omega1,
                                    theta2=sensor_config.omega2,
                                    r1=[sensor_config.px1, sensor_config.py1],
                                    r2=[sensor_config.px2, sensor_config.py2])
        self.mapping_pinv = np.linalg.pinv(self.mapping)
        self.pub = pub
        self.k = 0

    def callback(self, sensor_data):
        dx1 = sensor_data.mouse_sensor[0].x
        dy1 = sensor_data.mouse_sensor[0].y
        dx2 = sensor_data.mouse_sensor[1].x
        dy2 = sensor_data.mouse_sensor[1].y
        self.delta_sum += np.array([[dx1], [dy1], [dx2], [dy2]])
        self.k += 1
        if self.sub_sample <= self.k:
            delta_mm = self.delta_sum / self.cpmm
            t_now = sensor_data.header.stamp.to_sec()
            dt = (t_now - self.last_t)
            vel_mm_s = delta_mm/dt
            self.last_t = t_now
            self.k = 0
            self.delta_sum = np.array([[0], [0], [0], [0]])
            sensor_vel = self.mapping_pinv.dot(vel_mm_s)
            vel_msg = TactileVelSensor()
            vel_msg.header.stamp = sensor_data.header.stamp
            vel_msg.vx = sensor_vel[0, 0]
            vel_msg.vy = sensor_vel[1, 0]
            vel_msg.omega = sensor_vel[2, 0]
            self.pub.publish(vel_msg)

# ------------------- ROS Node -------------------
class TactileMappingNode:
    def __init__(self):
        self.mask = make_circular_mask(32)
        rospy.init_node("tactile_mapping_node", anonymous=True)
        self.frame_id = "tactile_sensor"
        self.area = 0.02*0.02*np.pi
        # Parameters
        model_path = rospy.get_param(
            "~model_path",
            "../model.pth" # Put path to model 
        )
        device = torch.device("cpu")

        # Publishers
        self.thickness_pub = rospy.Publisher("/tactile_mapping/thickness", Image, queue_size=1)
        self.scalars_pub = rospy.Publisher("/tactile_mapping/wrench", WrenchStamped, queue_size=1) # Updated
        self.hall13_pub = rospy.Publisher("/tactile_mapping/hall13", Float32MultiArray, queue_size=1)
        self.vel_pub = rospy.Publisher("/tactile_mapping/velocity", TactileVelSensor, queue_size=1)
        # Load model + scalers
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model = MultiOutputMLP()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        self.delta_hall13_x = 0
        self.delta_hall13_y = 0
        self.delta_hall13_z = 0

        self.in_scaler = checkpoint["in_scaler"]
        self.thickness_scaler = checkpoint["thickness_scaler"]
        self.scalars_scaler = checkpoint["scalars_scaler"]
        self.hall13_scaler = checkpoint["hall13_scaler"]

        # velocity mapping
        sensor_conf = SensorConfig()
        self.vel_map = VelocityMapping(sensor_config=sensor_conf,
                                       pub=self.vel_pub,
                                       sub_sample=1)

        # Subscribe to tactile sensor
        rospy.Subscriber("/tactile_reader/tactile_sensor", TactileSensor, self.callback)
        rospy.loginfo("Tactile mapping node started. Waiting for tactile_sensor messages...")


    def calc_offset(self, i):
        if i in [0, 3, 7, 8, 10, 11] :
            return self.delta_hall13_y, -self.delta_hall13_x, self.delta_hall13_z
        if i in [1, 2, 4, 5, 6, 9]:
            return -self.delta_hall13_y, self.delta_hall13_x, self.delta_hall13_z
        return 0, 0, 0

    def callback(self, msg):
        try:
            self.vel_map.callback(msg)
            # Extract first 12 hall sensors (x,y,z only)
            hall_values = []
            for i in range(12):
                hall = msg.hall_sensor[i]
                dx, dy, dz = self.calc_offset(i)
                hall_values.extend([hall.x-0.95*dx, hall.y-0.95*dy, hall.z-0.95*dz])
            hall_np = np.array(hall_values, dtype=np.float32).reshape(1, -1)
            hall_scaled = self.in_scaler.transform(hall_np)
            hall_tensor = torch.from_numpy(hall_scaled).float()

            # Run model
            with torch.no_grad():
                t_pred, s_pred, h13_pred = self.model(hall_tensor)

            # Inverse scale
            t_pred_np = self.thickness_scaler.inverse_transform(t_pred.numpy())
            s_pred_np = self.scalars_scaler.inverse_transform(s_pred.numpy())[0]
            h13_pred_np = self.hall13_scaler.inverse_transform(h13_pred.numpy())
            
            self.delta_hall13_x = msg.hall_sensor[12].x - h13_pred_np[0,0]
            self.delta_hall13_y = msg.hall_sensor[12].y - h13_pred_np[0,1]
            self.delta_hall13_z = msg.hall_sensor[12].z - h13_pred_np[0,2]
            
            # --- Publish thickness as sensor_msgs/Image (6x6) ---
            #t_map = t_pred_np.reshape(32, 32).astype(np.float32) * abs(s_pred_np[2]) /  self.area *self.mask
            t_map = t_pred_np.reshape(32, 32).astype(np.float32)
            t_map = np.rot90(t_map, k=1)  # 90° CCW rotation
            t_map = t_map * abs(s_pred_np[2]) / self.area * self.mask
            t_image_msg = Image()
            t_image_msg.header.stamp = rospy.Time.now()
            t_image_msg.height = t_map.shape[0]
            t_image_msg.width = t_map.shape[1]
            t_image_msg.encoding = "32FC1"  # float32, single channel
            t_image_msg.is_bigendian = 0
            t_image_msg.step = t_map.shape[1] * 4  # 4 bytes per float32
            t_image_msg.data = t_map.tobytes()
            self.thickness_pub.publish(t_image_msg)

            # --- Publish forces and torques in a WrenchStamped message ---
            wrench_msg = WrenchStamped()
            wrench_msg.header.stamp = msg.header.stamp
            wrench_msg.header.frame_id = self.frame_id
            
            wrench_msg.wrench.force.x = s_pred_np[0]
            wrench_msg.wrench.force.y = s_pred_np[1]
            wrench_msg.wrench.force.z = s_pred_np[2]
            
            wrench_msg.wrench.torque.x = s_pred_np[3]
            wrench_msg.wrench.torque.y = s_pred_np[4]
            wrench_msg.wrench.torque.z = s_pred_np[5]
            
            self.scalars_pub.publish(wrench_msg)

            # --- Publish hall13 ---
            hall13_msg = Float32MultiArray()
            hall13_msg.data = h13_pred_np.flatten().tolist()
            self.hall13_pub.publish(hall13_msg)

        except Exception as e:
            rospy.logerr(f"Error in tactile callback: {e}")

# ------------------- Main -------------------
if __name__ == "__main__":
    try:
        node = TactileMappingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
