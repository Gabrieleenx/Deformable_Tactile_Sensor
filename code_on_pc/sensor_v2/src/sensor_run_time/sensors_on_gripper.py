#!/usr/bin/env python3

import struct, serial
from std_msgs.msg import Int16MultiArray
from sensor_v2.msg import TactileSensor, MouseSensor, HallSensor
import rospy
import serial
import serial.tools.list_ports
import copy
import sys
import time
import numpy as np
import threading
from std_srvs.srv import Trigger, TriggerResponse
from gripper.msg import Gripper_monitor

def find_esp32s3_port(baudrate):
    sensor_1 = None
    sensor_2 = None
    ports = serial.tools.list_ports.comports()
    sensor_ports = []
    for port in ports:
        rospy.loginfo(f"Found device: {port.device} - {port.description} - {port.hwid}")
        if 'ESP32' in port.description or \
           'VID:PID=303A:1001' in port.hwid.upper():
            rospy.loginfo(f"ESP32-S3 found on port: {port.device}")
            sensor_ports.append(port.device)
    if len(sensor_ports) == 0:
        rospy.logwarn("ESP32-S3 device not found")
        sys.exit()  # Exit the script 
    time.sleep(0.1)

    for sensor_port in sensor_ports:
        msg = "n" + "\n"
        msg_start = "s" + "\n"
        sensor=serial.Serial(sensor_port, baudrate)
        sensor.timeout = 1

        time.sleep(0.2)
        sensor.write(msg.encode())
        i = 0
        attempts = 1000
        while not rospy.is_shutdown():
            sensor.write(msg.encode())
            msg_out = read_serial(sensor)

            if msg_out == "resynch":
                print("Vel sensors:", msg_out)
                break            
            
            if msg_out == "sensor_1":
                print("Vel sensors:", "sensor 1 connected")
                sensor_1 = sensor_port
                sensor.write(msg_start.encode())
                break
            elif msg_out == "sensor_2":
                print("Vel sensors:", "sensor 2 connected")
                sensor_2 = sensor_port
                sensor.write(msg_start.encode())
                break
            i+=1
            if i > attempts:
                break
    return sensor_1, sensor_2 

def read_serial(sensor_port):
    data = sensor_port.readline()[:-2] # keep everyting but the new line 
    if data: # if there is data and not just a blank line
        try:
            data_str = string = data.decode('utf-8')
            data_str_vals = data_str.split("/t") # remove b' and ' from byte array
            key = str(data_str_vals[0])
            if key == "n":
                return str(data_str_vals[1])
            else:
                print("else ", key)
                print(data_str_vals)

        except Exception as error:
            pass
    else:
        if sensor_port.isOpen():
            print("Vel sensors:", "Serial port is open")
        else:
            print("Vel sensors:", "Serial port is not open")


from dataclasses import dataclass
import csv

@dataclass
class ExpParams:
    A: float
    k: float
    C: float


class InterferenceCompensation:
    def __init__(self, calibration_csv):
        self.bias_pos = 100.0
        self.contact_pos = 6.5

        # calibration[index]["x"], ["y"], ["z"]
        self.calibration = {}

        with open(calibration_csv, newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                idx = int(row["hall_sensor"])
                axis = row["axis"]

                if idx not in self.calibration:
                    self.calibration[idx] = {}

                self.calibration[idx][axis] = ExpParams(
                    A=float(row["A"]),
                    k=float(row["k"]),
                    C=float(row["C"]),
                )

    def update_bias_position(self, pos):
        self.bias_pos = pos

    def get_compensation(self, index, pos):

        px = self.calibration[index]["x"]
        py = self.calibration[index]["y"]
        pz = self.calibration[index]["z"]

        cp = self.contact_pos
        bp = self.bias_pos

        dpos = pos - cp
        dbias = bp - cp

        return (
            px.A * (np.exp(-px.k * dpos) - np.exp(-px.k * dbias)),
            py.A * (np.exp(-py.k * dpos) - np.exp(-py.k * dbias)),
            pz.A * (np.exp(-pz.k * dpos) - np.exp(-pz.k * dbias)),
        )

    
        

class TactileSensorClass:
    def __init__(self, serial_port, sensor_nr, interference_path):
        self.e_list = []
        self.sensor_nr = sensor_nr
        self.serial_port = serial_port
        self.alpha = 0.1
        self.current_reading = TactileSensor()
        self.current_filtered_reading = TactileSensor()
        self.offset_filtered_reading = TactileSensor()
        self.previous_reading = TactileSensor()
        self.offset = TactileSensor()
        for i in range(13):
            hall_sensor = HallSensor()
            hall_sensor.x = 0.0
            hall_sensor.y = 0.0
            hall_sensor.z = 0.0
            hall_sensor.T = 0.0
            self.current_reading.hall_sensor.append(hall_sensor)
            self.offset.hall_sensor.append(copy.deepcopy(hall_sensor))
            self.previous_reading.hall_sensor.append(copy.deepcopy(hall_sensor)) 
            self.current_filtered_reading.hall_sensor.append(copy.deepcopy(hall_sensor)) 
            self.offset_filtered_reading.hall_sensor.append(copy.deepcopy(hall_sensor)) 
        self.pub = rospy.Publisher("~tactile_sensor"+str(sensor_nr), TactileSensor, queue_size=2, tcp_nodelay=True)
        rospy.Subscriber("/gripper_monitor", Gripper_monitor, self.callback_gripper_monitor, tcp_nodelay=True, queue_size=1)
        self.packet_size = 4*2 + 13*4*2  # int16 = 2 bytes
        self.reset_service = rospy.Service("~bias_sensor"+str(sensor_nr), Trigger, self.handle_reset_offset)

        self.pos = 140
        
        self.ser = None
        self.read_thread = None
        self.running = True

        self.interferance_comp = InterferenceCompensation(interference_path)


        self.baudrate = rospy.get_param('~baudrate', 115200)
        try:
            self.ser = serial.Serial(self.serial_port, self.baudrate, timeout=0.1)
        except serial.SerialException as e:
            rospy.logerr(f"Failed to open serial port {self.serial_port}: {e}")
            rospy.signal_shutdown("Serial port error")
            return

        
    def handle_reset_offset(self, req):
        rospy.loginfo("Resetting tactile sensor offset")
        self.offset = copy.deepcopy(self.offset_filtered_reading)
        self.interferance_comp.update_bias_position(self.pos)
        return TriggerResponse(success=True, message="Offset reset successful")

    def callback_gripper_monitor(self, data):
        self.pos = data.position 

    def run(self):
        while not rospy.is_shutdown():
            # look for sync byte
            b = self.ser.read()
            if b != b'\xAA':
                continue
            pkt = self.ser.read(self.packet_size)
            if len(pkt) != self.packet_size:
                continue

            start = time.perf_counter()

            values = struct.unpack("<" + "h"*56, pkt)  # little-endian, 56 int16
            tactile_sensor = TactileSensor()
            tactile_sensor.header.stamp = rospy.Time.now()
            tactile_sensor.header.frame_id = "tactile_sensor"+str(self.sensor_nr)
            mouse_sensor = MouseSensor()
            mouse_sensor.x = values[0]
            mouse_sensor.y = values[1]
            tactile_sensor.mouse_sensor.append(mouse_sensor)
            mouse_sensor = MouseSensor()
            mouse_sensor.x = values[2]
            mouse_sensor.y = values[3]
            tactile_sensor.mouse_sensor.append(mouse_sensor)

            for i in range(13):
                hall_sensor = HallSensor()


                self.current_reading.hall_sensor[i].x = values[0+4+4*i]* 0.150 
                self.current_reading.hall_sensor[i].y = values[1+4+4*i]* 0.150 
                self.current_reading.hall_sensor[i].z = values[2+4+4*i]* 0.242 


                hall_sensor.T = ((values[3+4+4*i]& 0xFFFF)- 46244.0)/45.2 + 25.0

                ix, iy, iz = self.interferance_comp.get_compensation(index=i, pos=self.pos)
                #print(ix, iy, iz, self.pos)

                self.current_filtered_reading.hall_sensor[i].x = self.alpha * (self.current_reading.hall_sensor[i].x - self.offset.hall_sensor[i].x - ix) + (1 - self.alpha) *self.previous_reading.hall_sensor[i].x
                self.current_filtered_reading.hall_sensor[i].y = self.alpha * (self.current_reading.hall_sensor[i].y - self.offset.hall_sensor[i].y - iy) + (1 - self.alpha) *self.previous_reading.hall_sensor[i].y
                self.current_filtered_reading.hall_sensor[i].z = self.alpha * (self.current_reading.hall_sensor[i].z - self.offset.hall_sensor[i].z - iz) + (1 - self.alpha) *self.previous_reading.hall_sensor[i].z
                self.previous_reading.hall_sensor[i].x = self.current_filtered_reading.hall_sensor[i].x
                self.previous_reading.hall_sensor[i].y = self.current_filtered_reading.hall_sensor[i].y
                self.previous_reading.hall_sensor[i].z = self.current_filtered_reading.hall_sensor[i].z

                self.offset_filtered_reading.hall_sensor[i].x = self.alpha * (self.current_reading.hall_sensor[i].x) + (1 - self.alpha) *self.offset_filtered_reading.hall_sensor[i].x
                self.offset_filtered_reading.hall_sensor[i].y = self.alpha * (self.current_reading.hall_sensor[i].y) + (1 - self.alpha) *self.offset_filtered_reading.hall_sensor[i].y
                self.offset_filtered_reading.hall_sensor[i].z = self.alpha * (self.current_reading.hall_sensor[i].z) + (1 - self.alpha) *self.offset_filtered_reading.hall_sensor[i].z



                hall_sensor.x = self.current_filtered_reading.hall_sensor[i].x 
                hall_sensor.y = self.current_filtered_reading.hall_sensor[i].y
                hall_sensor.z = self.current_filtered_reading.hall_sensor[i].z

                tactile_sensor.hall_sensor.append(hall_sensor)

            elapsed = time.perf_counter() - start
            self.e_list.append(elapsed)
            if len(self.e_list) >= 250:
                avg = np.mean(self.e_list)
                print(f"Average elapsed: {avg * 1e3:.2f} ms")
                self.e_list.clear()   # or self.e_list = []
            self.pub.publish(tactile_sensor)


if __name__ == "__main__":
    rospy.init_node("tactile_reader")
    port_sensor_1, port_sensor_2 = find_esp32s3_port(baudrate=115200)

    if port_sensor_1 != None:
        tactile1 = TactileSensorClass(port_sensor_1, sensor_nr=1, interference_path=".../sensor1_fit_parameters.csv") # file for magnetic cross inteference
        thread1 = threading.Thread(target=tactile1.run)
        thread1.start()

    if port_sensor_2 != None:
        tactile2 = TactileSensorClass(port_sensor_2, sensor_nr=2, interference_path=".../sensor2_fit_parameters.csv") # file for magnetic cross inteference
        thread2 = threading.Thread(target=tactile2.run)
        thread2.start()

    rospy.spin()
    if port_sensor_1 != None:
        thread1.join()
    if port_sensor_2 != None:
        thread2.join()
    
    
   
    

    
    
