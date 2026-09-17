#!/usr/bin/env python3

import struct, serial
from std_msgs.msg import Int16MultiArray
from sensor_v2.msg import TactileSensor, MouseSensor, HallSensor
import rospy
import serial
import serial.tools.list_ports
import copy
import numpy as np
from std_srvs.srv import Trigger, TriggerResponse



def find_esp32s3_port():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        rospy.loginfo(f"Found device: {port.device} - {port.description} - {port.hwid}")
        if 'ESP32' in port.description or \
           'VID:PID=303A:1001' in port.hwid.upper():
            rospy.loginfo(f"ESP32-S3 found on port: {port.device}")
            return port.device
    rospy.logwarn("ESP32-S3 device not found")
    return None








class TactileSensorClass:
    def __init__(self):
        rospy.init_node("tactile_reader")
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
        self.pub = rospy.Publisher("~tactile_sensor", TactileSensor, queue_size=10, tcp_nodelay=True)
        self.packet_size = 4*2 + 13*4*2  # int16 = 2 bytes
        self.reset_service = rospy.Service("~bias_sensor", Trigger, self.handle_reset_offset)

        
        self.ser = None
        self.read_thread = None
        self.running = True
        # Try to get port from param, else auto-detect
        port_param = rospy.get_param('~port', None)
        if port_param is None:
            port_param = find_esp32s3_port()
            if port_param is None:
                rospy.logerr("Cannot find ESP32-S3 port. Please specify port manually via ~port parameter.")
                rospy.signal_shutdown("No port found")
                return
        
        self.serial_port = port_param
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
        return TriggerResponse(success=True, message="Offset reset successful")

        
            
    def run(self):
        while not rospy.is_shutdown():
            # look for sync byte
            b = self.ser.read()
            if b != b'\xAA':
                continue
            pkt = self.ser.read(self.packet_size)
            if len(pkt) != self.packet_size:
                continue
            values = struct.unpack("<" + "h"*56, pkt)  # little-endian, 56 int16
            tactile_sensor = TactileSensor()
            tactile_sensor.header.stamp = rospy.Time.now()
            tactile_sensor.header.frame_id = "tactile_sensor"
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


                self.current_filtered_reading.hall_sensor[i].x = self.alpha * (self.current_reading.hall_sensor[i].x - self.offset.hall_sensor[i].x) + (1 - self.alpha) *self.previous_reading.hall_sensor[i].x
                self.current_filtered_reading.hall_sensor[i].y = self.alpha * (self.current_reading.hall_sensor[i].y - self.offset.hall_sensor[i].y) + (1 - self.alpha) *self.previous_reading.hall_sensor[i].y
                self.current_filtered_reading.hall_sensor[i].z = self.alpha * (self.current_reading.hall_sensor[i].z - self.offset.hall_sensor[i].z) + (1 - self.alpha) *self.previous_reading.hall_sensor[i].z
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
            self.pub.publish(tactile_sensor)



if __name__ == "__main__":
    

    try:
        tactile = TactileSensorClass()
        tactile.run()
    except rospy.ROSInterruptException:
        pass