#include <Arduino.h>
#include <SPI.h>
#include "SPI_PAW3335.h"
#include "hall_sensor.h"

const String sensor_name = "sensor_1";

#define MOSI_PIN 7
#define MISO_PIN 8
#define SCLK_PIN 4
#define CS_PIN_1   5 // 5, 6 xs
#define CS_PIN_2   6 // 5, 6 xs

PAW3335 sensor(2000000, SCLK_PIN, MISO_PIN, MOSI_PIN, CS_PIN_1, CS_PIN_2);

#define SCLK_PIN_hall 37
#define MISO_PIN_hall 35
#define MOSI_PIN_hall 36
#define A0_PIN_hall 12
#define A1_PIN_hall 11
#define A2_PIN_hall 10
#define A3_PIN_hall 9

HALL_SENSOR hall_sensor(1000000, SCLK_PIN_hall, MISO_PIN_hall, MOSI_PIN_hall, A0_PIN_hall, A1_PIN_hall, A2_PIN_hall, A3_PIN_hall);

bool handshake = false;

void setup() {
  pinMode(6, OUTPUT);
  digitalWrite(6, HIGH);
  // Start the Serial Monitor
  Serial.begin(115200);
  delay(2000);
  Serial.println("Init mouse sensors");
  sensor.init();
  sensor.start_up();
  Serial.println("Init hall sensors");
  hall_sensor.init();
  Serial.println("setup done");
}



void readSerial(String& msg){
  while (Serial.available()) {
    //handshake = false; 
    // get the new byte:
    char ch = Serial.read();
    msg += ch;
    // end of user input
    if (ch == '\n') {
      if (msg[0] == 'n'){
        handshake = true;
      }else{
        handshake = false;
      }
      msg = "";
      return;
    }
  }
}

struct Packet {
  int16_t dx0, dy0;
  int16_t dx1, dy1;
  int16_t hall[13][4]; // x,y,z,T
};

String msg = "";


int rate_hz = 250;   // <-- change here
int period_ms = 1000 / rate_hz;

void loop() {
  static unsigned long last = 0;
  unsigned long now = millis();
  if (now - last < period_ms) return;
  last = now;
  
  Packet pkt;
  auto d0 = sensor.read_delta(0);
  auto d1 = sensor.read_delta(1);
  pkt.dx0 = d0.dx;
  pkt.dy0 = d0.dy;
  pkt.dx1 = d1.dx;
  pkt.dy1 = d1.dy;

  auto hall_data = hall_sensor.read_all_sensors();
  for (int i=0; i<13; i++) {
    pkt.hall[i][0] = hall_data[i].x;
    pkt.hall[i][1] = hall_data[i].y;
    pkt.hall[i][2] = hall_data[i].z;
    pkt.hall[i][3] = hall_data[i].T;
  }
  if (handshake == false){
    Serial.write(0xAA);
    Serial.write((uint8_t*)&pkt, sizeof(pkt));
  }else{
    // send name of sensor 
    Serial.println(""); 
    Serial.print("n");
    Serial.print("/t");
    Serial.println(sensor_name); 
  }
    
  readSerial(msg);
  
}