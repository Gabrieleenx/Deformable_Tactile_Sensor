#include <Arduino.h>
#include <SPI.h>
#include "hall_sensor.h"


HALL_SENSOR::HALL_SENSOR(int spi_hz, int SCLK_PIN_, int MISO_PIN_, int MOSI_PIN_, int A0_PIN, int A1_PIN, int A2_PIN, int A3_PIN):
    spiSettings(spi_hz, MSBFIRST, SPI_MODE3),
    SCLK_PIN(SCLK_PIN_),
    MISO_PIN(MISO_PIN_),
    MOSI_PIN(MOSI_PIN_),
    A0_PIN(A0_PIN),
    A1_PIN(A1_PIN),
    A2_PIN(A2_PIN),
    A3_PIN(A3_PIN),
    Vspi(HSPI)
    {
        hallList = {{
                        {0, 0, 0, 0},
                        {1, 0, 0, 0},
                        {1, 1, 0, 0},
                        {0, 1, 0, 0},
                        {1, 0, 1, 0},
                        {1, 1, 1, 0},
                        {0, 1, 1, 0},
                        {0, 0, 0, 1},
                        {1, 0, 0, 1},
                        {0, 1, 0, 1},
                        {1, 0, 1, 1},
                        {0, 1, 1, 1},
                        {0, 0, 1, 1},
                        {1, 1, 1, 1}    
                    }};

};
    
void HALL_SENSOR::setHallCS(int cs){
    digitalWrite(A0_PIN, hallList[cs].A0);
    digitalWrite(A1_PIN, hallList[cs].A1);
    digitalWrite(A2_PIN, hallList[cs].A2);
    digitalWrite(A3_PIN, hallList[cs].A3);

}

void HALL_SENSOR::sendCommand(uint8_t *cmd, uint8_t cmdLength, uint8_t *response, uint8_t respLength, int cs) {
  setHallCS(cs+1);
  delayMicroseconds(6);
  
  

  for (int i = 0; i < cmdLength; i++) {
    Vspi.transfer(cmd[i]);
  }

  delayMicroseconds(6);

  for (int i = 0; i < respLength; i++) {
    response[i] = Vspi.transfer(0x00); // dummy write to read
  }

  delayMicroseconds(6);
  
  setHallCS(0);
}



void HALL_SENSOR::start_up(int cs){
    uint8_t resp[10];
    // === Step 1: Reset ===
    uint8_t rt_cmd[1] = { 0xF0 };
    sendCommand(rt_cmd, 1, resp, 1, cs+1);
    Serial.print("RT Status: 0x"); Serial.println(resp[0], HEX);
    delay(5);

    // === Step 2: Set COMM_MODE = 0b10 (SPI only) in register 0x01 ===
    Serial.println("Setting COMM_MODE = SPI only...");
    // This sets bits [3:2] = 0b10 → 0x08 in LSB
    uint8_t wr_comm_mode[] = { 0x60, 0x00, 0x00, 0x08, 0x04 };  // WR, access=0x00, data=0x0008, addr=0x01 << 2
    sendCommand(wr_comm_mode, 5, resp, 1, cs+1);
    Serial.print("WR Status: 0x"); Serial.println(resp[0], HEX);
    delay(2);

    // === Step 3: Start Single Measurement for ZYXT (all set) ===
    Serial.println("Sending SM (Single Measurement)...");
    uint8_t sm_cmd[] = { 0x3F, 0x0F };  // SM + ZYXT = 1111b
    sendCommand(sm_cmd, 2, resp, 1, cs+1);
    Serial.print("SM Status: 0x"); Serial.println(resp[0], HEX);
    delay(5);  // Wait for measurement to complete
    Serial.print("Started");
    Serial.println(cs+1);
}

void HALL_SENSOR::init(){

    pinMode(A0_PIN, OUTPUT);
    pinMode(A1_PIN, OUTPUT);
    pinMode(A2_PIN, OUTPUT);
    pinMode(A3_PIN, OUTPUT);
    Vspi.begin(SCLK_PIN, MISO_PIN, MOSI_PIN, 13);  // SPI pins: SCK, MISO, MOSI, CS
    Vspi.beginTransaction(spiSettings);  // Safe choice: 1 MHz
    setHallCS(0);
    delay(50); // Wait for sensor startup
    for (int i = 0; i<13; i++){
        start_up(i);
    }
    Serial.println("hall started");

}


std::array<hall_data_int_16, 13> HALL_SENSOR::read_all_sensors(){
    uint8_t smCmd[2] = {CMD_SM, 0x0F}; // Start Single Measurement XYZ + Temp
    uint8_t rmCmd[2] = {CMD_RM, 0x0F}; // Read Measurement result

    // start measurment on all sensors 
    for (int i = 0; i<13; i++){
        uint8_t result[9] = {0};
        sendCommand(smCmd, 2, result, 1, i);
    }

    delayMicroseconds(1000); // Wait for measurement to complete

    std::array<hall_data_int_16, 13> sensor_data;

    // read measurment on all sesnors
    for (int i = 0; i<13; i++){
        uint8_t result[9] = {0};
        sendCommand(rmCmd, 2, result, 9, i);

        int16_t x = (int16_t)((result[2] << 8) | result[3]);
        int16_t y = (int16_t)((result[4] << 8) | result[5]);
        int16_t z = (int16_t)((result[6] << 8) | result[7]);
        uint16_t t_raw = (int16_t)((result[0] << 8) | result[1]);
        /*
        float x_uT = x * 0.150f;
        float y_uT = y * 0.150f;
        float z_uT = z * 0.242f;
        float temp_C = ((float)t_raw - 46244.0f) / 45.2f + 25.0;

        sensor_data[i].x = x_uT;    // example values
        sensor_data[i].y = y_uT;
        sensor_data[i].z = z_uT;
        sensor_data[i].T = temp_C;
        */
        sensor_data[i].x = x;    // example values
        sensor_data[i].y = y;
        sensor_data[i].z = z;
        sensor_data[i].T = t_raw;
    }

    return sensor_data;

}