#pragma once
#include <Arduino.h>
#include <SPI.h>

// Commands from datasheet
#define CMD_SM  0x3F  // Start Single Measurement (XYZ + Temp)
#define CMD_RM  0x4F  // Read Measurement Result
#define CMD_EX  0x80  // Exit mode
#define CMD_RT  0xF0  // Reset


struct HallCs {
    int A0;
    int A1;
    int A2;
    int A3; 
};

struct hall_data
{
    float x;
    float y;
    float z;
    float T;
};

struct hall_data_int_16
{
    int16_t x;
    int16_t y;
    int16_t z;
    int16_t T;
};


class HALL_SENSOR{
    private:
        int SCLK_PIN; int MISO_PIN; int MOSI_PIN; int A0_PIN; int A1_PIN; int A2_PIN; int A3_PIN;
        std::array<HallCs, 14> hallList;  
        SPIClass Vspi; 
        SPISettings spiSettings;
        void setHallCS(int cs);
        void start_up(int cs);
        void sendCommand(uint8_t *cmd, uint8_t cmdLength, uint8_t *response, uint8_t respLength, int cs);

    public:
        HALL_SENSOR(int spi_hz, int SCLK_PIN_, int MISO_PIN_, int MOSI_PIN_, int A0_PIN, int A1_PIN, int A2_PIN, int A3_PIN);
        void init();
        std::array<hall_data_int_16, 13> read_all_sensors();

};



