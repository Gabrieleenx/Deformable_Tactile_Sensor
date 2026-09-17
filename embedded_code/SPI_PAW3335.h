#pragma once
#include <Arduino.h>
#include <SPI.h>

struct PAWdxdy {
    float dx;
    float dy;
};

class PAW3335{
    private:
        int SCLK_PIN; int MISO_PIN; int MOSI_PIN; std::array<int, 2> CS_PIN;
        //SPIClass spi3; // page 21/22, seems like spi3 is the most free to use one
        SPIClass Hspi; 
        SPISettings spiSettings;
        void write_(uint8_t registerAddress, uint8_t dataToWrite);
        byte read_(uint8_t registerAddress);
        
    public:
        PAW3335(int spi_hz, int SCLK_PIN_, int MISO_PIN_, int MOSI_PIN_, int CS_PIN_1, int CS_PIN_2);
        void start_up();
        void LoadPower_upInit();
        void init();
        PAWdxdy read_delta(int sensor_nr);
};



