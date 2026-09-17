#include <Arduino.h>
#include <SPI.h>
#include "SPI_PAW3335.h"

PAW3335::PAW3335(int spi_hz, int SCLK_PIN_, int MISO_PIN_, int MOSI_PIN_, int CS_PIN_1, int CS_PIN_2)
    :
    //spi3(HSPI),
    spiSettings(spi_hz, MSBFIRST, SPI_MODE0),
    SCLK_PIN(SCLK_PIN_),
    MISO_PIN(MISO_PIN_),
    MOSI_PIN(MOSI_PIN_),
    CS_PIN{CS_PIN_1, CS_PIN_2},
    Hspi(FSPI)
{
   
}

void PAW3335::init(){
     // Initialize the SPI bus using custom pins
    Serial.println(MOSI_PIN);
    Hspi.begin(SCLK_PIN, MISO_PIN, MOSI_PIN, CS_PIN[0]);  // SPI pins: SCK, MISO, MOSI, CS

    // Set CS pin as output
    pinMode(CS_PIN[0], OUTPUT);
    digitalWrite(CS_PIN[0], HIGH);  // Ensure chip select is high (inactive) at startup

    pinMode(CS_PIN[1], OUTPUT);
    digitalWrite(CS_PIN[1], HIGH);  // Ensure chip select is high (inactive) at startup
    delay(50);
}

void PAW3335::start_up(){
    /*Power-up sequence

    1. Apply power to VDD (Happens automatically)
    2. Wait for 50 ms
    3. Drive NCS high, and then low to reset the SPI port.
    4. Write 0x5A to Power_Up_Reset (0x3A) register.
    5. Wait for at least 5ms.
    6. Load Power‐up initialization register setting.
    7. Read from registers 0x02, 0x03, 0x04, 0x05 and 0x06 one time regardless of the motion bit state.

    This sensor has a 16 bit motion delta registers, i.e. Delta_X is made from Delta_X_L and Delta_X_H.
    */
    Serial.println("start sensor 1");
    digitalWrite(CS_PIN[0], LOW);
    delay(50);
    digitalWrite(CS_PIN[0], HIGH);
    delay(50);
    digitalWrite(CS_PIN[0], LOW);
    Hspi.beginTransaction(spiSettings);
    write_(0x3A, 0x5A);
    delay(5);
    LoadPower_upInit();
    digitalWrite(CS_PIN[0], HIGH);
    Serial.println("sensor 1 running");

    Serial.println("start sensor 2");
    digitalWrite(CS_PIN[1], LOW);

    delay(50);
    digitalWrite(CS_PIN[1], HIGH);
    delay(50);
    digitalWrite(CS_PIN[1], LOW);
    write_(0x3A, 0x5A);
    delay(5);
    LoadPower_upInit();
    digitalWrite(CS_PIN[1], HIGH);
    Serial.println("sensor 2 running");
    //spi1.endTransaction();
    
}

void PAW3335::write_(uint8_t registerAddress, uint8_t dataToWrite){
    delayMicroseconds(6);
    byte registerAddress_ = registerAddress;
    registerAddress_ |= 0x80; // make first bit a one
    Hspi.transfer(registerAddress_);         // Write the register address
    Hspi.transfer(dataToWrite);    
    delayMicroseconds(3);
}

PAWdxdy PAW3335::read_delta(int sensor_nr){
    digitalWrite(CS_PIN[sensor_nr], LOW);
    delayMicroseconds(3);
    byte d = read_(0x02);
    byte dx_l = read_(0x03);
    byte dx_h = read_(0x04);
    byte dy_l = read_(0x05);
    byte dy_h = read_(0x06);
    int16_t dx = (int16_t)((dx_h << 8) | dx_l);
    int16_t dy = (int16_t)((dy_h << 8) | dy_l);
    PAWdxdy delta;
    delta.dx = dx;
    delta.dy = dy;
    digitalWrite(CS_PIN[sensor_nr], HIGH);
    return delta;
}

byte PAW3335::read_(uint8_t registerAddress){
    delayMicroseconds(3);
    // Ensure MSB is cleared for read operation 
    registerAddress &= 0x7F;
    byte receivedData_ = Hspi.transfer(registerAddress);
    delayMicroseconds(3);
    byte receivedData = Hspi.transfer(0x00);
    
    return receivedData;
}

void PAW3335::LoadPower_upInit(){
    write_(0x40, 0x80);  // 1. Write register 0x40 (Performance) with value 0x80 (AWAKE| Disable Rest Mode)
    write_(0x55, 0x01); // 2. Write register 0x55 (unknown) with value 0x01 (unknown)
    delay(2);   // 3. Wait for at least 1ms
    write_(0x7F, 0x0E);    // 4. Write register 0x7F (unknown) with value 0x0E (unknown)
    write_(0x43, 0x1D);     // 5. Write register 0x43 (unknown) with value 0x1D (unknown)
    byte R1 = read_(0x46);   // 6. Read register 0x46 (unknown) and store in Var“R1” (unknown)
    Serial.print("R1 ");
    Serial.println(R1);
    write_(0x43, 0x1E);     // 7. Write register 0x43 (unknown) with value 0x1E (unknown)
    byte R2 = read_(0x46);     // 8. Read register 0x46 (unknown) and store in Var“R2” (unknown)
    Serial.print("R2 ");
    Serial.println(R2);
    write_(0x7F, 0x14);     // 9. Write register 0x7F (unknown) with value 0x14 (unknown)
    write_(0x6A, R1);     // 10. Write register 0x6A (unknown) with value “R1” (unknown)
    write_(0x6C, R2);     // 11. Write register 0x6C (unknown) with value “R2” (unknown)
    write_(0x7F, 0x00);     // 12. Write register 0x7F (unknown) with value 0x00 (unknown)
    write_(0x55, 0x00);     // 13. Write register 0x55 (unknown) with value 0x00 (unknown)
    write_(0x4E, 0x23);     // 14. Write register 0x4E (RESOLUTION) with value 0x23 (3000cpi)
    write_(0x77, 0x18);     // 15. Write register 0x77 (Run_DownShift) with value 0x18 (not sure)
    write_(0x7F, 0x05);     // 16. Write register 0x7F (unknown) with value 0x05 (unknown)
    write_(0x53, 0x0C);     // 17. Write register 0x53 (unknown) with value 0x0C (unknown)
    write_(0x5B, 0xEA);     // 18. Write register 0x5B (AXIS_CONTROL )with value 0xEA (swap and invert x and y)
    write_(0x61, 0x13);     // 19. Write register 0x61 (unknown) with value 0x13 (unknown)
    write_(0x62, 0x0B);     // 20. Write register 0x62 (unknown) with value 0x0B (unknown)
    write_(0x64, 0xD8);     // 21. Write register 0x64 (unknown) with value 0xD8 (unknown)
    write_(0x6D, 0x86);     // 22. Write register 0x6D (unknown) with value 0x86 (unknown)
    write_(0x7D, 0x84);     // 23. Write register 0x7D (RUN_DOWNSHIFT_MULT) with value 0x84 ()
    write_(0x7E, 0x00);     // 24. Write register 0x7E (REST_DOWNSHIFT_MULT) with value 0x00 (2)
    write_(0x7F, 0x06);     // 25. Write register 0x7F (unknown) with value 0x06 (unknown)
    write_(0x60, 0xB0);     // 26. Write register 0x60 (unknown) with value 0xB0 (unknown) 
    write_(0x61, 0x00);     // 27. Write register 0x61 (unknown) with value 0x00 (unknown)
    write_(0x7E, 0x40);     // 28. Write register 0x7E (REST_DOWNSHIFT_MULT) with value 0x40
    write_(0x7F, 0x0A);     // 29. Write register 0x7F (unknown) with value 0x0A (unknown)
    write_(0x4A, 0x23);     // 30. Write register 0x4A (unknown) with value 0x23 (unknown)
    write_(0x4C, 0x28);     // 31. Write register 0x4C (unknown) with value 0x28 (unknown)
    write_(0x49, 0x00);     // 32. Write register 0x49 (unknown) with value 0x00 (unknown)
    write_(0x4F, 0x02);     // 33. Write register 0x4F (unknown) with value 0x02 (unknown)
    write_(0x7F, 0x07);     // 34. Write register 0x7F (unknown) with value 0x07 (unknown)
    write_(0x42, 0x16);     // 35. Write register 0x42 (unknown) with value 0x16 (unknown)
    write_(0x7F, 0x09);     // 36. Write register 0x7F (unknown) with value 0x09 (unknown)
    write_(0x40, 0x03);     // 37. Write register 0x40 (PERFORMANCE) with value 0x03
    write_(0x7F, 0x0C);     // 38. Write register 0x7F (unknown) with value 0x0C (unknown)
    write_(0x54, 0x00);     // 39. Write register 0x54 (unknown) with value 0x00 (unknown)
    write_(0x44, 0x44);     // 40. Write register 0x44 (unknown) with value 0x44 (unknown)
    write_(0x56, 0x40);     // 41. Write register 0x56 (ANGLE_SNAP) with value 0x40 (not su)
    write_(0x42, 0x0C);     // 42. Write register 0x42 (unknown) with value 0x0C (unknown)
    write_(0x43, 0xA8);     // 43. Write register 0x43 (unknown) with value 0xA8 (unknown)
    write_(0x4E, 0x8B);     // 44. Write register 0x4E (RESOLUTION) with value 0x8B (not in list)
    write_(0x59, 0x63);     // 45. Write register 0x59 (RAW_DATA_GRAB_STATUS) with value 0x63 (unclear, usually read reg)
    write_(0x7F, 0x0D);     // 46. Write register 0x7F (unknown) with value 0x0D (unknown)
    write_(0x5E, 0xC3);     // 47. Write register 0x5E (unknown) with value 0xC3 (unknown)
    write_(0x4F, 0x02);     // 48. Write register 0x4F (unknown) with value 0x02 (unknown)
    write_(0x7F, 0x14);     // 49. Write register 0x7F (unknown) with value 0x14 (unknown)
    write_(0x4A, 0x67);     // 50. Write register 0x4A (unknown) with value 0x67 (unknown)
    write_(0x6D, 0x82);     // 51. Write register 0x6D (unknown) with value 0x82 (unknown)
    write_(0x73, 0x83);     // 52. Write register 0x73 (unknown) with value 0x83 (unknown)
    write_(0x74, 0x00);     // 53. Write register 0x74 (unknown) with value 0x00 (unknown)
    write_(0x7A, 0x16);     // 54. Write register 0x7A (REST2_PERIOD) with value 0x16
    write_(0x63, 0x14);     // 55. Write register 0x63 (unknown) with value 0x14 (unknown)
    write_(0x62, 0x14);     // 56. Write register 0x62 (unknown) with value 0x14 (unknown)
    write_(0x7F, 0x10);     // 57. Write register 0x7F (unknown) with value 0x10 (unknown)
    write_(0x48, 0x0F);     // 58. Write register 0x48 (unknown) with value 0x0F (unknown)
    write_(0x49, 0x88);     // 59. Write register 0x49 (unknown) with value 0x88 (unknown)
    write_(0x4C, 0x1D);     // 60. Write register 0x4C (unknown) with value 0x1D (unknown)
    write_(0x4F, 0x08);     // 61. Write register 0x4F (unknown) with value 0x08 (unknown)
    write_(0x51, 0x6F);     // 62. Write register 0x51 (unknown) with value 0x6F
    write_(0x52, 0x90);     // 63. Write register 0x52 (unknown) with value 0x90
    write_(0x54, 0x64);     // 64. Write register 0x54 (unknown) with value 0x64 (unknown)
    write_(0x55, 0xF0);     // 65. Write register 0x55 (unknown) with value 0XF0 (unknown)
    write_(0x5C, 0x40);     // 66. Write register 0x5C (unknown) with value 0x40
    write_(0x61, 0xEE);     // 67. Write register 0x61 (unknown) with value 0xEE (unknown)
    write_(0x62, 0xE5);     // 68. Write register 0x62 (unknown) with value 0xE5 (unknown)
    write_(0x7F, 0x00);     // 69. Write register 0x7F (unknown) with value 0x00 (unknown)
    write_(0x5B, 0x40);     // 70. Write register 0x5B (AXIS_CONTROL) with value 0x40 (invert y)
    write_(0x61, 0xAD);     // 71. Write register 0x61 (unknown) with value 0xAD (unknown)
    write_(0x51, 0xEA);     // 72. Write register 0x51 (unknown) with value 0xEA
    write_(0x19, 0x9F);     // 73. Write register 0x19  (unknown) with value 0x9F
// 74. Read register 0x20 at 1ms interval until 0x0F is obtained of read up to 55ms, this register read interval must be
// carried out at 1ms interval with timing tolerance of +/‐ 1%.
    int i = 0;
    while(i < 55)
    {   
        delayMicroseconds(1000);
        byte d = read_(0x0F);     //s
        Serial.print("D ");
        Serial.println(d);
        /* code */
        i++;
    }

    write_(0x19, 0x10); // 75. Write register 0x19 (unknown)  with value 0x10
    write_(0x61, 0xD5);  // 76. Write register 0x61 (unknown) with value 0xD5 (unknown)
    write_(0x40, 0x00); // 77. Write register 0x40 (PERFORMANCE) with value 0x00 (Enable Rest Mode)
    write_(0x7F, 0x00);  // 78. Write register 0x7F (unknown) with value 0x00 (unknown)
    byte d = read_(0x02);
    d = read_(0x03);
    d = read_(0x04);
    d = read_(0x05);
    d = read_(0x06);    
    // set resolution to 4900 cpi
    write_(0x4E, 0x3A);
    // ripple control
    //write_(0x5A, 0x90);

    delayMicroseconds(10);
    
    // lif cut off 2mm 
    write_(0x7F, 0x0C);
    write_(0x40, 0x14);
    write_(0x41, 0x14);
    write_(0x42, 0x20);
    write_(0x43, 0x18);
    write_(0x44, 0xC7);
    write_(0x45, 0x05);
    write_(0x4A, 0x0A);
    write_(0x4B, 0x08);
    write_(0x4C, 0x45);
    write_(0x4E, 0x0F);
    write_(0x54, 0x00);
    write_(0x56, 0x2A);
    write_(0x59, 0x93);
    write_(0x6D, 0x5F);
    write_(0x7F, 0x00);

    delayMicroseconds(10);
    // Force awake
    write_(0x40, 0x80);
}










