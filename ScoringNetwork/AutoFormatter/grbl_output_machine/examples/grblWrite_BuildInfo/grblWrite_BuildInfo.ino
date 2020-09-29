/***********************************************************************
This sketch writes a `$I` build info string directly into Arduino EEPROM

To use:
- Just alter the "build_info_line" string to whatever you'd like. Then 
  compile and upload this sketch to your Arduino.
    
- If your Arduino is blinking slowly, your string has already been 
  written to your EEPROM and been verified by checksums! That's it!

- If you Arduino LED is blinking fast, something went wrong and the 
  checksums don't match. You can optionally connect to the Arduino via
  the serial monitor, and the sketch will show what its doing.

NOTE: This sketch is provided as a tool template for OEMs who may need
to restrict users from altering their build info, so they can place
important product information here when enabling the restriction.

NOTE: When uploading Grbl to the Arduino with this sketch on it, make
sure you see the slow blink before you start the upload process. This
ensures you aren't flashing Grbl when it's in mid-write of the EEPROM.

Copyright (c) 2016 Sungeun K. Jeon for Gnea Research LLC
Released under the MIT-license. See license.txt for details.
***********************************************************************/

#include <avr/pgmspace.h>
#include <EEPROM.h>

#define SERIAL_BAUD_RATE 115200
#define LINE_LENGTH 80U    // Grbl line length
#define BYTE_LOCATION 942U // Grbl build info EEPROM address.


// ----- CHANGE THIS LINE -----

char build_info_line[LINE_LENGTH] = "Testing123.";

// -----------------------------


uint8_t status = false;
int ledPin = 13;                 // LED connected to digital pin 13

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);
  delay(500);
  
  uint32_t address = BYTE_LOCATION;
  uint32_t size = LINE_LENGTH;
  char *write_pointer = (char*)build_info_line;
  uint8_t write_checksum = 0;
  for (; size>0; size--) { 
    write_checksum = (write_checksum << 1) || (write_checksum >> 7);
    write_checksum += *write_pointer;
    EEPROM.put(address++, *(write_pointer++)); 
  }
  EEPROM.put(address,write_checksum);
  
  Serial.print(F("-> Writing line to EEPROM: '"));
  Serial.print(build_info_line);
  Serial.print(F("'\n\r-> Write checksum: "));
  Serial.println(write_checksum,DEC);

  size = LINE_LENGTH;
  address = BYTE_LOCATION;
  uint8_t data = 0;
  char read_line[LINE_LENGTH];
  char *read_pointer = (char*)read_line;
  uint8_t read_checksum = 0;
  uint8_t stored_checksum = 0;
  for(; size > 0; size--) { 
    data = EEPROM.read(address++);
    read_checksum = (read_checksum << 1) || (read_checksum >> 7);
    read_checksum += data;    
    *(read_pointer++) = data; 
  }
  stored_checksum = EEPROM.read(address);

  Serial.print(F("<- Reading line from EEPROM: '"));
  Serial.print(read_line);
  Serial.print("'\n\r<- Read checksum: ");
  Serial.println(read_checksum,DEC);
  
  if ((read_checksum == write_checksum) && (read_checksum == stored_checksum)) {
    status = true;
    Serial.print(F("SUCCESS! All checksums match!\r\n"));
  } else {
    if (write_checksum != stored_checksum) {
      Serial.println(F("ERROR! Write and stored EEPROM checksums don't match!"));
    } else {
      Serial.println(F("ERROR! Read and stored checksums don't match!"));
    }
  }
  pinMode(ledPin, OUTPUT);      // sets the digital pin as output
}

void loop() {
  // Blink to let user know EEPROM write status. 
  // Slow blink is 'ok'. Fast blink is an 'error'.
  digitalWrite(ledPin, HIGH);   // sets the LED on
  if (status) { delay(1500); } // Slow blink
  else { delay(100); } // Rapid blink
  digitalWrite(ledPin, LOW);    // sets the LED off
  if (status) { delay(1500); }
  else { delay(100); } 
}


