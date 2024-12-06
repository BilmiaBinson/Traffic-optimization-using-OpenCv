#include <Arduino.h>

const int LED_PINS[4][3] = {
  {2, 3, 4},   // North: Red, Yellow, Green
  {5, 6, 7},   // South: Red, Yellow, Green
  {8, 9, 10},  // East: Red, Yellow, Green
  {11, 12, 13} // West: Red, Yellow, Green
};

char directions[4] = {'N', 'S', 'E', 'W'};

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++) {
      pinMode(LED_PINS[i][j], OUTPUT);
    }
    digitalWrite(LED_PINS[i][0], HIGH);  // Start with all red
  }
}

void setAllRed() {
  for (int i = 0; i < 4; i++) {
    digitalWrite(LED_PINS[i][0], HIGH);  // Red on
    digitalWrite(LED_PINS[i][1], LOW);   // Yellow off
    digitalWrite(LED_PINS[i][2], LOW);   // Green off
  }
}

void controlTrafficLight(int dirIndex, int timing) {
  // Yellow for 3 seconds
  digitalWrite(LED_PINS[dirIndex][1], HIGH);  // Yellow on
  delay(3000);
  
  // Green for specified timing
  digitalWrite(LED_PINS[dirIndex][1], LOW);   // Yellow off
  digitalWrite(LED_PINS[dirIndex][2], HIGH);  // Green on
  delay(timing * 1000);
  
  // Yellow for 3 seconds
  digitalWrite(LED_PINS[dirIndex][2], LOW);   // Green off
  digitalWrite(LED_PINS[dirIndex][1], HIGH);  // Yellow on
  delay(3000);
  
  // Back to red
  digitalWrite(LED_PINS[dirIndex][1], LOW);   // Yellow off
  digitalWrite(LED_PINS[dirIndex][0], HIGH);  // Red on
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    char direction = input.charAt(0);
    int timing = input.substring(2).toInt();
    
    for (int i = 0; i < 4; i++) {
      if (directions[i] == direction) {
        setAllRed();  // Ensure all other directions are red
        controlTrafficLight(i, timing);
        break;
      }
    }
    
    // Signal Python that this cycle is complete
    Serial.println("DONE");
  }
}