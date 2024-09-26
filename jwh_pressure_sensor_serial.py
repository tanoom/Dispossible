import serial
import time

class PressureSensor:
    def __init__(self, port="COM5", baud_rate=115200):
        self.port = port
        self.baud_rate = baud_rate
        self.ser = serial.Serial(port, baud_rate, timeout=1)
        self.offset = self._calculate_offset()

    def _calculate_offset(self):
        offset = None
        while offset is None:
            initial_data = self.extract_number(self.read_pressure())
            if type(initial_data) == int:
                offset = initial_data / 4.67
                break
            time.sleep(0.1)
        print(f"Offset data: {offset}")
        return offset

    def read_pressure(self):
        try:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8').strip()
                return line
        except serial.SerialException as e:
            print(f"Error reading from serial port: {e}")
        return None

    def extract_number(self, s):
        if s is None:
            return None
        colon_pos = s.find(':')
        if colon_pos == -1:
            return None
        start_pos = colon_pos + 2
        number_str = ''
        for char in s[start_pos:]:
            if char.isdigit():
                number_str += char
            else:
                break
        return int(number_str) if number_str else None

    def get_real_mass(self):
        pressure_data = self.extract_number(self.read_pressure())
        if pressure_data:
            try:
                pressure_value = pressure_data / 4.67 - self.offset
                return pressure_value
            except ValueError:
                print(f"Received invalid data: {pressure_data}")
        return None

    def close(self):
        self.ser.close()
        print("Serial port closed. Program terminated.")

if __name__ == "__main__":
    sensor = PressureSensor()
    print("Press 'q' to stop the program.")
    try:
        while True:
            real_mass = sensor.get_real_mass()
            if real_mass is not None:
                print(f"Received pressure data (after offset): {real_mass}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        sensor.close()