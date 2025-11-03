#!/usr/bin/env python3
"""
Smart Locking System - Wiring Diagram Generator

This script generates a wiring diagram PNG showing the hardware circuit connections
for the smart locking system including:
- Microcontroller (ESP32/Arduino)
- Solenoid Actuator
- Sensors (Magnetic, Motion, Pressure)
- Power Supply
- Communication Modules (WiFi, Bluetooth)
"""

from PIL import Image, ImageDraw, ImageFont
import os

def generate_wiring_diagram():
    # Image dimensions
    width, height = 1200, 900
    background_color = (240, 240, 245)  # Light gray-blue
    
    # Create a new image
    img = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(img)
    
    # Define colors
    color_black = (0, 0, 0)
    color_red = (220, 53, 69)
    color_green = (40, 167, 69)
    color_blue = (0, 123, 255)
    color_yellow = (255, 193, 7)
    color_gray = (128, 128, 128)
    color_box = (230, 230, 250)
    
    # Try to use a default font
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        title_font = label_font = small_font = ImageFont.load_default()
    
    # Title
    title = "Smart Locking System - Hardware Wiring Diagram"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (width - title_width) // 2
    draw.text((title_x, 20), title, fill=color_black, font=title_font)
    
    # Draw main components
    # Central Microcontroller
    mcu_x, mcu_y = 600, 450
    mcu_width, mcu_height = 120, 100
    draw.rectangle(
        [mcu_x - mcu_width//2, mcu_y - mcu_height//2, 
         mcu_x + mcu_width//2, mcu_y + mcu_height//2],
        fill=color_box, outline=color_blue, width=3
    )
    draw.text((mcu_x - 45, mcu_y - 20), "MICROCONTROLLER", fill=color_blue, font=label_font)
    draw.text((mcu_x - 40, mcu_y + 10), "(ESP32/Arduino)", fill=color_blue, font=small_font)
    
    # Power Supply (Left Top)
    ps_x, ps_y = 150, 150
    draw.rectangle(
        [ps_x - 50, ps_y - 40, ps_x + 50, ps_y + 40],
        fill=color_box, outline=color_red, width=3
    )
    draw.text((ps_x - 40, ps_y - 20), "POWER SUPPLY", fill=color_red, font=label_font)
    draw.text((ps_x - 35, ps_y + 10), "(12V/5V)", fill=color_red, font=small_font)
    
    # Solenoid Actuator (Right Top)
    act_x, act_y = 1050, 150
    draw.rectangle(
        [act_x - 50, act_y - 40, act_x + 50, act_y + 40],
        fill=color_box, outline=color_green, width=3
    )
    draw.text((act_x - 50, act_y - 20), "SOLENOID ACTUATOR", fill=color_green, font=label_font)
    draw.text((act_x - 40, act_y + 10), "(Lock Control)", fill=color_green, font=small_font)
    
    # Magnetic Sensor (Left)
    ms_x, ms_y = 150, 450
    draw.rectangle(
        [ms_x - 45, ms_y - 35, ms_x + 45, ms_y + 35],
        fill=color_box, outline=color_yellow, width=2
    )
    draw.text((ms_x - 45, ms_y - 18), "MAGNETIC SENSOR", fill=color_yellow, font=small_font)
    draw.text((ms_x - 35, ms_y + 5), "(Door Status)", fill=color_yellow, font=small_font)
    
    # Motion Sensor (Left Bottom)
    mot_x, mot_y = 150, 750
    draw.rectangle(
        [mot_x - 45, mot_y - 35, mot_x + 45, mot_y + 35],
        fill=color_box, outline=color_yellow, width=2
    )
    draw.text((mot_x - 40, mot_y - 18), "MOTION SENSOR", fill=color_yellow, font=small_font)
    draw.text((mot_x - 35, mot_y + 5), "(PIR Detector)", fill=color_yellow, font=small_font)
    
    # WiFi Module (Right)
    wifi_x, wifi_y = 1050, 450
    draw.rectangle(
        [wifi_x - 45, wifi_y - 35, wifi_x + 45, wifi_y + 35],
        fill=color_box, outline=color_blue, width=2
    )
    draw.text((wifi_x - 40, wifi_y - 18), "WiFi MODULE", fill=color_blue, font=small_font)
    draw.text((wifi_x - 30, wifi_y + 5), "(ESP-NOW)", fill=color_blue, font=small_font)
    
    # Bluetooth Module (Right Bottom)
    ble_x, ble_y = 1050, 750
    draw.rectangle(
        [ble_x - 50, ble_y - 35, ble_x + 50, ble_y + 35],
        fill=color_box, outline=color_blue, width=2
    )
    draw.text((ble_x - 45, ble_y - 18), "BLUETOOTH MODULE", fill=color_blue, font=small_font)
    draw.text((ble_x - 35, ble_y + 5), "(BLE/HC-05)", fill=color_blue, font=small_font)
    
    # Draw connections (wiring lines)
    line_width = 2
    
    # Power connections
    draw.line([ps_x, ps_y + 40, mcu_x - 60, mcu_y], fill=color_red, width=line_width)
    draw.text(((ps_x + mcu_x - 60)//2 - 30, ((ps_y + 40 + mcu_y)//2)), "+12V", fill=color_red, font=small_font)
    
    draw.line([ps_x, ps_y + 40, act_x, act_y + 40], fill=color_red, width=line_width)
    
    # Ground connections
    draw.line([ps_x + 15, ps_y + 40, mcu_x - 30, mcu_y + 50], fill=color_black, width=line_width)
    draw.text(((ps_x + 15 + mcu_x - 30)//2 - 20, ((ps_y + 40 + mcu_y + 50)//2)), "GND", fill=color_black, font=small_font)
    
    draw.line([ps_x + 15, ps_y + 40, act_x + 30, act_y + 40], fill=color_black, width=line_width)
    
    # MCU to Solenoid
    draw.line([mcu_x + 60, mcu_y - 20, act_x - 50, act_y], fill=color_green, width=line_width)
    draw.text(((mcu_x + 60 + act_x - 50)//2 - 30, ((mcu_y - 20 + act_y)//2) - 15), "Control", fill=color_green, font=small_font)
    
    # MCU to Magnetic Sensor
    draw.line([mcu_x - 60, mcu_y, ms_x + 45, ms_y], fill=color_yellow, width=line_width)
    draw.text(((mcu_x - 60 + ms_x + 45)//2 - 20, ((mcu_y + ms_y)//2) - 15), "Analog", fill=color_yellow, font=small_font)
    
    # MCU to Motion Sensor
    draw.line([mcu_x - 60, mcu_y + 30, mot_x + 45, mot_y], fill=color_yellow, width=line_width)
    draw.text(((mcu_x - 60 + mot_x + 45)//2 - 20, ((mcu_y + 30 + mot_y)//2) + 10), "GPIO", fill=color_yellow, font=small_font)
    
    # MCU to WiFi
    draw.line([mcu_x + 60, mcu_y - 30, wifi_x - 45, wifi_y - 20], fill=color_blue, width=line_width)
    draw.text(((mcu_x + 60 + wifi_x - 45)//2 - 15, ((mcu_y - 30 + wifi_y - 20)//2) - 15), "UART/SPI", fill=color_blue, font=small_font)
    
    # MCU to Bluetooth
    draw.line([mcu_x + 60, mcu_y + 30, ble_x - 50, ble_y], fill=color_blue, width=line_width)
    draw.text(((mcu_x + 60 + ble_x - 50)//2 - 20, ((mcu_y + 30 + ble_y)//2) + 10), "Serial", fill=color_blue, font=small_font)
    
    # Add legend
    legend_y = 820
    draw.text((20, legend_y), "Legend: Red=Power  Green=Control  Yellow=Sensors  Blue=Communication", fill=color_gray, font=small_font)
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'wiring_diagram.png')
    
    # Save the image
    img.save(output_path)
    print(f"Wiring diagram saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    generate_wiring_diagram()
