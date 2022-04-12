import time

import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1306

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# Raspberry Pi pin configuration:
RST = None     # on the PiOLED this pin isnt used
# Note the following are only used with SPI:
DC = 23
SPI_PORT = 0
SPI_DEVICE = 0

# 128x32 display with hardware I2C:
disp = Adafruit_SSD1306.SSD1306_128_32(rst=RST)

# Initialize library.
disp.begin()

# Clear display.
disp.clear()
disp.display()

# Create blank image for drawing.
# Make sure to create image with mode '1' for 1-bit color.
width = disp.width
height = disp.height
image = Image.new('1', (width, height))

# Get drawing object to draw on image.
draw = ImageDraw.Draw(image)

# First define some constants to allow easy resizing of shapes.
padding = -2
top = padding

# Move left to right keeping track of the current x position for drawing shapes.
x = 0

# Custom arial font
font = ImageFont.truetype('FontsFree-Net-arial-bold.ttf', 10)

# Draw outputs in the display
def displayText(condition):
    while True:
        # 4 different texts
        draw.text((x, top), "=====================",  font=font, fill=255)
        draw.text((x, top+10), "Lung Condition Detector",  font=font, fill=255)
        draw.text((x, top+18), condition,  font=font, fill=255)
        draw.text((x, top+27), "=====================",  font=font, fill=255)

        # Display image.
        disp.image(image)
        disp.display()
        time.sleep(.1)

