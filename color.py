import cv2
import numpy as np
import pyperclip
import colorsys

# Function to convert RGB to CMYK
def rgb_to_cmyk(r, g, b):
    if (r == 0) and (g == 0) and (b == 0):
        return 0, 0, 0, 100
    c = 1 - r / 255.
    m = 1 - g / 255.
    y = 1 - b / 255.
    min_cmy = min(c, m, y)
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy
    return int(c * 100), int(m * 100), int(y * 100), int(k * 100)

def generate_shades_and_tints(base_color_hex):
    def hex_to_rgb(hex_color):
        # 將十六進制顏色碼轉換為 RGB
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def rgb_to_hex(rgb_color):
        # 將 RGB 轉換為十六進制顏色碼
        return '#{:02x}{:02x}{:02x}'.format(rgb_color[0], rgb_color[1], rgb_color[2])

    def generate_shades_and_tints2(base_color_hex, num_shades=3, num_tints=3):
        # 將十六進制顏色碼轉換為 RGB
        base_color_rgb = hex_to_rgb(base_color_hex)
        
        # 轉換為 HSV 以進行顏色調整
        base_color_hsv = colorsys.rgb_to_hsv(base_color_rgb[0] / 255.0, base_color_rgb[1] / 255.0, base_color_rgb[2] / 255.0)
        
        shades = []
        tints = []
        
        # 生成深色
        for i in range(1, num_shades + 1):
            dark_value = base_color_hsv[2] * (1 - i * 0.1)  # 調整深色的亮度
            dark_value = max(0, min(1, dark_value))  # 確保不超出範圍
            dark_rgb = colorsys.hsv_to_rgb(base_color_hsv[0], base_color_hsv[1], dark_value)
            dark_rgb = tuple(int(c * 255) for c in dark_rgb)
            shades.append(rgb_to_hex(dark_rgb))
        
        # 生成淺色
        for i in range(1, num_tints + 1):
            light_value = base_color_hsv[2] * (1 + i * 0.1)  # 調整淺色的亮度
            light_value = max(0, min(1, light_value))  # 確保不超出範圍
            light_rgb = colorsys.hsv_to_rgb(base_color_hsv[0], base_color_hsv[1], light_value)
            light_rgb = tuple(int(c * 255) for c in light_rgb)
            tints.append(rgb_to_hex(light_rgb))
        
        return shades, tints

    # 輸入的基礎顏色
    # base_color_hex = '#b39728'

    # 生成三個深色和三個淺色
    shades, tints = generate_shades_and_tints2(base_color_hex, num_shades=3, num_tints=3)
    shades = shades[::-1]
    image = np.zeros((100, 700, 3), dtype=np.uint8)

    # 填充原始顏色方塊
    image[:, 300:400] = hex_to_rgb(base_color_hex)[::-1]

    # 填充淺色方塊
    for i, color in enumerate(tints):
        image[:, 400+i*100:500+i*100] = hex_to_rgb(color)[::-1]

    # 填充深色方塊
    for i, color in enumerate(shades):
        image[:, i*100:(i+1)*100] = hex_to_rgb(color)[::-1]

    return image, shades, tints

# Function to handle mouse events on the color plate
def color_plate_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        color = param[y, x].tolist()  # BGR format
        color_rgb = color[::-1]  # Convert to RGB for displaying
        color_hex = '#{:02x}{:02x}{:02x}'.format(color_rgb[0], color_rgb[1], color_rgb[2])
        pyperclip.copy(color_hex)
        print(f"Color {color_rgb} (RGB), {color_hex} (Hex) copied to clipboard.")

# Function to handle mouse events on the camera feed
def mouse_callback(event, x, y, flags, param):
    global frame, pixel_selected, color_preview, shades, tints, color_plate_image

    if event == cv2.EVENT_LBUTTONDOWN:
        color = frame[y, x].tolist()  # BGR format
        color_rgb = color[::-1]  # Convert to RGB for displaying
        color_preview[:] = color  # Update the color preview window

        color_hex = '#{:02x}{:02x}{:02x}'.format(color_rgb[0], color_rgb[1], color_rgb[2])
        pyperclip.copy(color_hex)

        # Generate color plate
        color_plate_image, shades, tints = generate_shades_and_tints(color_hex)
        cv2.imshow("Color Plate", color_plate_image)

        c, m, y_cmyk, k = rgb_to_cmyk(color_rgb[0], color_rgb[1], color_rgb[2])
        cmyk_str = f"CMYK({c}%, {m}%, {y_cmyk}%, {k}%)"

        print(f"Pixel selected at: ({x}, {y}) with color: {color} (BGR), {color_rgb} (RGB), {color_hex} (Hex), {cmyk_str}")
        pixel_selected = (x, y)

        cv2.drawMarker(frame, (x, y), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

# Initialize camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

cv2.namedWindow('Camera')
cv2.setMouseCallback('Camera', mouse_callback)

frame = None
pixel_selected = None
color_preview = np.zeros((100, 100, 3), np.uint8)
shades, tints = [], []
color_plate_image = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Resize frame for display
    display_frame = cv2.resize(frame, (640, 480))  # Adjust to your preferred display size
    
    if pixel_selected:
        cv2.drawMarker(display_frame, pixel_selected, color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        
    cv2.imshow('Camera', display_frame)
    cv2.imshow('Color Preview', color_preview)

    if color_plate_image is not None:
        cv2.imshow("Color Plate", color_plate_image)
        cv2.setMouseCallback("Color Plate", color_plate_callback, color_plate_image)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
