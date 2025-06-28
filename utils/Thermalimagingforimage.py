import cv2
import numpy as np
from Thermalimaging import generateColourMap

# Use raw string to fix Windows path issues
img_pth = r"C:\Users\raiom.LAPTOP-59QT21KS\Computer Vision\leopard.png"
og_img = cv2.imread(img_pth, cv2.IMREAD_UNCHANGED)

if og_img is None:
    print("Image not loaded. Check the file path.")
    exit()

colorMap = generateColourMap()

# Fix typo in tileGridSize
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Resize or remove border if needed (optional)
frame_roi = og_img  # or og_img[:-3, :] if needed

# Normalize image to 8-bit range
normed = cv2.normalize(frame_roi, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Convert to grayscale for CLAHE
gray = cv2.cvtColor(normed, cv2.COLOR_BGR2GRAY)

# Apply CLAHE
cl1 = clahe.apply(gray)

# Convert back to 3 channels to apply color map
nor = cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)

# Apply custom thermal-style color map
colorized_img = cv2.LUT(nor, colorMap)

output_path = r"C:\Users\raiom.LAPTOP-59QT21KS\Computer Vision\thermal_leopard.png"
cv2.imwrite(output_path, colorized_img)

# Display
cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
cv2.imshow("preview", cv2.resize(colorized_img, dsize=(640, 480), interpolation=cv2.INTER_LINEAR))
cv2.waitKey(0)
cv2.destroyAllWindows()
