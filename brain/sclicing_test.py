import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the grayscale image
image_path =  r"C:\Users\user\Desktop\CAPSTONE\brain\1a8106f3-e4ca-4e18-a275-065d020704f4.jpg"
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 2: Define a function to map grayscale intensities to pseudo-color
def map_intensity_to_color(intensity, num_shades=1000):
    """
    Maps grayscale intensity to a color from a spectrum with `num_shades` distinct colors.
    Intensity is in the range [0, 255].
    """
    # Normalize the intensity to the range [0, num_shades)
    normalized_intensity = int((intensity / 255) * (num_shades - 1))
    
    # Define a list of colors (from blue to red)
    color_map = [
        [0, 0, 255],    # Blue
        [0, 128, 255],  # Light Blue
        [0, 255, 255],  # Cyan
        [0, 255, 128],  # Light Green
        [0, 255, 0],    # Green
        [128, 255, 0],  # Yellow Green
        [255, 255, 0],  # Yellow
        [255, 128, 0],  # Orange
        [255, 0, 0],    # Red
        [128, 0, 0]     # Dark Red
    ]
    
    # Return the color corresponding to the normalized intensity
    return color_map[normalized_intensity]

# Step 3: Apply the pseudo-color to the entire image
pseudo_colored_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)

# Apply the color mapping for each pixel
for i in range(gray_image.shape[0]):
    for j in range(gray_image.shape[1]):
        intensity = gray_image[i, j]
        pseudo_colored_image[i, j] = map_intensity_to_color(intensity, num_shades=10)

# Step 4: Display the original and pseudo-colored images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(pseudo_colored_image)
plt.title('Pseudo-Colored Image with More Shades')
plt.axis('off')

plt.show()
