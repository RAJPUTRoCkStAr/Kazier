import cv2
import numpy as np
import requests

class Utils:
    def __init__(self):
        self.color_dict = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'yellow': (0, 255, 255),
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255),
            'gray': (128, 128, 128),
            'light_gray': (211, 211, 211),
            'dark_gray': (169, 169, 169),
            'orange': (0, 165, 255),
            'pink': (203, 192, 255),
            'purple': (128, 0, 128),
            'brown': (42, 42, 165),
            'lime': (0, 255, 0),
            'gold': (0, 215, 255),
            'silver': (192, 192, 192),
            'navy': (128, 0, 0),
            'teal': (128, 128, 0)
        }

        self.font_dict = {
            'hershey_simplex': cv2.FONT_HERSHEY_SIMPLEX,
            'hershey_plain': cv2.FONT_HERSHEY_PLAIN,
            'hershey_duplex': cv2.FONT_HERSHEY_DUPLEX,
            'hershey_complex': cv2.FONT_HERSHEY_COMPLEX,
            'hershey_triplex': cv2.FONT_HERSHEY_TRIPLEX,
            'hershey_complex_small': cv2.FONT_HERSHEY_COMPLEX_SMALL,
            'hershey_script_simplex': cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            'hershey_script_complex': cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
            'italic': cv2.FONT_ITALIC,
            'hershey_simplex_bold': cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC,
            'hershey_plain_bold': cv2.FONT_HERSHEY_PLAIN | cv2.FONT_ITALIC
        }

    def download_image_from_url(self, url):
        response = requests.get(url)
        image = np.array(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Error decoding the image from the URL.")
        return image

    def make_background_black(self, image):
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        black = np.zeros_like(image)
        black[:] = 0
        return black

    def rotate_image(self, image, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH))

    def resize_to_same_height(self, img1, img2):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        if h1 != h2:
            new_w = int(w2 * (h1 / h2))
            img2 = cv2.resize(img2, (new_w, h1))
        return img1, img2

    def resize_to_same_width(self, img1, img2):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        if w1 != w2:
            new_h = int(h2 * (w1 / w2))
            img2 = cv2.resize(img2, (w1, new_h))
        return img1, img2

    def hstack_images(self, img1, img2):
        img1, img2 = self.resize_to_same_height(img1, img2)
        return np.hstack((img1, img2))

    def vstack_images(self, img1, img2):
        img1, img2 = self.resize_to_same_width(img1, img2)
        return np.vstack((img1, img2))

    def detect_color(self, image, color_name):
        if color_name in self.color_dict:
            lower_bound, upper_bound = self.get_color_bounds(color_name)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            return mask
        else:
            raise ValueError(f"Color {color_name} not found in color dictionary.")

    def get_color_bounds(self, color_name):
        color_ranges = {
            'red': [(0, 50, 50), (10, 255, 255)],
            'green': [(36, 25, 25), (70, 255, 255)],
            'blue': [(94, 80, 2), (126, 255, 255)],
            'yellow': [(22, 93, 0), (45, 255, 255)],
            'cyan': [(78, 158, 124), (138, 255, 255)],
            'magenta': [(125, 50, 50), (175, 255, 255)],
            'white': [(0, 0, 200), (180, 20, 255)],
            'gray': [(0, 0, 40), (180, 20, 200)],
            'black': [(0, 0, 0), (180, 255, 30)],
            'orange': [(10, 100, 20), (25, 255, 255)],
            'pink': [(160, 50, 50), (180, 255, 255)],
            'purple': [(129, 50, 70), (158, 255, 255)],
            'brown': [(10, 100, 20), (20, 255, 200)],
            'lime': [(29, 86, 6), (64, 255, 255)],
            'gold': [(24, 100, 100), (32, 255, 255)],
            'silver': [(0, 0, 192), (180, 18, 255)],
            'navy': [(100, 100, 50), (140, 255, 255)],
            'teal': [(85, 50, 50), (100, 255, 255)]
        }
        return color_ranges.get(color_name, [(0, 0, 0), (0, 0, 0)])

    def detect_corners(self, image, block_size=2, ksize=3, k=0.04):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.cornerHarris(gray, block_size, ksize, k)
        corners = cv2.dilate(corners, None)
        image[corners > 0.01 * corners.max()] = [0, 0, 255] 
        return image

    def add_text(self, image, text, position, font_name='hershey_simplex', font_scale=1, color_name='white', thickness=2, align='left'):
        color = self.color_dict.get(color_name, (255, 255, 255))
        font = self.font_dict.get(font_name, cv2.FONT_HERSHEY_SIMPLEX)
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        if align == 'center':
            position = (position[0] - text_width // 2, position[1] + text_height // 2)
        elif align == 'right':
            position = (position[0] - text_width, position[1] + text_height // 2)
        elif align == 'left':
            position = (position[0], position[1] + text_height // 2)
        
        return cv2.putText(image, text, position, font, font_scale, color, thickness)

# Usage
def main():
    utils = Utils()
    image_url = 'https://image.shutterstock.com/image-vector/dotted-spiral-vortex-royaltyfree-images-600w-2227567913.jpg'  # Replace with the actual image URL
    image = utils.download_image_from_url(image_url)
    black_background_image = utils.make_background_black(image)
    rotated_image = utils.rotate_image(image, 45)
    img2 = cv2.imread('med/ig.jpg')  # Load another image for stacking
    hstacked_image = utils.hstack_images(image, img2)
    vstacked_image = utils.vstack_images(image, img2)
    detected_color = utils.detect_color(image, 'green')
    image_with_corners = utils.detect_corners(image)
    image_with_text_left = utils.add_text(image, 'Hello World', (50, 50), font_name='hershey_triplex', color_name='blue', align='left')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()