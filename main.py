import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

from scipy.datasets import face
from em import EM


def get_data(N=100, sigma=1):
    # 1. Load the sample face (RGB) from SciPy
    # image_rgb = face()
    image_path = r"C:\Users\jonathanz\Desktop\pin.png"
    image_rgb = cv2.imread(image_path)
    # 2. Convert to 2D grayscale
    #    The result is a 2D array (height x width)
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # plt.imshow(img, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # Get image dimensions
    h, w = img.shape

    images = np.zeros((N, h, w))
    for i in range(N):
        angle = random.uniform(0, 360)
        scale = random.uniform(0.5, 1.5)
        # scale = 1
        tx = random.randint(-50, 50)
        ty = random.randint(-50, 50)
        M = cv2.getRotationMatrix2D(center=(w/2, h/2), angle=angle, scale=scale)

        # The translation component is stored in M[0,2] and M[1,2]
        # M[0,2] += tx
        # M[1,2] += ty

        # Apply the affine transformation
        transformed_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR).astype(np.float64)
        
        # add noise
        transformed_img += np.random.randn(*transformed_img.shape) * sigma
        
        images[i] = transformed_img
        # plt.imshow(transformed_img, cmap='gray')
        # plt.axis('off')
        # plt.show()
    
    return images


if __name__ == "__main__":
    
    images = get_data(N=10, sigma=10)
    em = EM(img_shape=images[0].shape, rotation_res=1, scale_res=0.2)
    img, sigma = em.recover_img(images)
    
    print(f"Done. Sigma = {sigma}")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()
