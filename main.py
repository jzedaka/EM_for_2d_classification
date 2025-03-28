import argparse
import cv2
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from em import EM
from time import time


def get_data(N=100, sigma=1, save=False, translations=True):
    """
    Generates syntetic data - an 64xy4 filled circle with added noise and random affine transformations
    """

    # image_path = r"C:\Users\jonathanz\Desktop\pin.png"
    # image_rgb = cv2.imread(image_path)
    S = 64
    cx, cy = S // 2, S // 2
    img = np.zeros((S, S))
    radius = 16
    # Fill the array
    for x in range(S):
        for y in range(S):
            # Compute squared distance from the center
            if (x - cx)**2 + (y - cy)**2 <= radius**2:
                img[x, y] = 1
    
    # plt.imshow(img, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # Get image dimensions
    h, w = img.shape

    images = np.zeros((N, h, w), dtype=np.float32)
    for i in range(N):
        angle = np.random.uniform(0, 360)
        scale = np.random.uniform(0.5, 1.5)
        # scale = 1
        tx = np.random.randn() * 0.05 * h
        ty = np.random.randn() * 0.05 * h
        M = cv2.getRotationMatrix2D(
            center=(w/2, h/2), angle=angle, scale=scale)

        # The translation component is stored in M[0,2] and M[1,2]
        if translations:
            M[0, 2] += tx
            M[1, 2] += ty

        # Apply the affine transformation
        transformed_img = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)

        # add noise
        transformed_img += np.random.randn(*transformed_img.shape) * sigma

        images[i] = transformed_img

        # plt.imshow(transformed_img, cmap='gray')
        # plt.axis('off')
        # plt.show()

    if save:
        with mrcfile.new(r'C:\Users\jonathanz\Desktop\input_images.mrcs', overwrite=True) as mrc:
            mrc.set_data(images)

    return images, img


if __name__ == "__main__":

    # Parse the command line inputs
    parser = argparse.ArgumentParser(description='EM arguments parser.')
    parser.add_argument('--mrc_stack_path', type=str, required=False, help='--mrc_stack_path <path to images>')
    sys_args, unknown = parser.parse_known_args()

    if sys_args.mrc_stack_path is not None:
        print("Reading images from file ...")
        with mrcfile.open(sys_args.mrc_stack_path, permissive=True) as mrc:
            images = mrc.data
    else:
        images, img = get_data(N=100, sigma=0.05)

    em = EM(img_shape=images[0].shape,
            rotation_res=5,
            scale_res=0.2,
            trans_res=3,
            max_iter=10,
            verbose=True)

    t0 = time()
    rec_img = em.recover_img(images)

    t1 = time()
    print(f"runtime = {(t1 - t0) / 60} min")
    plt.imshow(rec_img, cmap='gray')
    plt.axis('off')
    plt.show()
    print('Done')
