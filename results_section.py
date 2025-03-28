import argparse
import cv2
import mrcfile
import numpy as np
from em import EM
from time import time
import matplotlib.pyplot as plt
from main import get_data


def calc_mse(img, rec_img):
    e = []
    for s in np.linspace(0.5, 1.5, 11):
        e.append(np.mean((img-s*rec_img)**2))
    return np.min(e)

def exp_1(M=10):
    Ns = np.array([5, 10, 30])
    sigmas = np.array([0.1, 1, 2])
    errors = np.zeros((Ns.size, sigmas.size, M))
    for i, n in enumerate(Ns):
        for j, sigma in enumerate(sigmas):
            for k in range(M):
                images, img = get_data(N=n, sigma=sigma, translations=False)

                em = EM(img_shape=images[0].shape,
                        rotation_res=5,
                        scale_res=0.2,
                        trans_res=3,
                        max_iter=10,
                        verbose=False)

                rec_img = em.recover_img(images)
                errors[i, j, k] = calc_mse(img=img, rec_img=rec_img)
                print(f'n = {n}, sigma = {sigma}, rep = {k}, err = {errors[i, j, k]}')

    plt.figure()

    for i, n in enumerate(Ns):
        means = np.mean(errors[i, :], axis=1)
        plt.plot(sigmas, means, '--o', label=f"n = {n}")
        stds = np.std(errors[i, :], axis=1)
        plt.fill_between(sigmas, means-stds, means+stds, alpha=0.3)
    
    plt.grid()
    plt.legend()
    plt.xlabel("Sigma")
    plt.ylabel("MSE")
    plt.title("Error vs sigma")

    plt.figure()
    for i, sigma in enumerate(sigmas):
        means = np.mean(errors[:, i], axis=1)
        plt.plot(Ns, np.mean(errors[:, i], axis=1), '--o', label=f"sigma = {sigma}")
        stds = np.std(errors[:, i], axis=1)
        plt.fill_between(Ns, means-stds, means+stds, alpha=0.3)
    
    plt.grid()
    plt.legend()
    plt.xlabel("n")
    plt.ylabel("MSE")
    plt.title("Error vs n")

def exp_2(M=20):
    sigmas = np.array([0.001, 0.01, 0.1, 1, 2])
    recon_snr = np.zeros((sigmas.size, M))
    data_snr = np.zeros((sigmas.size, M))
    n = 5
    for j, sigma in enumerate(sigmas):
        for k in range(M):
            images, img = get_data(N=n, sigma=sigma, translations=False)
            em = EM(img_shape=images[0].shape,
                    rotation_res=5,
                    scale_res=0.2,
                    trans_res=3,
                    max_iter=10,
                    verbose=False)

            rec_img = em.recover_img(images)
            recon_snr[j, k] = calc_mse(img=img, rec_img=rec_img)
            data_snr[j, k] = np.mean((images-img)**2)
            print(f'n = {n}, sigma = {sigma}, rep = {k}, snr = {recon_snr[j, k]}')

    data_snr = np.mean(data_snr, axis=-1)
    plt.figure()

    means = np.mean(recon_snr, axis=1)
    plt.plot(data_snr, means, '--o')
    stds = np.std(recon_snr, axis=1)
    plt.fill_between(data_snr, means-stds, means+stds, alpha=0.3)
    
    plt.grid()
    plt.xlabel("Data SNR")
    plt.ylabel("Reconstruction SNR")
    plt.xscale("log")

if __name__ == "__main__":
    exp_1()
    exp_2()
    plt.show()
