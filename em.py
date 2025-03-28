import numpy as np
import cv2
import matplotlib.pyplot as plt
import concurrent.futures


class EM():

    def __init__(self, img_shape, rotation_res=5, scale_res=0.1,
                 trans_res=1, max_iter=10, verbose=False):
        self.verbose = verbose
        self.J = img_shape[0] * img_shape[1]
        self.translations_sigma = 0.05 * img_shape[0]
        self.rotation_res = rotation_res
        scale_res = scale_res
        rots = np.arange(0, 360, rotation_res)
        n_scales = int(1 // scale_res)
        scales = np.linspace(0.5, 1.5, n_scales)
        n_trans = int(2 * 0.05 * img_shape[0] // trans_res)
        translations = np.linspace(-int(0.05 * img_shape[0]), int(0.05 * img_shape[0]), n_trans).astype(np.int32)
        # translations = np.array([0])
        self.phis = np.zeros(
            [rots.size*scales.size*translations.size*translations.size, 2, 3])
        self.lpriors = np.zeros(
            [rots.size*scales.size*translations.size*translations.size])
        c = 0
        for j, r in enumerate(rots):
            for k, s in enumerate(scales):
                for l, t_x in enumerate(translations):
                    for m, t_y in enumerate(translations):
                        self.lpriors[c] = self.lprior(rot=r, scale=s, t_x=t_x, t_y=t_y)
                        self.phis[c] = cv2.getRotationMatrix2D(center=(img_shape[1]/2,
                                                                    img_shape[0]/2),
                                                            angle=r,
                                                            scale=s)
                        self.phis[c][0, 2] += t_x
                        self.phis[c][1, 2] += t_y
                        c += 1

        if self.verbose:
            print(f'model is ready. table size = {self.phis.shape}')

        self.img_shape = img_shape
        self.max_iter = max_iter

    def lprior(self, rot, scale, t_x, t_y):
        lp =  -1 * ((t_x ** 2) + (t_y ** 2)) / (2 * self.translations_sigma ** 2) + np.log(scale) + 4*np.log(1 - (scale/5))
        return lp

    def ll(self, phiX, A, sigma):
        """
        Calculate the log-liklihood of the transformed image phiX.
        the constant terms are neglected.
        """
        return -1 * np.sum(np.square(A-phiX)) / (2*(sigma**2))

    def single_img(self, i):
        new_A = np.zeros_like(self.A)
        sigma_sqr = 0
        self.phiXs = np.zeros((self.phis.shape[0], self.img_shape[0], self.img_shape[1]), dtype=np.float16)

        def single_phi(j):
            self.phiXs[j] = cv2.warpAffine(self.images[i], self.phis[j], self.img_shape[::-1],
                                           flags=cv2.INTER_NEAREST).astype(np.float16)
            self.lls[i, j] = self.ll(phiX=self.phiXs[j], A=self.A, sigma=self.sigma) + self.lpriors[j]

        # multi-threading to reduce run-time
        with concurrent.futures.ThreadPoolExecutor() as executor:
                list(executor.map(single_phi, list(range(self.phis.shape[0]))))

        max_ll = np.max(self.lls[i])
        log_denom = np.log(np.sum(np.exp(self.lls[i] - max_ll))) + max_ll
        self.ws[i] = np.exp(self.lls[i] - log_denom)
        non_zeros = np.argwhere(self.ws[i] > 0)[:, 0]
        # print(non_zeros.size)
        for j in non_zeros:
            new_A += (1 / self.images.shape[0]) * self.ws[i, j] * self.phiXs[j]
            sigma_sqr += (1 / (self.images.shape[0]*self.J)) * \
                self.ws[i, j] * np.sum((self.A-self.phiXs[j])**2)

        return new_A, sigma_sqr

    def recover_img(self, images):
        self.images = images

        # initialize parameters
        self.sigma = np.mean(np.abs(images))
        self.A = images[0]
        # self.A = np.mean(images, axis=0)

        self.ws = np.zeros((images.shape[0], self.phis.shape[0]))
        self.lls = np.zeros((images.shape[0], self.phis.shape[0]))
        for iteration in range(self.max_iter):
            if self.verbose:
                print(f'Iter {iteration}')
            
            new_A = np.zeros_like(self.A)
            sigma_sqr = 0
            for i, X_i in enumerate(self.images):
                dA, dS = self.single_img(i)
                new_A += dA
                sigma_sqr += dS

            self.sigma = np.sqrt(sigma_sqr)
            realtive_change = np.mean(np.abs((self.A-new_A))) / (np.mean(np.abs(self.A)) + 1e-4)
            if self.verbose:
                print(f"Realtive_change = {realtive_change}")
            self.A = new_A
            if realtive_change < 0.01:
                return self.A

            if self.verbose:
                print(f'Sigma = {self.sigma}')
            # plt.imshow(self.A, cmap='gray')
            # plt.axis('off')
            # plt.show()

        return self.A
