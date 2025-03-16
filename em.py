import numpy as np
import cv2
import matplotlib.pyplot as plt

class EM():

    def __init__(self, img_shape, rotation_res=5, scale_res=0.1, max_iter=10):
        self.J = img_shape[0] * img_shape[1]
        self.translations_sigma = 0.05 * img_shape[0] * img_shape[1]
        self.rotation_res = rotation_res
        self.scale_res = scale_res
        self.rots = np.arange(0, 360, rotation_res)
        n_scales = int(2 // scale_res)
        self.scales = np.linspace(0.5, 1.5, n_scales)
        self.img_shape = img_shape
        self.max_iter = max_iter
        # self.n_trans_params = self.rots.size * self.scales.size
        # self.params = np.zeros([n_trans_params, 2])
        # for i in range(self.n_trans_params):
        #     self.params[i, 0] = 

    def prior(self, rot, scale, t_x, t_y):
        p = np.exp(-1 * ((t_x ** 2) + (t_y ** 2)) / (2 * self.translations_sigma ** 2))
        assert p == 1
        return p

    def get_phi_mat(self, rot, scale, t_x=None, t_y=None):
        return cv2.getRotationMatrix2D(center=(self.img_shape[1]/2, self.img_shape[0]/2),
                                       angle=rot, scale=scale)


    def ll(self, X, phi, A, sigma):
        phiX = cv2.warpAffine(X, phi, self.img_shape[::-1], flags=cv2.INTER_LINEAR).astype(np.float64)
        # plt.imshow(X-phiA, cmap='gray')
        # plt.axis('off')
        # plt.show()
        # return np.exp(np.dot(phiX,A).sum() / (sigma**2))
        # cross_val = np.sum(phiX*A)
        # # weight is exponential of cross-correlation / sigma^2
        # return  np.exp(cross_val / (sigma**2))
        return -1 * np.sum((A-phiX)**2) / (2*(sigma**2))

    def recover_img(self, images):
        # images = images / 256
        images = images / 1000

        # initialize parameters
        sigma = 0.1
        A = images[0]
        # A = np.mean(images, axis=0)

        ws = np.zeros((images.shape[0], self.rots.size, self.scales.size))
        lls = np.zeros((images.shape[0], self.rots.size, self.scales.size))
        for iteration in range(self.max_iter):
            print(f'iter {iteration}')
            # E step
            for i, X_i in enumerate(images):
                for j, r in enumerate(self.rots):
                    for k, s in enumerate(self.scales):
                        phi_mat = self.get_phi_mat(rot=r, scale=s, t_x=None, t_y=None)
                        lls[i, j, k]  = self.ll(X=X_i, phi=phi_mat, A=A, sigma=sigma) * self.prior(rot=r, scale=s, t_x=0, t_y=0)

            for i, X_i in enumerate(images):
                max_ll = np.max(lls[i])
                # log_denom = np.log(np.sum(np.exp(lls[i])))
                log_denom = np.log(np.sum(np.exp(lls[i] - max_ll))) + max_ll
                denom = np.exp(log_denom)
                # print(f'denom = {denom}')
                ws[i] = np.exp(lls[i] - log_denom)

            print(np.sum(ws, axis=(1, 2)))
            # M step
            new_A = np.zeros_like(A)
            sigma_sqr = 0
            for i, X_i in enumerate(images):
                for j, r in enumerate(self.rots):
                    for k, s in enumerate(self.scales):
                        phi = self.get_phi_mat(rot=r, scale=s)
                        phiXi = cv2.warpAffine(X_i, phi, self.img_shape[::-1], flags=cv2.INTER_LINEAR).astype(np.float64)
                        new_A += (1 / images.shape[0]) * ws[i, j, k] * phiXi
                        sigma_sqr += (1 / (images.shape[0]*self.J)) * ws[i, j, k] * np.sum((A-phiXi)**2)

            sigma = np.sqrt(sigma_sqr)
            print(A.mean(), new_A.mean())
            A = new_A
            print(f'Sigma = {sigma}')
            # plt.imshow(A, cmap='gray')
            # plt.axis('off')
            # plt.show()
        
        return A * 1000, sigma * 1000
