import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.fftpack import fft, ifft
from sklearn.cluster import AgglomerativeClustering


class SSICOV(object):
    def __init__(self, dt, min_order=2, max_order=30, eps_freq=1e-2, eps_zeta=4e-2, eps_mac=5e-3, eps_cluster=.25,
                 methods=1):
        self.min_order = min_order
        self.max_order = max_order
        self.dt = dt
        self.eps_freq = eps_freq
        self.eps_zeta = eps_zeta
        self.eps_mac = eps_mac
        self.eps_cluster = eps_cluster
        self.methods = methods
        self.stab_plot_data = None

    def fit(self, data):
        Nyy, N = data.shape
        if Nyy > N:
            data = data.T
            (Nyy, N) = data.shape
        IRF = self.NExT(data)
        U, S, V = self.block_hankel(IRF)
        if U is None:
            fn = None
            zeta = None
            phi = None
            return
        # Stability Check
        fn2 = []
        zeta2 = []
        phi2 = []
        mac = []
        stability_status = []
        fn0, zeta0, phi0 = self.modal_ID(U, S, V, self.max_order, Nyy)
        for i in range(self.max_order - 1, self.min_order - 1, -1):
            fn1, zeta1, phi1 = self.modal_ID(U, S, V, i, Nyy)
            [a, b, c, d, e] = self.stability_check(fn0, zeta0, phi0, fn1, zeta1, phi1)
            fn2.insert(0, a)
            zeta2.insert(0, b)
            phi2.insert(0, c)
            mac.insert(0, d)
            stability_status.insert(0, e)
            fn0 = fn1
            zeta0 = zeta1
            phi0 = phi1

        fn_s, zeta_s, phi_s, mac_s = self.get_stable_poles(fn2, zeta2, phi2, mac, stability_status)

        if len(fn_s) <= 0:
            print('No stable poles found')
            fn = None
            zeta = None
            phi = None
            return fn, zeta, phi

        fn3, zeta3, phi3 = self.clusterFun(fn_s, zeta_s, phi_s)

        # average the clusters to get the frequency and mode shapes
        fn = []
        zeta = []
        phi = []
        for i in range(len(fn3)):
            fn.append(np.mean(fn3[i]))
            zeta.append(np.mean(zeta3[i]))
            phi.append(np.mean(phi3[i], axis=1))

        fn = np.array(fn)
        zeta = np.array(zeta)
        phi = np.array(phi)
        # sort the frequency
        I = np.argsort(fn)
        fn = fn[I]
        zeta = zeta[I]
        phi = phi[I]

        return fn, zeta, phi

    def NExT(self, data):
        Ts = 500 * self.dt
        (d1, d2) = data.shape
        M = int(Ts / self.dt)
        IRF = []
        for r1 in range(d1):
            sub_IRF = []
            for r2 in range(d1):
                y1 = fft(data[r1, :])
                y2 = fft(data[r2, :])
                h0 = ifft(y1 * y2.conjugate())
                sub_IRF.append(h0[0:M])
            IRF.append(sub_IRF)
        IRF = np.array(IRF)
        if d1 == 1:
            IRF = IRF.squeeze()
            IRF = IRF / IRF[1]
        return IRF

    def block_hankel(self, h):
        (d1, d2, d3) = h.shape
        if d1 != d2:
            print('The IRF must be a 3D matrix with shape (M, M, N)')
        N1 = int(d3 / 2) - 1
        M = d2
        T1 = np.zeros((M * N1, M * N1))
        for r1 in range(N1):
            for r2 in range(N1):
                T1[r1 * M:(r1 + 1) * M, r2 * M:(r2 + 1) * M] = h[:, :, N1 + r1 - r2 + 1]
        if np.any(T1 == 0) or np.any(T1 == float('inf')):
            U = None
            S = None
            V = None
            return
        try:
            U, S, V = np.linalg.svd(T1)
            S = np.diag(S)
        except LinAlgError as e:
            print('SVD computation does not converge')
        return U, S, V

    def modal_ID(self, U, S, V, N_modes, Nyy):
        if N_modes >= S.shape[0]:
            print('N_modes is larger than the number of row of S')
            O = np.dot(U, np.sqrt(S[:N_modes, :N_modes]))
            GAMMA = np.dot(np.sqrt(S[:N_modes, :N_modes]), V.T)
        else:
            O = np.dot(U[:, :N_modes], np.sqrt(S[:N_modes, :N_modes]))
            GAMMA = np.dot(np.sqrt(S[:N_modes, :N_modes]), V[:, :N_modes].T)

        # Get A and its eigen decomposition
        IndO = min(Nyy, O.shape[0])
        C = O[:IndO, :]
        jb = int(O.shape[0] / IndO)
        A = np.dot(np.linalg.pinv(O[:IndO * (jb - 1), :]), O[len(O) - IndO * (jb - 1):, :])
        Di, Vi = np.linalg.eig(A)

        mu = np.log(Di) / self.dt
        fn = np.abs(mu[1::2]) / (2 * np.pi)
        zeta = - np.real(mu[1::2]) / np.abs(mu[1::2])
        phi = np.real(np.dot(C[:IndO, :], Vi))
        phi = phi[:, 1::2]
        return fn, zeta, phi

    def stability_check(self, fn0, zeta0, phi0, fn1, zeta1, phi1):
        def err_check(x0, x1, eps):
            if abs(1 - x0 / x1) < eps:
                y = 1
            else:
                y = 0
            return y

        stability_status = []
        fn = []
        zeta = []
        phi = []
        mac = []

        # frequency stability
        N0 = len(fn0)
        N1 = len(fn1)

        for i1 in range(N0):
            for i2 in range(N1):
                stab_fn = err_check(fn0[i1], fn1[i2], self.eps_freq)
                stab_zeta = err_check(zeta0[i1], zeta1[i2], self.eps_zeta)
                stab_phi, dummy_mac = self.get_mac(phi0[:, i1], phi1[:, i2], self.eps_mac)

                # get stability status
                if stab_fn == 0:
                    stab_status = 0
                elif stab_fn == 1 and stab_zeta == 1 and stab_phi == 1:
                    stab_status = 1
                elif stab_fn == 1 and stab_zeta == 0 and stab_phi == 1:
                    stab_status = 2
                elif stab_fn == 1 and stab_zeta == 1 and stab_phi == 0:
                    stab_status = 3
                elif stab_fn == 1 and stab_zeta == 0 and stab_phi == 0:
                    stab_status = 5
                else:
                    print('stablity_status is undefined')
                fn.append(fn1[i2])
                zeta.append(zeta1[i2])
                phi.append(phi1[:, i2])
                mac.append(dummy_mac)
                stability_status.append(stab_status)

        fn = np.array(fn)
        ind = np.argsort(fn)
        fn = np.sort(fn)
        zeta = np.array(zeta)[ind]
        phi = np.array(phi)[ind]
        mac = np.array(mac)[ind]
        stability_status = np.array(stability_status)[ind]
        return fn, zeta, phi, mac, stability_status

    def get_stable_poles(self, fn, zeta, phi, mac, stability_status):
        fn_s = []
        zeta_s = []
        phi_s = []
        mac_s = []
        for f_i in range(len(fn)):
            for s_i in range(len(stability_status[f_i])):
                if stability_status[f_i][s_i] == 1 and zeta[f_i][s_i] > 0:
                    fn_s.append(fn[f_i][s_i])
                    zeta_s.append(zeta[f_i][s_i])
                    phi_s.append(phi[f_i][s_i])
                    mac_s.append(mac[f_i][s_i])

        fn_s = np.array(fn_s)
        zeta_s = np.array(zeta_s)
        phi_s = np.array(phi_s)
        mac_s = np.array(mac_s)

        # Normalized mode shape
        for r in range(phi_s.shape[1]):
            phi_s[:, r] = phi_s[:, r] / np.max(np.abs(phi_s[:, r]))
            if phi_s.shape[0] > 1 and (phi_s[0, r] - phi_s[1, r]) < 0:
                phi_s[:, r] = - phi_s[:, r]
        return fn_s, zeta_s, phi_s, mac_s

    def clusterFun(self, fn, zeta, phi):
        c = len(phi)
        pos = np.zeros((c, c))
        for i in range(c):
            for j in range(c):
                _, mac0 = self.get_mac(phi[i], phi[j], self.eps_mac)
                pos[i, j] = np.abs((fn[i] - fn[j]) / fn[j]) + 1 - mac0
        if len(pos) == 1:
            print('linkage failed: at lease two observations are required')
            return
        myClus = AgglomerativeClustering(linkage='single', affinity='euclidean', n_clusters=None,
                                         distance_threshold=self.eps_cluster).fit_predict(pos)
        Ncluster = np.max(myClus) + 1
        fn_c = []
        zeta_c = []
        phi_c = []
        for r in range(Ncluster):
            count = 0
            index_list = []
            fn_t = []
            zeta_t = []
            phi_t = []
            for index, value in enumerate(myClus):
                if value == r:
                    count += 1
                    index_list.append(index)
            if count > 5:
                dummyZeta = zeta[index_list]
                dummyFn = fn[index_list]
                dummyPhi = phi[index_list]
                q1 = np.quantile(dummyZeta, 0.25)
                q3 = np.quantile(dummyZeta, 0.75)
                valMin = max(0, (q1 - abs(q3 - q1) * 1.5))
                valMax = q3 + abs(q3 - q1) * 1.5
                mask = np.logical_and(dummyZeta < valMax, dummyZeta > valMin)

                for c in range(len(dummyZeta)):
                    if mask[c] == 1:
                        fn_t.append(dummyFn[c])
                        zeta_t.append(dummyZeta[c])
                        phi_t.append(dummyPhi[c])

                fn_c.append(fn_t)
                zeta_c.append(zeta_t)
                phi_c.append(phi_t)
        if len(fn_c) < 1:
            fn_c = None
            zeta_c = None
            phi_c = None
        return fn_c, zeta_c, phi_c

    def get_mac(self, phi1, phi2, eps):
        phi1_r = np.array(phi1).flatten('F')
        phi1_T = np.mat(phi1).H.A.flatten()
        phi2_r = np.array(phi2).flatten('F')
        phi2_T = np.mat(phi2).H.A.flatten()
        Num = np.abs(np.dot(phi1_T, phi2_r)) ** 2
        D1 = np.dot(phi1_T, phi1_r).real
        D2 = np.dot(phi2_T, phi2_r).real
        dummy_mac = Num / np.abs(D1 * D2)
        if dummy_mac > (1 - eps):
            y = 1
        else:
            y = 0
        return y, dummy_mac


if __name__ == '__main__':
    data = pd.read_csv(r'beamdata1.csv', header=None)
    print(data.shape)
    print(data.head())
    data = data.values
    model = SSICOV(dt=.01)
    model.fit(data)
