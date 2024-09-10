import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SSA(object):
    
    # tipe data yang didukung kelas SSA
    __supported_types = (pd.Series, np.ndarray, list)
    
    def __init__(self, tseries, L, save_mem=True):
        """
        Menguraikan deret waktu yang diberikan dengan singular spectrum analysis. Data time series diasumsikan 
        memiliki interval waktu yang konsisten antara setiap titik data.
        
        Parameters
        ----------
        tseries  : Data deret waktu asli, dalam bentuk pandas series, numpy array, atau list python. 
        L        : Panjang jendela. harus berupa integer dengan 2 <= L <= N/2, dimana N panjang data deret waktu.
        save_mem : Menghemat memori dengan tidak menyimpan matriks dasar. Direkomendasikan untuk deret waktu yang panjang dengan ribuan nilai. Setelan default ke True.
        
        Note: Meskipun numpy array atau daftar digunakan untuk deret waktu awal, semua deret waktu yang dikembalikan 
        akan berbentuk objek Pandas Series atau DataFrame.
        """
        # memeriksa tipe data yang dimasukkan
        if not isinstance(tseries, self.__supported_types):
            raise TypeError("Unsupported time series object. Try Pandas Series, NumPy array or list.")
        
        # memeriksa panjang jendela sesuai yang sudah ditetapkan
        self.N = len(tseries)
        if not 2 <= L <= self.N/2:
            raise ValueError("The window length must be in the interval [2, N/2].")
        
        self.L = L
        self.orig_TS = pd.Series(tseries)
        self.K = self.N - self.L + 1

        # membangun matriks lintasan X (embedding)
        self.X = np.array([self.orig_TS.values[i:L+i] for i in range(0, self.K)]).T
        
        # dekomposisi matriks lintasan
        self.U, self.Sigma, VT = np.linalg.svd(self.X)
        self.d = np.linalg.matrix_rank(self.X)
        
        self.TS_comps = np.zeros((self.N, self.d))

        if not save_mem:
            # konstruksi dan simpan seluruh matriks dasar
            self.X_elem = np.array([ self.Sigma[i]*np.outer(self.U[:,i], VT[i,:]) for i in range(self.d) ])

            # menerapkan diagonal averaging ke matriks dasar, dan menyimpan nya sebagai kolom dalam array           
            for i in range(self.d):
                X_rev = self.X_elem[i, ::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.V = VT.T
        else:
            # rekontruksi matriks dasar tanpa menyimpannya
            for i in range(self.d):
                X_elem = self.Sigma[i]*np.outer(self.U[:,i], VT[i,:])
                X_rev = X_elem[::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.X_elem = "Re-run with save_mem=False to retain the elementary matrices."
            
            # array V mungkin saja terlalu besar untuk disimpan, sehingga tidak begitu digunakan
            self.V = "Re-run with save_mem=False to retain the V matrix."

        # menghitung matriks korelasi w
        self.calc_wcorr()

    def components_to_df(self, n=0):
        """
        Mengembalikan seluruh komponen data deret waktu ke dalam bentuk objek Pandas DataFrame..
        """
        if n > 0:
            n = min(n, self.d)
        else:
            n = self.d
        
        # Buat daftar kolom - sebut saja F0, F1, F2, ...
        cols = ["F{}".format(i) for i in range(n)]
        return pd.DataFrame(self.TS_comps[:, :n], columns=cols, index=self.orig_TS.index)
    
    def reconstruct(self, indices):
        """
        Merekonstruksi deret waktu dari komponen-komponen dasarnya, dengan menggunakan indeks yang diberikan. 
        Mengembalikan objek Series Pandas dengan deret waktu yang telah direkonstruksi.
        
        Parameters
        ----------
        indices: Bilangan bulat, daftar bilangan bulat, atau objek slice(n,m), yang mewakili komponen dasar untuk dijumlahkan.
        """
        if isinstance(indices, int): indices = [indices]
        
        ts_vals = self.TS_comps[:,indices].sum(axis=1)
        return pd.Series(ts_vals, index=self.orig_TS.index)
    
    def calc_wcorr(self):
        """
        Kalkulasi matriks korelasi w untuk deret waktu.
        """
             
        # Calculate the weights
        w = np.array(list(np.arange(self.L)+1) + [self.L]*(self.K-self.L-1) + list(np.arange(self.L)+1)[::-1])
        
        def w_inner(F_i, F_j):
            return w.dot(F_i*F_j)
        
        # Calculated weighted norms, ||F_i||_w, then invert.
        F_wnorms = np.array([w_inner(self.TS_comps[:,i], self.TS_comps[:,i]) for i in range(self.d)])
        F_wnorms = F_wnorms**-0.5
        
        # Calculate Wcorr.
        self.Wcorr = np.identity(self.d)
        for i in range(self.d):
            for j in range(i+1,self.d):
                self.Wcorr[i,j] = abs(w_inner(self.TS_comps[:,i], self.TS_comps[:,j]) * F_wnorms[i] * F_wnorms[j])
                self.Wcorr[j,i] = self.Wcorr[i,j]

    def plot_wcorr(self, min=None, max=None):
        """
        Plots the w-correlation matrix for the decomposed time series.
        """
        if min is None:
            min = 0
        if max is None:
            max = self.d
        
        if self.Wcorr is None:
            self.calc_wcorr()
        
        ax = plt.imshow(self.Wcorr)
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label("$W_{i,j}$")
        plt.clim(0,1)
        
        # For plotting purposes:
        if max == self.d:
            max_rnge = self.d-1
        else:
            max_rnge = max
        
        plt.xlim(min-0.5, max_rnge+0.5)
        plt.ylim(max_rnge+0.5, min-0.5)