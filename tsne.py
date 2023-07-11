import numpy as np
from sklearn.metrics import pairwise_distances

class sne:
    def __init__(self, n_components, perplexity, learning_rate, n_iter=10):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.y = None

    def p_i(self, idx, X, s):
        p = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            num = np.exp(-np.linalg.norm(X[idx]-X[i]) / (2*np.square(s)))
            den = 0
            for j in range(X.shape[0]):
                if i!=j:
                    den += np.exp(-np.linalg.norm(X[i]-X[j]) / (2*np.square(s)))
            p[i] = num / den
        return p

    def p(self, dist, sigma):
        num = np.exp(-dist / (2*np.square(sigma.reshape(-1,1))))
        np.fill_diagonal(num, 0)
        num += 1e-8
        return num / num.sum(axis=1).reshape([-1,1])

    def shannon(self, mtr):
        sh = -np.sum(mtr * np.log2(mtr))
        return 2**sh

    def gradient(self, p, q, y):
        n = p.shape[0]
        grad = 0
        for i in range(n):
            for j in range(n):
                grad += (p[i,j]-q[i,j])*(y[i]-y[j])
        return 4*grad

    def search(self, id, perp, X, tol=1e-10, max_iters=1000, l=1e-20, r=10000):

        for _ in range(max_iters):

            s = (r + l) / 2
            p_i = self.p_i(id, X, np.array([s]))
            val = self.shannon(p_i)

            if val > perp:
                r = s
            else:
                l = s

            if np.abs(val - perp) <= tol:
                return s

        return s

    def joint_p(self, X, perp):
        n_sample = X.shape[0]
        dist = pairwise_distances(X)
        sigma = self.sigma(X, perp)
        p = self.p(dist, sigma)
        return (p+p.T) / (2*n_sample)

    def q_joint(self, y):
        dists = pairwise_distances(y)
        e = 1 / (1 + dists)
        np.fill_diagonal(e, 0)
        return e / np.sum(np.sum(e))

    def sigma(self, X, perp):
        sigma = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            sigma[i] = self.search(i, perp, X)
        return sigma

    def m(self, t):
        return 0.5 if t < 250 else 0.8

    def fit(self, X):
        n_sample = X.shape[0]
        sigma = self.sigma(X, self.perplexity)
        P = self.joint_p(X, self.perplexity)

        self.y = np.random.normal(loc=0, scale=10**(-4), size=(n_sample, self.n_components))
        for i in range(self.n_iter):
            q = self.q_joint(self.y)
            for j in range(2, n_sample):
                self.y[j] = self.y[j-1]+self.learning_rate*self.gradient(P, q, self.y)+self.m(i)*(self.y[j-1]-self.y[j-2])

        return self.y