class NaiveBayes:
    def __init__(self):
        
        self.n_classes = None
        self.means = None
        self.variance = None
    
    def fit(self, X, y):
        
        self.y = np.array(y)
        
        self.n_classes = np.unique(y)
        self.mean = np.zeros((len(self.n_classes), X.shape[1]))
        self.var = np.zeros((len(self.n_classes), X.shape[1]))
        self.priors = np.zeros(len(self.n_classes))
        
        for i in self.n_classes:
            x_c = X[y == i]
            self.mean[i,:] = np.mean(x_c, axis = 0)
            self.var[i,:] = np.var(x_c, axis = 0)
            self.priors[i] = x_c.shape[0] / X.shape[0]
    
    def cond_prob(self, n, x):
        mean = self.mean[n]
        var = self.var[n]
        n = np.exp(- (x-mean)**2 / (2*var))
        d = np.sqrt(2*np.pi*var)
        return n / d
        
    def predict(self, X):
        y_pred = []
        for x in X:
            post = []
            for ind, c in enumerate(self.n_classes):
                prior = np.log(self.priors[ind])
                log_likelihood = np.sum(np.log(self.cond_prob(ind, x)))
                prob = prior + log_likelihood
                post.append(prob)
            y_pred.append(self.n_classes[np.argmax(post)])
            
        return np.array(y_pred)
        
class LinearDiscriminantAnalysis:
    
    def __init__(self):
        
        self.mean = None
        self.classes = None
        self.cov = None
        
    def fit(self, X, y):
        
        
        self.y = np.array(y)
        self.classes = np.unique(y)
        self.priors = np.zeros(len(self.classes))
        self.mean = np.zeros((len(self.classes), X.shape[1]))
        self.cov = np.cov(X[y==self.classes[0]], rowvar = False)
    
        
        for k in self.classes:
            arr = X[y == k]
            self.priors[k] = arr.shape[0] / X.shape[0]
            self.mean[k,:] = np.mean(arr, axis = 0)
        
    def predict(self, X):
        y_pred = []
        for x in X:
            x = x.reshape(-1, 1)
            post = []
            for ind, i  in enumerate(self.classes):
                delta = np.dot(np.dot(x.T,np.linalg.inv(self.cov)),self.mean[ind])-(1/2)*np.dot(np.dot(self.mean[ind], np.linalg.inv(self.cov)), self.mean[ind]) + np.log(self.priors[ind])
                post.append(delta)
                
            y_pred.append(self.classes[np.argmax(post)])    
        
        return np.array(y_pred)             
