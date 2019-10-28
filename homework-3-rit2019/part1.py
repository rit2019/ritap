import math
import numpy as np
import scipy.special
import torch
from helpers import setup_params, create_gmm_data

def numpy_gaussian_neg_log_likelihood(mean, cov, X):
    k = len(mean)
    Xc = X - mean
    out = -(
        -1/2 * np.sum(np.dot(Xc, np.linalg.inv(cov)) * Xc, axis=1)
        -1/2 * np.log(np.linalg.det(cov)) - k/2 * np.log(2*math.pi)
    )
    return out

def numpy_gmm_neg_log_likelihood(weights, mean_tensor, cov_tensor, X):
    n_samples, n_features = X.shape
    n_components = weights.shape[0]
    log_prob = np.zeros((n_samples, n_components))
    for i, (mean, cov) in enumerate(zip(mean_tensor, cov_tensor)):
        log_prob[:, i] = -numpy_gaussian_neg_log_likelihood(mean, cov, X)
    log_prob_weighted = log_prob + np.log(weights)
    gmm_log_prob = scipy.special.logsumexp(log_prob_weighted, axis=1)
    return -gmm_log_prob

########################################
# Student code will be below: My code below
# Rewrite numpy functions above using torch functions
########################################
def gaussian_neg_log_likelihood(mean, cov, X):
    #raise NotImplementedError()
    k = len(mean)
    Xc = X - mean
    out = -(
        -1/2 * torch.sum(torch.matmul(Xc, torch.inverse(cov)) * Xc, dim=1)
        -1/2 * torch.log(torch.det(cov))- k/2 * torch.log(torch.tensor(2*math.pi))
    )
    return out

def gmm_neg_log_likelihood(weights, mean_tensor, cov_tensor, X):
    n_samples, n_features = X.shape
    n_components = weights.shape[0]
    
    log_prob = torch.zeros((n_samples, n_components))
    for i, (mean, cov) in enumerate(zip(mean_tensor, cov_tensor)):
        log_prob[:, i] = -gaussian_neg_log_likelihood(mean, cov, X)
    log_prob_weighted = log_prob + torch.log(weights)
    gmm_log_prob = torch.logsumexp(log_prob_weighted, dim=1)
    #return -torch.mean(gmm_log_prob) 
    return -gmm_log_prob
    #return np.zeros(X.size()[0]) # Placeholder (obviously need to change this)
########################################
# Student code will be above
########################################

if __name__ == '__main__':
    weights, mean_tensor, cov_tensor = setup_params()
    X_test = create_gmm_data(weights, mean_tensor, cov_tensor)

    # Uncomment lines below if you want to check your output
    #numpy_nll_all = numpy_gmm_neg_log_likelihood(weights.numpy(), mean_tensor.numpy(), cov_tensor.numpy(), X_test[:10,:].numpy())
    #for nll in numpy_nll_all:
    #    print('%.5f' % nll)

    nll_all = gmm_neg_log_likelihood(weights, mean_tensor, cov_tensor, X_test[:10,:])
    for nll in nll_all:
        print('%.5f' % nll.item())  # item() extracts value from scalar tensor
