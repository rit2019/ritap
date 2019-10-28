import torch
import numpy as np
from sklearn.mixture import GaussianMixture

# Generate random mixture model
def setup_params(n_components=3, n_features=2, torch_random_seed=0):
    torch.manual_seed(torch_random_seed)
    weights = (1/n_components)*torch.ones(n_components)
    mean_tensor = torch.randn(n_components, n_features)
    cov_tensor = torch.zeros(n_components, n_features, n_features)
    for i in range(cov_tensor.size()[0]):
        A = torch.randn(n_features, n_features)
        AAt = torch.matmul(A, A.t())
        assert torch.all(AAt == AAt.t())
        cov_tensor[i, :, :] = AAt
    return weights, mean_tensor, cov_tensor

# Create GaussianMixture from sklearn so we can generate sample data
def create_sklearn_gmm(weights, mean_tensor, cov_tensor, random_state=0):
    n_components = len(weights)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=random_state)
    gmm.weights_ = weights.numpy()
    gmm.means_ = mean_tensor.numpy()
    gmm.covariances_ = cov_tensor.numpy()
    gmm.precisions_ = np.array([np.linalg.inv(cov) for cov in gmm.covariances_])
    gmm.precisions_cholesky_ = np.array([np.linalg.cholesky(prec) for prec in gmm.precisions_])
    return gmm


def create_gmm_data(weights, mean_tensor, cov_tensor, n_samples=1000, random_state=0):
    gmm_sklearn = create_sklearn_gmm(weights.detach(), mean_tensor.detach(),
                                     cov_tensor.detach(), random_state=random_state)
    import warnings
    from sklearn.utils import shuffle
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # Warnings should be ignored (possibly related to a bug in numpy)
        X_test_numpy, y = gmm_sklearn.sample(n_samples)
        X_test_numpy, y = shuffle(X_test_numpy, y, random_state=0)
        X_test = torch.from_numpy(X_test_numpy).float()
    return X_test


