import torch
from helpers import setup_params, create_gmm_data

def params_to_theta(param_list):
    """Converts constrained parameters to unconstrained versions theta."""
    weights, mean_tensor, cov_tensor = param_list
    theta_weights = torch.log(weights)
    theta_mean_tensor = mean_tensor.clone()
    theta_cov_tensor = torch.zeros_like(cov_tensor)
    for i, cov in enumerate(cov_tensor):
        theta_cov_tensor[i, :, :] = torch.cholesky(cov)
    return theta_weights, theta_mean_tensor, theta_cov_tensor
    
def theta_to_params(theta_list):
    ########################################
    # Student code will be below
    # Convert unconstrained `theta_list` to constrained `params_list`
    ########################################
    # Placeholders (need to change obviously)
    weights = torch.softmax(theta_list[0],dim=0)
    
    mean_tensor = theta_list[1]
    
    
    cov_tensor = torch.zeros_like(theta_list[2])
    
    
    for i, cov in enumerate(theta_list[2]):
        cov_tensor[i, :, :] = torch.matmul(cov, cov.t())    

    ########################################
    # Student code will be above
    ########################################
    return weights, mean_tensor, cov_tensor

if __name__ == '__main__':
    # Setup parameters
    param_list = setup_params()

    # Convert from constrained params to unconstrained theta
    theta_list = params_to_theta(param_list)

    # Convert back from theta to params and check
    param_list_test = theta_to_params(theta_list)
    for a, b in zip(param_list, param_list_test):
        if b is not None:
            print(torch.allclose(a, b))
        else:
            print('theta_to_params has not been implemented yet')
