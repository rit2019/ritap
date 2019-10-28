import torch
from helpers import setup_params, create_gmm_data
from part1 import gmm_neg_log_likelihood

# Setup parameters and test dataset
weights, mean_tensor, cov_tensor = setup_params()
X_test = create_gmm_data(weights, mean_tensor, cov_tensor)
param_list = [weights, mean_tensor, cov_tensor]

########################################
# Student code will be below
# Use backward to compute gradients of mean negative log likelihood
# and print out gradients
########################################
weights=param_list[0].clone().detach().requires_grad_(True)
mean_tensor=param_list[1].clone().detach().requires_grad_(True)
cov_tensor=param_list[2].clone().detach().requires_grad_(True)

a = torch.mean(gmm_neg_log_likelihood(weights, mean_tensor, cov_tensor,X_test))
a.backward()

param_list = [weights, mean_tensor, cov_tensor]

########################################
# Student code will be above
########################################
# Print out gradient values for each parameter
for param in param_list:
    print(param.grad)
