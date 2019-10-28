import torch
from helpers import setup_params, create_gmm_data
from part1 import gmm_neg_log_likelihood
from part4 import params_to_theta, theta_to_params

# Create true model and sample from this model
true_params_list = setup_params(torch_random_seed=0)
X_train = create_gmm_data(*true_params_list, n_samples=1000)

# Generate initial random parameters for mixture (notice different seed than above)
init_params_list = setup_params(torch_random_seed=1)
theta_list = params_to_theta(init_params_list)

# Set requires_grad for each theta
for theta in theta_list:
    theta.requires_grad_(True)

# Loop through gradient descent
max_iter = 100
learning_rate = 0.5
torch.manual_seed(1)
for it in range(max_iter):
    
    ########################################
    # Student code will be below
    # Compute vanilla gradient descent without using `torch.optim` package
    # Hint: First convert theta_list to param_list via part4 functions
    ########################################
    param_list = theta_to_params(theta_list)
    weights=param_list[0]
    ##print(weights)
    mean_tensor=param_list[1]
    cov_tensor=param_list[2]
    
    a = gmm_neg_log_likelihood(weights, mean_tensor, cov_tensor, X_train)
    #a = gmm_neg_log_likelihood(*true_params_list, X_train)
    mean_nll = torch.mean(a)
    
    for theta in theta_list:
        if theta.grad is not None:
            theta.grad.zero_() 
    mean_nll.backward() 
    
    with torch.no_grad():
        
        for theta in theta_list:
           
            theta -= learning_rate * theta.grad

    ########################################
    # Student code will be above 
    ########################################
    print('%.5f' % mean_nll.detach().numpy())
