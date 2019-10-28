import torch
from helpers import setup_params, create_gmm_data
from part1 import gmm_neg_log_likelihood

# Setup parameters
params_list = setup_params()
X_opt = create_gmm_data(*params_list, n_samples=10)
max_iter = 100
learning_rate = 0.1

# Loop over test points
for i, x in enumerate(X_opt):
    # Loop over gradient descent iterations
    x.requires_grad_(True)
    for it in range(max_iter):
        ########################################
        # Student code will be below
        # Optimize for highest density x using simple gradient descent without using `torch.optim` package
        ########################################
        #pass # Placeholder
        out = gmm_neg_log_likelihood(*params_list, x.reshape(1, -1))
        
        out.backward()
        
        with torch.no_grad():
           #x.grad.zero_()
           x -=learning_rate * x.grad
           x.grad.zero_()

        ########################################
        # Student code will be above
        ########################################
    # Print out the optimized x
    print(x)
