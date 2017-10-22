
import numpy as np
import math
import scipy.stats as ss
from helpers import *





# EXERCISE 2 ###############################################################################################################


def compute_loss(y, tx, w, cost = "mse", lambda_ = 0):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    N = tx.shape[0]
    error_vec = y - tx.dot(w)
    if (cost == "mse"):
        return (1 / (2*N)) * np.linalg.norm(error_vec)**2
    elif (cost == "mae"):
        return (1 / N) * np.sum(np.abs(error_vec))
    elif (cost == "ridge"):
        return (1 / (2*N)) * np.linalg.norm(y - tx.dot(w))**2 + lambda_ * np.linalg.norm(w)**2
    elif (cost == "logistic"):
        return np.sum(np.log(1 + np.exp(tx.dot(w))) - y * tx.dot(w))
    elif (cost == "reg_logistic"):
        return np.sum(np.log(1 + np.exp(tx.dot(w))) - tx.dot(w) * y) + (lambda_ / 2) * np.linalg.norm(w)**2
    else:
        raise IllegalArgument("Invalid cost argument in compute_loss")
    return 0






def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    losses = np.zeros((len(w0), len(w1)))
    for i in range(len(w0)):
        for j in range(len(w1)):
            losses[i, j] = compute_loss(y, tx, np.array([w0[i], w1[j]]))
    
    argmin_index = np.argmin(losses);
    best_i = argmin_index // len(w1)
    best_j = argmin_index % len(w1)
    # best_i, best_j = np.unravel_index(np.argmin(losses), losses.shape)

    
    return losses, [w0[best_i], w1[best_j]]




def compute_gradient(y, tx, w, cost = "mse", lambda_ = 0):
    """Compute the gradient."""
    N = tx.shape[0];
    error_vec = y - tx.dot(w)
    if (cost == "mse"):
        return -(1/N) * np.transpose(tx).dot(error_vec)
    elif (cost == "mae"):
        # Note that here it is not a gradient strictly speaking because the mae is not differentiable everywhere
        sub_grad = [0 if error_vec[i] == 0 else error_vec[i] / np.abs(error_vec[i]) for i in range(N)]
        return -(1 / N) * np.transpose(tx).dot(sub_grad)
    elif (cost == "logistic"):
        try:
            w.shape[1]
        except IndexError:
            w = np.expand_dims(w, 1)
        return np.transpose(tx).dot(sigmoid(tx.dot(w)) - y)
    elif (cost == "reg_logistic"):
        try:
            w.shape[1]
        except IndexError:
            w = np.expand_dims(w, 1)
        return np.transpose(tx).dot(sigmoid(tx.dot(w)) - y) + lambda_ * w
    else:
        raise IllegalArgument("Invalid cost argument in compute_gradient function.")
    return 0




# Be careful, gamma of 2 for MSE doesn't converge
def gradient_descent(y, tx, initial_w, max_iters, gamma, cost ="mse", lambda_ = 0, tol = 1e-6, thresh_test_conv = 10, update_gamma = False):
    """Gradient descent algorithm."""
    if (cost not in ["mse", "mae", "logistic", "reg_logistic"]):
        raise IllegalArgument("Invalid cost argument in gradient_descent function")
    
    # ensure that w_initial is formatted in the right way if implementing logistic regression
    if (cost == "logistic" or cost == "reg_logistic"):
        try:
            initial_w.shape[1]
        except IndexError:
            initial_w = np.expand_dims(initial_w, 1)
    
        
    # Define parameters to store w and loss
    ws = [initial_w]
    initial_loss = compute_loss(y, tx, initial_w, cost = cost, lambda_ = lambda_)
    losses = [initial_loss]
    w = initial_w
    test_conv = 0
    dist_succ_loss = tol + 1
    n_iter = 0;
    
    while (n_iter < max_iters and dist_succ_loss > tol):
        grad = compute_gradient(y, tx, w, cost = cost, lambda_ = lambda_)
        
        # updating w
        w = w - gamma * grad
        loss = compute_loss(y, tx, w, cost = cost, lambda_ = lambda_)
        
        # Test of divergence, test_conv counts the number of consecutive iterations for which loss has increased
        if (loss > losses[-1]):
            test_conv += 1
            if (test_conv >= thresh_test_conv):
                print("Stopped computing because 10 consecutive iterations have involved an increase in loss.")
                return losses, ws
        else:
            test_conv = 0

        # store w and loss
        ws.append(w)
        losses.append(loss)
     
        # update distance between two successive w
        dist_succ_loss = np.abs(losses[-1] - losses[-2])
        
        # update n_iter: number of iterations
        n_iter += 1
        
        # update gamma if update_gamma is True
        if (update_gamma == True):
            gamma = 1 / (1 + n_iter) * gamma
    
    # Printing the results
    try:
        initial_w.shape[1]
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}, cost: {cost}".format(
              bi=n_iter-1, ti=max_iters - 1, l=loss, w0=w[0,0], w1=w[1,0], cost = cost))
    except (IndexError, AttributeError):
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}, cost: {cost}".format(
                  bi=n_iter-1, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1], cost = cost))
    return losses, ws






def compute_stoch_gradient(y, tx, w, batch_size=1, cost ="mse"):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    batched = [x for x in batch_iter(y, tx, batch_size)][0]
    y_b = batched[0]
    tx_b = batched[1]
    error_vec = y_b - tx_b.dot(w)

    if (cost == "mse"):
        stoch_grad = (-2 / batch_size) * np.transpose(tx_b).dot(error_vec)
    elif (cost == "mae"):
        sub_grad_abs = [0 if error_vec[i] == 0 else error_vec[i] / np.abs(error_vec[i]) for i in range(batch_size)]
        stoch_grad = -(1 / batch_size) * np.transpose(tx_b).dot(sub_grad_abs)
    else:
        raise IllegalArgument("Invalid cost argument in compute_stoch_gradient")
    
    return stoch_grad


# batxh_iter is used to sample the training data used to compute the stochastic gradient
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]




def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma, cost = "mse", tol = 1e-6, thresh_test_conv = 100):
    # Define parameters to store w and loss
    ws = [initial_w]
    initial_loss = compute_loss(y, tx, initial_w, cost)
    losses = [initial_loss]
    w = initial_w
    test_conv = 0
    dist_succ_w = tol + 1
    n_iter = 0
    while (n_iter < max_iters and dist_succ_w > tol):
        stoch_grad = compute_stoch_gradient(y, tx, w, batch_size, cost)
        loss = compute_loss(y, tx, w, cost)
        
        # Test of convergence, test_conv counts the number of consecutive iterations for which loss has increased
        if (loss > losses[n_iter]):
            test_conv += 1
            if (test_conv == thresh_test_conv):
                print("Stopped computing because 10 consecutive iterations have involved an increase in loss.")
                return losses, ws
        else:
            test_conv = 0
        
        # updating w
        w = w - gamma * stoch_grad
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        
        # computing distance between new w and former one
        dist_succ_w = np.linalg.norm(ws[len(ws)-1] - ws[len(ws)-2])
        
        # updating the number of iteration
        n_iter += 1

    print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}, cost: {cost}".format(
              bi=n_iter-1, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1], cost = cost))
    
    return losses, ws



# EXERCISE 3 ###################################################################################################################



def least_squares(y, tx):
    """calculate the least squares solution."""
    N = tx.shape[0]
    w = np.linalg.solve(np.transpose(tx).dot(tx), np.transpose(tx).dot(y))
    loss = (1 / (2*N)) * np.linalg.norm(y - tx.dot(w))**2
    
    return loss, w



# Build the feature matrix with polynomial basis
def build_poly(x, degree):
    """
    polynomial basis functions for input data x, for j=0 up to j=degree.
    return tx
    """
    # Exception handling if x has only one column: cannot call x.shape[1]
    try:
        D = x.shape[1]
    except IndexError:
        D = 1
        
    N = x.shape[0]
    
    # First column is offset column of 1
    tx = np.ones(N)
    for d in range(degree):
        tx = np.c_[tx, x**(d+1)]
            
    return tx






def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    returns [x_train, y_train], [x_val, y_val]
    """
    # set seed
    np.random.seed(seed)
    
    N = x.shape[0]
    
    shuffle_indices = np.random.permutation(np.arange(N))
    shuffled_y = y[shuffle_indices]
    shuffled_x = x[shuffle_indices]
    
    split_index = round(N * ratio);
    
    y_train = shuffled_y[:split_index]
    x_train = shuffled_x[:split_index]
    
    y_val = shuffled_y[split_index:]
    x_val = shuffled_x[split_index:]
    
    return [x_train, y_train], [x_val, y_val]





def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = tx.shape[0]
    D = tx.shape[1]
    lambda_prime = lambda_ * 2 * N
    I = np.diag(np.ones(D))
    w = np.linalg.solve(np.transpose(tx).dot(tx) + lambda_prime * I, np.transpose(tx).dot(y))
    loss = (1 / (2*N)) * np.linalg.norm(y - tx.dot(w))**2 + lambda_ * np.linalg.norm(w)**2
    #loss = (1 / (2*N)) * np.linalg.norm(y - tx.dot(w))**2 + lambda_ * np.linalg.norm(w)**2
    return loss, w






# EXERCISE 4 #############################################################################################################

def build_k_indices(y, k_fold, seed):
    """
    build k indices for k-fold.
    return np.array(k_indices)
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)



def cross_validation(y, x, k_indices, k, lambda_, degree, algo = "mse"):
    """
    return the loss of ridge regression.
    k_indices defines the partition of the data.
    k defines the index of the data partition that is used for testing.
    algo: defines which algorithm to use to fit the data.
        "ls" use least squares.
        "ridge" use reifge regression
    return 
    """
    
    test_indices = k_indices[k]
    x_test = x[test_indices]
    y_test = y[test_indices]
    
    train_indices = k_indices[[x for x in range(k_indices.shape[0]) if x != k]]
    x_train = x[np.ravel(train_indices)]
    y_train = y[np.ravel(train_indices)]

    
    tx_test = build_poly(x_test, degree)
    tx_train = build_poly(x_train, degree)
    
    ridge_train_loss, w = ridge_regression(y_train, tx_train, lambda_)
    
    train_loss = compute_loss(y_train, tx_train, w, cost = "mse", lambda_ = lambda_)
    test_loss = compute_loss(y_test, tx_test, w, cost = "mse", lambda_ = lambda_)

    return train_loss, test_loss



def cross_validation_best_degree_lambda(y, x, degrees, k_fold, seed, lambda_min = -4, lambda_max = 0, nb_lambda = 30, plot = False):
    """
    Finds the degree and lambda that yield the less test rmse thanks to cross validation 
    for a particular partition of the data defined by seed.
    Be careful that we are only working here a a single partition of the data and that it may not be representative for the choice
    of lambda and of the degree. 
    Might be a good idea to compute the rmse over several seeds and to look for the best lambda and degree.
    Take a look at averaged_cross_validation_best_degree_lambda for a function 
    looking at an averaged test rmse.
    degrees must be n array containing the degrees that are tested.
    lambda_min, lambda_max and nb_lambda define the interval of test for lambda.
    plot is a boolean allowing the function to plot the rmse
    return best_degree, best_lambda
    """
    lambdas = np.logspace(lambda_min, lambda_max, nb_lambda)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    min_rmse_te = math.inf
    for d in degrees:
        # define lists to store the loss of training data and test data
        rmse_tr = []
        rmse_te = []
        for lambda_ in lambdas:
            tmp_rmse_tr = []
            tmp_rmse_te = []
            for k in range(k_fold):
                loss_train, loss_test = cross_validation(y, x, k_indices, k, lambda_, d)
                tmp_rmse_tr += [np.sqrt(2 * loss_train)]
                tmp_rmse_te += [np.sqrt(2 * loss_test)]
    
            new_rmse_tr = sum(tmp_rmse_tr) / k_fold
            new_rmse_te = sum(tmp_rmse_te) / k_fold
            
            rmse_tr += [new_rmse_tr]
            rmse_te += [new_rmse_te]

            if (new_rmse_te < min_rmse_te):
                best_lambda = lambda_
                best_degree = d
                min_rmse_te = new_rmse_te
            
        if ((d == degree[0] or d == degree[len(degree)-1]) and plot == True):
            cross_validation_visualization(lambdas, rmse_tr, rmse_te)
            plt.figure()

    return best_degree, best_lambda




def averaged_cross_validation_best_degree_lambda(y, x, degrees, k_fold, nb_seed = 100, \
                                                 lambda_min = -4, lambda_max = 0, nb_lambda = 30, plot = False):
    lambdas = np.logspace(lambda_min, lambda_max, nb_lambda)
    
    seeds = range(nb_seed)
    
    # define list to store the variable
    rmse_te = np.empty((len(seeds), len(degrees), len(lambdas)))    
    
    for ind_seed, seed in enumerate(seeds):
        k_indices = build_k_indices(y, k_fold, seed)
        for ind_d, d in enumerate(degrees):
            for ind_lambda_, lambda_ in enumerate(lambdas):
                tmp_rmse_te = []
                for k in range(k_fold):
                    loss_test = cross_validation(y, x, k_indices, k, lambda_, d)[1]
                    tmp_rmse_te += [np.sqrt(2 * loss_test)]

                rmse_te[ind_seed, ind_d, ind_lambda_] = sum(tmp_rmse_te) / k_fold


    rmse_te_mean = np.mean(rmse_te, axis=0)
    
    best_index = np.argmin(rmse_te_mean)
    best_ind_d = best_index // len(lambdas)
    best_ind_lambda = best_index % len(lambdas)
    
    if (plot == True):
        plt.plot(degrees, rmse_te_mean[:, best_ind_lambda], marker=".", color='r', label='test error')
        plt.xlabel("degree")
        plt.ylabel("rmse")
        plt.title("cross validation")
        plt.legend(loc=2)
        plt.grid(True)
        plt.savefig("cross_validation")

        plt.figure()
        plt.semilogx(lambdas, rmse_te_mean[best_ind_d], marker=".", color='r', label='test error')
        plt.xlabel("lambda")
        plt.ylabel("rmse")
        plt.title("cross validation")
        plt.legend(loc=2)
        plt.grid(True)
        plt.savefig("cross_validation")
    
    best_degree = degrees[best_ind_d]
    best_lambda = lambdas[best_ind_lambda]
    
    return best_degree, best_lambda



# EXERCISE 5 #################################################################################################################

def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(t) / (1 + np.exp(t))




def compute_hessian(y, tx, w, cost = "mse", lambda_ = 0):
    N = tx.shape[0]
    if (cost == "mse"):
        return (1 / N) * np.transpose(tx).dot(tx)
    elif (cost == "logistic"):
        S_diag = sigmoid(tx.dot(w)) * (1 - sigmoid(tx.dot(w)))
        S = np.diag(np.squeeze(S_diag))
        return np.transpose(tx).dot(S).dot(tx)
    elif (cost == "reg_logistic"):
        D = tx.shape[1]
        S_diag = sigmoid(tx.dot(w))
        S = np.diag(np.squeeze(S_diag))
        return np.transpose(tx).dot(S).dot(tx) + np.identity(D) * lambda_
    else:
        raise IllegalArgument("Invalid cost argument in compute_hessian")
    return 0




def newton(y, tx, initial_w, max_iters, gamma, cost ="mse", lambda_ = 0, tol = 1e-6, thresh_test_conv = 10, update_gamma = False):
    """Gradient descent algorithm."""
    if (cost not in ["mse", "logistic", "reg_logistic"]):
        raise IllegalArgument("Invalid cost argument in gradient_descent function")
    
    # ensure that w_initial is formatted in the right way if implementing logistic regression
    if (cost == "logistic" or cost == "reg_logistic"):
        try:
            initial_w.shape[1]
        except IndexError:
            initial_w = np.expand_dims(initial_w, 1)
    
        
    # Define parameters to store w and loss
    ws = [initial_w]
    initial_loss = compute_loss(y, tx, initial_w, cost = cost, lambda_ = lambda_)
    losses = [initial_loss]
    w = initial_w
    test_conv = 0
    dist_succ_loss = tol + 1
    n_iter = 0;
    
    while (n_iter < max_iters and dist_succ_loss > tol):
        grad = compute_gradient(y, tx, w, cost = cost, lambda_ = lambda_)
        hess = compute_hessian(y, tx, w, cost = cost)
        
        # updating w
        z = np.linalg.solve(hess, -gamma * grad)
        w = z + w
        loss = compute_loss(y, tx, w, cost = cost, lambda_ = lambda_)
        
        # Test of divergence, test_conv counts the number of consecutive iterations for which loss has increased
        if (loss > losses[-1]):
            test_conv += 1
            if (test_conv >= thresh_test_conv):
                print("Stopped computing because 10 consecutive iterations have involved an increase in loss.")
                return losses, ws
        else:
            test_conv = 0

        # store w and loss
        ws.append(w)
        losses.append(loss)
     
        # update distance between two successive w
        dist_succ_loss = np.abs(losses[-1] - losses[-2])
        
        # update n_iter: number of iterations
        n_iter += 1
        
        # update gamma if update_gamma is True
        if (update_gamma == True):
            gamma = 1 / (1 + n_iter) * gamma
    
    # Printing the results
    try:
        initial_w.shape[1]
        print("Newton({bi}/{ti}): loss={l}, w0={w0}, w1={w1}, cost: {cost}".format(
              bi=n_iter-1, ti=max_iters - 1, l=loss, w0=w[0,0], w1=w[1,0], cost = cost))
    except (IndexError, AttributeError):
        print("Newton({bi}/{ti}): loss={l}, w0={w0}, w1={w1}, cost: {cost}".format(
                  bi=n_iter-1, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1], cost = cost))
    return losses, ws











# TESTS #######################################################################################################################




def test_ls_grid_GD_SGD():
    height, weight, gender = load_data_from_ex02(sub_sample=False, add_outlier=False)
    x, mean_x, std_x = standardize(height)
    y, tx = build_model_data(x, weight)
    
    ls_loss, ls_w = least_squares(y, tx)
    
    w0_grid_test = np.linspace(-100, 100, 100)
    w1_grid_test = np.linspace(-100, 100, 100)
    grid_loss, grid_w = grid_search(y, tx, w0_grid_test, w1_grid_test)
    
    initial_w = np.array([0, 0])
    gamma_GD = 0.7
    gamma_GD_mae = 10
    max_iters = 500
    GD_loss, GD_w = gradient_descent(y, tx, initial_w, max_iters, gamma_GD, cost='mse', tol=1e-2, thresh_test_conv=10)
    GD_loss_mae, GD_w_mae = gradient_descent(y, tx, initial_w, max_iters, gamma_GD_mae, cost='mae', tol=1e-2, thresh_test_conv=10)

    gamma_SGD = 0.1
    gamma_SGD_mae = 2
    max_iters = 100
    batch_size = 200
    SGD_loss, SGD_w = stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma_GD, cost='mse', tol=1e-2, thresh_test_conv=10)
    SGD_loss_mae, SGD_w_mae = stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma_GD_mae, cost='mae', tol=1e-2, thresh_test_conv=10)
    
    
    print("ls_w:", ls_w)
    print("grid_w:", grid_w)
    print("GD_w:", GD_w[len(GD_w)-1])
    print("GD_w_mae:", GD_w_mae[len(GD_w_mae)-1])
    print("SGD_w:", SGD_w[len(GD_w_mae)-1])
    print("SGD_w_mae", SGD_w_mae[len(GD_w_mae)-1])
    
    return 0;






# functions to test test_ls_grid_GD_SGD
def load_data_from_ex02(sub_sample=False, add_outlier=False):
    """Load data and convert it to the metrics system."""
    path_dataset = "height_weight_genders.csv"
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1, 2])
    height = data[:, 0]
    weight = data[:, 1]
    gender = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[0],
        converters={0: lambda x: 0 if b"Male" in x else 1})
    # Convert to metric system
    height *= 0.025
    weight *= 0.454

    # sub-sample
    if sub_sample:
        height = height[::50]
        weight = weight[::50]

    if add_outlier:
        # outlier experiment
        height = np.concatenate([height, [1.1, 1.2]])
        weight = np.concatenate([weight, [51.5/0.454, 55.3/0.454]])

    return height, weight, gender

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx











def train_test_split_demo(x, y, degree, ratio, seed):
    """polynomial regression with different split ratios and different degrees."""
    train_set, val_set = split_data(x, y, ratio, seed)
    
    x_train = train_set[0]
    x_val = val_set[0]
    
    y_train = train_set[1]
    y_val = val_set[1]
    
    tx_train = build_poly(x_train, degree)
    tx_val = build_poly(x_val, degree)
        
    loss, w = least_squares(y_train, tx_train)
    
    train_loss = compute_loss(y_train, tx_train, w, cost = "mse")
    val_loss = compute_loss(y_val, tx_val, w, cost = "mse")
    
    rmse_tr = np.sqrt(2 * train_loss)
    rmse_te = np.sqrt(2 * val_loss)

    print("proportion={p}, degree={d}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(
          p=ratio, d=degree, tr=rmse_tr, te=rmse_te))


    
    
    
    
def test_newton_gradient_descent_reg_logistic():
    # load data.
    height, weight, gender = load_data_from_ex02()

    # build sampled x and y.
    seed = 1
    y = np.expand_dims(gender, axis=1)
    X = np.c_[height.reshape(-1), weight.reshape(-1)]
    y, X = sample_data(y, X, seed, size_samples=200)
    x, mean_x, std_x = standardize(X)
    
    print(y.shape)
    print(x.shape)
    
    max_iter = 10000
    gamma_gd = 0.01
    gamma_newt = 4.5
    lambda_ = 0.1
    threshold = 1e-8

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    
    initial_w = np.zeros((tx.shape[1], 1))
    
    loss, w = gradient_descent(y, tx, initial_w, max_iter, gamma_gd, cost='reg_logistic', lambda_=lambda_, tol=threshold, thresh_test_conv=10, update_gamma=False)
    
    loss, w = newton(y, tx, initial_w, max_iter, gamma_newt, cost='reg_logistic', lambda_=lambda_, tol=threshold, thresh_test_conv=10, update_gamma=False)
    
    return 0
    
    
    
    
    
def sample_data(y, x, seed, size_samples):
    """sample from dataset."""
    np.random.seed(seed)
    num_observations = y.shape[0]
    random_permuted_indices = np.random.permutation(num_observations)
    y = y[random_permuted_indices]
    x = x[random_permuted_indices]
    return y[:size_samples], x[:size_samples]
    
    
    
    
    
    
    
# EXCEPTIONS ##################################################################################################################
    
class IllegalArgument(Exception):
    pass
