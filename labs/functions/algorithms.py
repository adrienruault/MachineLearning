
import numpy as np
from helpers import *





def compute_loss(y, tx, w, cost = "mse"):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    N = tx.shape[0]
    error_vec = y - tx.dot(w)
    if (cost == "mse"):
        loss = (1 / (2*N)) * np.linalg.norm(error_vec)**2
    elif (cost == "mae"):
        loss = (1 / N) * np.sum(np.abs(error_vec))
    else:
        print("Invalid cost argument in compute_loss")
        raise NotImplementedError
    return loss






def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    losses = np.zeros((len(w0), len(w1)))
    for i in range(len(w0)):
        for j in range(len(w1)):
            losses[i, j] = compute_loss(y, tx, np.array([w0[i], w1[j]]))
    
    argmin_index = np.argmin(losses);
    best_i = argmin_index % len(w0)
    best_j = argmin_index // len(w1)
    
    return losses, w0[best_i], w1[best_j]




def compute_gradient(y, tx, w, cost = "mse"):
    """Compute the gradient."""
    N = tx.shape[0];
    error_vec = y - tx.dot(w)
    if (cost == "mse"):
        grad = -(1/N) * np.transpose(tx).dot(error_vec)
    elif (cost == "mae"):
        sub_grad = [0 if error_vec[i] == 0 else error_vec[i] / np.abs(error_vec[i]) for i in range(N)]
        grad = -(1 / N) * np.transpose(tx).dot(sub_grad)
    else:
        print("Invalid argument in compute_gradient function.")
        raise NotImplementedError
    return grad




# Be careful, gamma of 2 for MSE doesn't converge
def gradient_descent(y, tx, initial_w, max_iters, gamma, cost ="mse", tol = 1e-6, thresh_test_conv = 10):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    initial_loss = compute_loss(y, tx, initial_w, cost)
    losses = [initial_loss]
    w = initial_w
    test_conv = 0
    dist_succ_w = tol + 1
    n_iter = 0;
    while (n_iter < max_iters and dist_succ_w > tol):
        grad = compute_gradient(y, tx, w, cost)
        loss = compute_loss(y, tx, w, cost)
        
        # Test of convergence, test_conv counts the number of consecutive iterations for which loss has increased
        if (loss > losses[n_iter]):
            test_conv += 1
            if (test_conv >= thresh_test_conv):
                print("Stopped computing because 10 consecutive iterations have involved an increase in loss.")
                return losses, ws
        else:
            test_conv = 0
        
        # updating w
        w = w - gamma * grad
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
        # update distance between two successive w
        dist_succ_w = np.linalg.norm(ws[len(ws)-1] - ws[len(ws)-2])
        
        # update n_iter: number of iterations
        n_iter += 1

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
        print("Invalid cost argument in compute_stoch_gradient")
        raise NotImplementedError
    
    return stoch_grad






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
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
        # computing distance between new w and former one
        dist_succ_w = np.linalg.norm(ws[len(ws)-1] - ws[len(ws)-2])
        
        # updating the number of iteration
        n_iter += 1

    return losses, ws











