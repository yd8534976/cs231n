import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  num_train = scores.shape[0]
  num_class = scores.shape[1]
  for i in xrange(num_train):
    maxscore = np.max(scores[i])
    scores[i] -= maxscore
    denominator = np.sum(np.exp(scores[i]))
    p = np.exp(scores[i]) / denominator
    loss -= np.log(p[y[i]])
    for j in xrange(num_class):
        if j == y[i]:
            dW[:, j] -= X[i] * (1 - p[j])
        else:
            dW[:, j] += X[i] * p[j]
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  maxscores = np.max(scores, axis=1)
  scores -= np.reshape(maxscores, (-1, 1))
  num_train = scores.shape[0]
  num_class = scores.shape[1]
  p = np.exp(scores)
  p = p / np.reshape(np.sum(p, axis=1), (-1, 1))
  loss -= np.sum(np.log(p[range(0, num_train), y])) / num_train
  loss += 0.5 * reg * np.sum(W * W)
  temp = p
  temp[range(num_train), y] -= 1
  dW += X.T.dot(temp) / num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

