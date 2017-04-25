""" Module with functionality to try out different classifiers """
import numpy as np
import cv2
import matplotlib.pyplot as plt

ERR_VIS =True

def train_svm(tr_feat_targ, C=2.0, gamma=3.0, iterations=100, model_fpath=""):
  """
  Function trains svm using given training set and returns a model to save
  """

  # Initialize and set svm type & parameters
  svm = cv2.ml.SVM_create()
  svm.setKernel(cv2.ml.SVM_LINEAR)
  svm.setType(cv2.ml.SVM_C_SVC)
  svm.setC(C)
  svm.setGamma(gamma)
  tr_feat = tr_feat_targ[:, :-1]
  tr_targ = tr_feat_targ[:, -1]
  tr_targ = tr_targ.astype(np.int32)
  svm.train(tr_feat, cv2.ml.ROW_SAMPLE, tr_targ)
  svm.save(model_fpath)

  # Predict on training samples
  tr_feat  = tr_feat_targ[:, :-1]
  tr_preds = svm.predict(tr_feat)
  return svm, tr_preds


# NOTE: Function might not be needed
def test_svm(feat, svm_model):
  """
  Function tests svm using given test data and model and returns predictions
  """
  retval, preds = svm.predict(feat)
  return preds


def train_test_svm(tr_feat_targ, tst_feat_targ, retrain=True, model_fpath="models/svm.yaml",
                   C=2.0, gamma=3.0, iterations=50):
  """
  Function trains and tests svm using given data and returns training & testing
  errors

  Input:
  -----
  tr_feat_targ,
  tst_feat_targ: [mx(n+1)] matrices containing m samples, n features, targets
  retrain      : Trains new model if True, else loads model from model_fpath

  model_fpath: Path to store / load svm models from
               If there is an older model already in path & retrain is set 'True'
               older model get overwritten
  C, Gamma   : SVM parameters
  iterations : No of iterations to run SVM

  """
  tr_feat     = tr_feat_targ[:, :-1]
  tr_targets  = tr_feat_targ[:, -1]
  tst_feat    = tst_feat_targ[:, :-1]
  tst_targets = tst_feat_targ[:, -1]

  # Train/Load trained model and predict on training & testing set
  if retrain:
    svm, tr_preds = train_svm(tr_feat_targ, C, gamma, iterations, model_fpath)
  else:
    svm = cv2.SVM()
    svm_model = svm.load(model_fpath)
    tr_preds  = test_svm(tr_feat, svm)

  tst_preds = svm.predict(tst_feat)

  # Compute erfloat32rors
  tr_acc = (tr_targets == tr_preds[1].reshape(tr_preds[1].shape[0],)).astype(np.uint8)
  tst_acc = (tst_targets == tst_preds[1].reshape(tst_preds[1].shape[0],)).astype(np.uint8)
  mean_tr_acc  = tr_acc.sum() / tr_acc.shape[0] * 100   # Mean training accuracy
  mean_tst_acc = tst_acc.sum() / tst_acc.shape[0] * 100 # Mean testing accuracy
  np.savetxt("tr_preds.txt", tr_preds[1], fmt="%.1f")
  print "Training predictions written to tr_preds.txt"
  np.savetxt("tst_preds.txt", tst_preds[1], fmt="%.1f")
  print "Testing predictions written to tst_preds.txt"
  print "******calculating training and testing accuracy******"
  print "Training Accuracy: ", mean_tr_acc
  print "Testing  Accuracy: ", mean_tst_acc

  # Plot errors
  if ERR_VIS:
    plt.ioff()
    plt.plot(tr_acc)
    plt.savefig("results/tr_acc.png") 
    plt.close()
    plt.plot(tst_acc)
    plt.savefig("results/tst_acc.png")
    print "testing accuracy written to results/tst_acc.png"
    plt.close()

  return mean_tr_acc, mean_tst_acc


if __name__ == "__main__":
  tr_feat_targ_path  = "data/tr_feat.txt"
  tst_feat_targ_path = "data/tst_feat.txt"
  model_fpath        = "model/linsvm.yaml"

  # If features are precomputed, load from specified path
  # Meant to be useful when classifier parameters are under tuning
  tr_feat_targ  = np.loadtxt(tr_feat_targ_path)
  tst_feat_targ = np.loadtxt(tst_feat_targ_path)

  train_test_svm(tr_feat_targ, tst_feat_targ, model_fpath)
