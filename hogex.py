import numpy as np
import csv
import proto.ml.classifiers as cls
import proto.createHogMatrix as hm


if __name__=="__main__":
  a=[0,37]
  twoDHogVals = hm.createHMatrix(a)
  twoDHogVals = twoDHogVals.astype(np.float32)
  cls.train_test_svm(twoDHogVals, twoDHogVals)
