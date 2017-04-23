import numpy as np
import csv
import proto.ml.classifiers as cls
import proto.createHogMatrix as hm


if __name__=="__main__":
  a=range(0,43)
  print a
  twoDTrainHogVals = np.zeros((1,1569))
  twoDTestHogVals = np.zeros((1,1569))
  twoDHogVals = hm.createHMatrix(a)
  i=0	
  twoDHogVals = twoDHogVals.astype(np.float32)
  twoDTrainHogVals = twoDTrainHogVals.astype(np.float32)
  twoDTestHogVals = twoDTestHogVals.astype(np.float32)
  list_2r =[]
  list_1r =[]
  i2 = 0
  i1 = 2
  while(i1<twoDHogVals.shape[0]):
    samp_00 = twoDHogVals[i2, :]
    samp_01 = twoDHogVals[i2+1, :]
    list_2r.append(samp_00)
    list_2r.append(samp_01)
    list_1r.append(twoDHogVals[i1,:])
    i2=i2+3
    i1=i1+3
  print i1
  mat_2r = np.array(list_2r)
  mat_1r = np.array(list_1r)
  print mat_2r.shape, mat_1r.shape
  cls.train_test_svm(mat_2r, mat_1r)
