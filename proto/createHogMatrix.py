import numpy as np
import csv

def createHMatrix(a):
  '''input : two class indices i and j 
   output :  a 2 d array with hog va;ues of images stacked up one over the other for classes i and j 
  '''
  classIndex = 0	
  HOGfilepath = "GTSRB_Final_Training_HOG/GTSRB/Final_Training/HOG/HOG_01/"
  gtFilePath = "final_training/Final_Training/Images/"
  imageNumber = 0
# find the number of images
  for classIndex in a:
    gtClassFileName=gtFilePath+format(classIndex,'05d')+"/"+'GT-'+ format(classIndex, '05d') + '.csv'
    gtFile = open(gtClassFileName,'r')
    gtReader = csv.reader(gtFile, delimiter=';')
    gtReader.next()
    for row in gtReader:
      imageNumber = imageNumber+1
    gtFile.close()	
    print imageNumber
# end of finding number of images

  twoDHogVals = np.zeros((imageNumber,1569))
  index = 0
#for writing the hog values into a 2 D array with each row has hog vaklues of each image	
  for classIndex in a:
    gtClassFileName=gtFilePath+format(classIndex,'05d')+"/"+'GT-'+ format(classIndex, '05d') + '.csv'
    gtFile = open(gtClassFileName,'r')
    gtReader = csv.reader(gtFile, delimiter=';')
    gtReader.next()
    for row in gtReader:
      cindex = row[0].split("_")
      cIndex=int(cindex[0])
    gtFile.close()	
    print classIndex, " " ,cIndex	
    for c in range(0,cIndex+1):
      for k in range(0,30):
        fileName = HOGfilepath+format(classIndex,'05d')+"/"+format(c,'05d')+"_"+format(k,'05d')+".txt"
        oneDHogVals = np.array([])
        file = open(fileName,'r')
        for line in file:
          oneDHogVals = np.append(oneDHogVals,line)
        oneDHogVals = np.append(oneDHogVals,classIndex)
        twoDHogVals[index] = oneDHogVals
        index = index+1	
        file.close()		
    #print twoDHogVals
  #np.savetxt('proto/data/hog.csv', twoDHogVals, delimiter=', ')
  return twoDHogVals
# end of function createMatrix
