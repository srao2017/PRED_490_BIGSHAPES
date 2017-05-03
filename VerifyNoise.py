import os
import csv

def readTrainingData(filename):
    
    trainingData =[]
    trainingDataNine =[]
    
    for row in xrange(9): trainingDataNine += [[0]*9]    
    i = 0
    j = 0
    fo = open (filename+str("_9x9_out.csv"), 'wb')
    writer = csv.writer(fo)

    with open(filename+str("_out.txt")) as fi:
        reader = csv.reader(fi)
        for row in reader:
            trainingData = list(row)
            idx = 0
            for i in range(9):
                for j in range(9):
                    trainingDataNine[i][j] = trainingData[idx]
                    idx += 1
                    j += 1
                i += 1

            for k in range(len(trainingDataNine)):
                writer.writerow(trainingDataNine[k])
                if (k+1) % 9 == 0:
                    writer.writerow("")


    fo.close()
    
def main():
    os.chdir("C:/EDUCATION/NWU/490 - Deep Learning/Week 5/Test")
    filename = input("Enter filename: ")
    readTrainingData(filename)
     
 
if __name__ == "__main__": main()