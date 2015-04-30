import csv
import math
from operator import itemgetter
import copy
import numpy

def readCSVFile(fileName):
    with open(fileName, 'rb') as f:
        reader = csv.reader(f)
        trainData = map(tuple, reader)
    return trainData

def getAttributeNames(data):
    d = {}
    row1=data[0]
    for attrnum in range(len(row1)):
        d[attrnum] = row1[attrnum]
    return d

##Change cleanData so that it takes the avg for numerical vals and mode for nominal
def cleanData(data):
    cleanData=[]
    for rowNum in range(len(data)):
        if rowNum<1:
            cleanData.append(data[rowNum])
        else:
            if '?' not in data[rowNum]:
                row = data[rowNum]
                r=[]
                for elem in row:
                    r.append(float(elem))
                cleanData.append(r)
    cd=[]
    for rownum in range(len(cleanData)):
        if rownum<1:
            cd.append(cleanData[rownum])
        else:
            if cleanData[rownum][0]>=0 and cleanData[rownum][1]>=0:
                if cleanData[rownum][0]<=1 and cleanData[rownum][1]<=1:
                    cd.append(cleanData[rownum])
    return cd

def findColWithNumericalData(data):
    d=copy.deepcopy(data)
    d = numpy.transpose(d)
    rowsWithNumericalData=[]
    for row in range(len(data[0])):
        if len(set(d[row]))>20:
            rowsWithNumericalData.append(row)
    return rowsWithNumericalData
            
    
def binningNumericalData(data):
    colsToBin=findColWithNumericalData(data)
    binNum=10
    d=copy.deepcopy(data[1:len(data)])
    d = numpy.transpose(d)
    for col in colsToBin:
        av=numpy.mean(d[col])
        maxi=max(d[col])+av/binNum
        mini=min(d[col])-av/binNum
        split=float(maxi-mini)/binNum
        for i in range(binNum):
            for row in range(1,len(data)):
                if data[row][col]>=mini+i*split and data[row][col]<mini+(i+1)*split:
                     data[row][col]=i+1
    return data


def preProcessData(FileName):
    d = readCSVFile(FileName)
    cd = cleanData(d)
    dataToTrain = binningNumericalData(cd)
    return dataToTrain

def targetEntropy(data):
    winCount=0
    loseCount=0
    for rownum in range(1,len(data)):
        if data[rownum][13]==1:
            winCount+=1
        else:
            loseCount+=1
    probWin=float(winCount)/(winCount+loseCount)
    probLose=float(loseCount)/(winCount+loseCount)
    entropy = -probLose*math.log(probLose,2)-probWin*math.log(probWin,2)
    return entropy

def columnEntropy(data, col):
    nominalVals=[]
    for rownum in range(1,len(data)):
        if data[rownum][col] not in nominalVals:
            nominalVals.append(data[rownum][col])
    nominalVals=sorted(nominalVals)
    winCounts=[]
    loseCounts=[]
    for elem in nominalVals:
        wc=0
        lc=0
        for rownum in range(1,len(data)):
            if data[rownum][col]==elem:
                if data[rownum][13]==1:
                    wc+=1
                else:
                    lc+=1
        winCounts.append(wc)
        loseCounts.append(lc)
    probWin=[]
    probLose=[]
    for elemNum in range(len(winCounts)):
        probWin.append(float(winCounts[elemNum])/(sum(winCounts)+sum(loseCounts)))
        probLose.append(float(loseCounts[elemNum])/(sum(winCounts)+sum(loseCounts)))
    entropyPerNominal=[]
    for elemNum in range(len(probWin)):
        if probLose[elemNum]==0 or probWin[elemNum]==0:
            e=0
        else:
            e=-probLose[elemNum]*math.log(probLose[elemNum],2)-probWin[elemNum]*math.log(probWin[elemNum],2)
        entropyPerNominal.append(e)
    return (entropyPerNominal, nominalVals, col)

def infoGain(data):
    target=targetEntropy(data)
    infoGainList=[]
    for col in range(len(data[0])):
        l=columnEntropy(data,col)
        for elemNum in range(len(l[0])):
            d={'infoGain':(target-l[0][elemNum])}
            d['nominalVal'] = l[1][elemNum]
            d['col'] = l[2]
            infoGainList.append(d)
    return infoGainList
    

def createTree(data, attributes, target):
   
    data    = data[:]
    vals    = [instance[target] for instance in data]
    default = max(set(vals), key=vals.count)

    if not data or (len(attributes) - 1) <= 0:
        return default

    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        
        entropyMinusgainList = []
        entropy = targetEntropy(data)
        for attr in attributes:
            print(attr)
            entropyMinusgainList.append(entropy-infoGain(attr,attributes.index(attr)))
        best = max(entropyMinusgainList)

        tree = {best:{}}

    return tree

def getInstances(data,best,val):
    returnData = []
    for instance in data:
        if (instance[best] == val):
            returnData.append(instance)
    return returnData
