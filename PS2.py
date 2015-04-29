import csv
import math
from operator import itemgetter


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

def biningNumericalData(data):
    binNum=10
    splitNumWinPerc = float(1)/binNum
    splitNumTemp = float(100)/binNum
    splitNumRunDif = float(150)/binNum
    for i in range(0,binNum):
        for rownum in range(len(data)):
            if data[rownum][0]>=i*splitNumWinPerc and data[rownum][0]<((i+1)*splitNumWinPerc):
                    data[rownum][0]=i+1
            if data[rownum][1]>=i*splitNumWinPerc and data[rownum][1]<((i+1)*splitNumWinPerc):
                    data[rownum][1]=i+1
            if data[rownum][3]>=i*splitNumTemp and data[rownum][3]<((i+1)*splitNumTemp):
                    data[rownum][3]=i+1
            if data[rownum][11]>=-25+i*splitNumRunDif and data[rownum][11]<(-25+(i+1)*splitNumRunDif):
                    data[rownum][11]=i+1
            if data[rownum][12]>=-25+i*splitNumRunDif and data[rownum][12]<(-25+(i+1)*splitNumRunDif):
                    data[rownum][12]=i+1
    return data


def preProcessData(FileName):
    d = readCSVFile(FileName)
    cd = cleanData(d)
    dataToTrain = biningNumericalData(cd)
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

def infoGain(data, col):
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
        e=-probLose[elemNum]*math.log(probLose[elemNum],2)-probWin[elemNum]*math.log(probWin[elemNum],2)
        entropyPerNominal.append(e)
    m = min(entropyPerNominal)
    i = entropyPerNominal.index(m)
    return [m,i]


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
