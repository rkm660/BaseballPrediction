import csv
import math
from operator import itemgetter
import copy
import numpy
from collections import Counter

def readCSVFile(fileName):
    with open(fileName, 'rb') as f:
        reader = csv.reader(f)
        trainData = map(list, reader)
    return trainData

##Change cleanData so that it takes the avg for numerical vals and mode for nominal
def cleanData(data):
    numericalCols = findColWithNumericalData(data) 
    rd=[]
    modes=[]
    medians=[]
    d=copy.deepcopy(data[1:len(data)])
    for rowNum in range(len(d)):
        if '?' not in d[rowNum]:
            r=[]
            for elem in range(len(d[rowNum])):
                r.append(float(d[rowNum][elem]))
            rd.append(r)
    rd = numpy.transpose(rd)
    for row in range(len(data[0])):
        mo=Counter(rd[row])
        modes.append(mo.most_common(1)[0][0])
        medians.append(float(numpy.median(rd[row])))
    for rowN in range(1,len(data)):
        if '?' in data[rowN]:
            colQ=[i for i,x in enumerate(data[rowN]) if x == '?']
            for col in colQ:
                if col in numericalCols:
                    data[rowN][col]=medians[col]
                else:
                    data[rowN][col]=modes[col]
    cleanData=[]
    cleanData.append(data[0])
    for rownum in range(1,len(data)):
        row=data[rownum]
        r=[]
        for elem in range(len(row)):
            r.append(float(row[elem]))
        cleanData.append(r)

    cd=[]
    cd.append(cleanData[0])
    for r in range(1,len(cleanData)):
        if cleanData[r][0]>=0 and cleanData[r][0]<=1:
            if cleanData[r][1]>=0 and cleanData[r][1]<=1:
                if cleanData[r][4]>=0 and cleanData[r][5]>=0 and cleanData[r][6]>=0 and cleanData[r][7]>=0 and cleanData[r][8]>=0 and cleanData[r][9]>=0:
                    if cleanData[r][10] in [0,1] and cleanData[r][13] in [0,1]:
                        cd.append(cleanData[r])
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
        maxi=max(d[col])
        mini=min(d[col])
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
            d['binVal'] = l[1][elemNum]
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

        infoGainData = infoGain(data)
        infoGainDict = {}
        for obj in infoGainData:
           infoGainDict[str(obj['col']) + str(int(obj['nominalVal']))] = obj["infoGain"]
        best = max(infoGainDict, key=infoGainDict.get)
        print(best)
        for obj in infoGainData:
            if (str(obj['col']) + str(int(obj['nominalVal'])) == best):
                bestCol = obj["col"]
                print(obj)
        tree = {best:{}}
        for val in getUniqueValues(data,bestCol):
            subtree = createTree(getInstances(data,bestCol,val),
                [attr for attr in attributes if attr != bestCol],
                target)

            tree[best][val] = subtree



        
        return tree



def getInstances(data, best, val):
    returnList = []
    for record in data:
        value = record[best]
        if (value == val):
            returnList.append(record)
    return returnList


def getUniqueValues(data,best):
    returnList = []
    for record in data:
        returnList.append(record[best])
    return list(set(returnList))

    

raw = preProcessData("btrain.csv")
attributes = raw[0]
data = raw[1:]
target = len(attributes)-1
print(createTree(data, attributes, target))




