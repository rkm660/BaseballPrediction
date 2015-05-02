import csv
import math
from operator import itemgetter
import copy
import numpy
from collections import Counter
import random

#######################################################
"""
readCSVFile opens an input csv file and returns the data
in a list format

"""
########################################################




def readCSVFile(fileName):
    with open(fileName, 'rb') as f:
        reader = csv.reader(f)
        trainData = map(list, reader)
    return trainData





#######################################################
"""
cleanData parses the input data -- chcking for question
marks and negative values

"""
########################################################

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
        if cleanData[r][2]<0:
            cleanData[r][2]=2
        if cleanData[r][0]>=0 and cleanData[r][0]<=1:
            if cleanData[r][1]>=0 and cleanData[r][1]<=1:
                if cleanData[r][4]>=0 and cleanData[r][5]>=0 and cleanData[r][6]>=0 and cleanData[r][7]>=0 and cleanData[r][8]>=0 and cleanData[r][9]>=0:
                    if cleanData[r][10] in [0,1] and cleanData[r][13] in [0,1]:
                        cd.append(cleanData[r])
    return cd



#######################################################
"""
findColWithNumericalData takes in a data object and
isolates the columns containing numerical data
"""
########################################################



def findColWithNumericalData(data):
    d=copy.deepcopy(data)
    d = numpy.transpose(d)
    rowsWithNumericalData=[]
    for row in range(len(data[0])):
        if len(set(d[row]))>20:
            rowsWithNumericalData.append(row)
    return rowsWithNumericalData
            

#######################################################
"""
binningNumericalData takes a data object as input
and returns the data with the numberical columns binned
into ranges

"""
########################################################

    
def binningNumericalData(data):
    colsToBin=findColWithNumericalData(data)
    binNum=5
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


#######################################################
"""
preProcessData opens an input csv file and returns the
parsed and cleaned data

"""
########################################################


def preProcessData(FileName):
    d = readCSVFile(FileName)
    cd = cleanData(d)
    dataToTrain = binningNumericalData(cd)
    return dataToTrain

#######################################################
"""
targetEntropy calculates and returns the entropy for
the target attribute

"""
########################################################


def targetEntropy(data):
    winCount=0
    loseCount=0
    for rownum in range(1,len(data)):
        if data[rownum][13]==1:
            winCount+=1
        else:
            loseCount+=1
    if winCount==0:
        probWin=0
    else:
        probWin=float(winCount)/(winCount+loseCount)
    if loseCount==0:
        probLose=0
    else:
        probLose=float(loseCount)/(winCount+loseCount)
    if probWin<=0 or probLose <=0:
        entropy=0
    else:
        entropy = -probLose*math.log(probLose,2)-probWin*math.log(probWin,2)
    return entropy


#######################################################
"""
attributeEntropy takes data and a column as input, and
returns the entropy for that specific column

"""
########################################################

def attributeEntropy(data, col):
    nominalVals=[]
    for rownum in range(1,len(data)):
        if data[rownum][col] not in nominalVals:
            nominalVals.append(data[rownum][col])
    nominalVals=sorted(nominalVals)
    winCounts=[]
    loseCounts=[]
    nomValCounts=[]
    for elem in nominalVals:
        wc=0
        lc=0
        nvc=0
        for rownum in range(1,len(data)):
            if data[rownum][col]==elem:
                nvc+=1
                if data[rownum][13]==1:
                    wc+=1
                else:
                    lc+=1
        winCounts.append(wc)
        loseCounts.append(lc)
        nomValCounts.append(nvc)
    probWin=[]
    probLose=[]
    probBinOccurs=[]
    for elemNum in range(len(winCounts)):
        probWin.append(float(winCounts[elemNum])/(sum(winCounts)+sum(loseCounts)))
        probLose.append(float(loseCounts[elemNum])/(sum(winCounts)+sum(loseCounts)))
        probBinOccurs.append(float(nomValCounts[elemNum])/(sum(nomValCounts)))
    entropyPerNominal=[]
    for elemNum in range(len(probWin)):
        if probLose[elemNum]<=0 or probWin[elemNum]<=0:
            e=0
        else:
            e=-probLose[elemNum]*math.log(probLose[elemNum],2)-probWin[elemNum]*math.log(probWin[elemNum],2)
            e=e*probBinOccurs[elemNum]
        entropyPerNominal.append(e)
    s=sum(entropyPerNominal)
    return (s)

#######################################################
"""
infoGain calculates the information game in bits for
each attribute and returns a list of said values

"""
########################################################



def infoGain(data, listOfAttr):
    allAttrNames=['winpercent', ' oppwinpercent', ' weather', ' temperature', ' numinjured', ' oppnuminjured', ' startingpitcher', ' oppstartingpitcher', ' dayssincegame', ' oppdayssincegame', ' homeaway', ' rundifferential', ' opprundifferential']
    target=targetEntropy(data)
    infoGainList=[]
    for col in range(len(allAttrNames)):
        if allAttrNames[col] in listOfAttr:
            l=attributeEntropy(data,col)
            infoGainList.append([target-l,col])
    return infoGainList
    

#######################################################
"""
createTree returns a tree in a nested dictionary format

"""
########################################################


def createTree(data, attributes, target):
    allAttrNames=['winpercent', ' oppwinpercent', ' weather', ' temperature', ' numinjured', ' oppnuminjured', ' startingpitcher', ' oppstartingpitcher', ' dayssincegame', ' oppdayssincegame', ' homeaway', ' rundifferential', ' opprundifferential']
    d = data
    vals = [instance[target] for instance in data]
    default = max(set(vals), key=vals.count)
    attributeNames = copy.deepcopy(attributes)
    if not d or (len(attributeNames) - 1) <= 0:
        return default
    elif targetEntropy(d)==0:
        return default
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:

        infoGainList = infoGain(d,attributeNames)
        best=infoGainList[0][0]
        bestCol=infoGainList[0][1]
        for i in range(1,len(infoGainList)):
            if best<infoGainList[i][0]:
                best=infoGainList[i][0]
                bestCol=infoGainList[i][1]
        #tree = {best:{}}
        tree = {}
        for val in getUniqueValues(d,bestCol):
            subtree = createTree(getInstances(d,bestCol,val),
                [attr for attr in attributeNames if attr != allAttrNames[bestCol]],
                target)

            #print attributeNames[bestCol]
            tree[allAttrNames[bestCol]+str(float(val))] = subtree
        
        return tree

#######################################################
"""
getInstances is a helper function for createTree
that gets an object containing an attribute "bin"
"""
########################################################

def getInstances(data, best, val):
    returnList = []
    for record in data:
        value = record[best]
        if (value == val):
            returnList.append(record)
    return returnList

#######################################################
"""
getUniqueValues is a helper function for createTree
that gets a list of the unique values in the best column
for traversing

"""
########################################################


def getUniqueValues(data,best):
    returnList = []
    for record in data:
        returnList.append(record[best])
    return list(set(returnList))


#######################################################
"""
testAccuracy attempts to measure the accuracy of a given model

"""
########################################################

def testAccuracy(model,validationSet):
    vd=preProcessData(validationSet)
    attributeNames=vd[0]
    numCorrect=0
    numIncorrect=0
    for row in range(1,len(vd)):
        c=validateRow(model,vd[row],[])
        if not c:
            numCorrect+=1
            #print 'correct '+ str(numCorrect)
        else:
            numIncorrect+=1
            #print 'wrong '+ str(numIncorrect)
    print numCorrect
    print numIncorrect
    print 'percent accuracy: ' + str(float(numCorrect)/(numIncorrect+numCorrect))


#######################################################
"""
validateRow checks to see whether a given instance of data
in the training set is correct

"""
########################################################


def validateRow(model,row,keys):
    newKeys = keys
    allAttrNames=['winpercent', ' oppwinpercent', ' weather', ' temperature', ' numinjured', ' oppnuminjured', ' startingpitcher', ' oppstartingpitcher', ' dayssincegame', ' oppdayssincegame', ' homeaway', ' rundifferential', ' opprundifferential']
    if newKeys:
        branch = copy.deepcopy(model)
        for elem in newKeys:
           branch=branch[elem]
    else:
        branch = model
    if isinstance(branch,dict):
        k=branch.keys()
        keyVals=[]
        for elem in k:
            keyName = elem[0:len(elem)-3]
            keyVals.append(float(elem[len(elem)-3:len(elem)]))
        if keyName in allAttrNames:
            col = allAttrNames.index(keyName)
            rowVal = row[col]
            if rowVal in keyVals:
                for e in keyVals:
                    if e == rowVal:
                        newKeys.append(keyName+str(e))
                return validateRow(model,row,newKeys)
            else:
                return True
    else:
        if branch!=row[len(allAttrNames)]:
            t=False
        else:
            t=True
        return t

raw = preProcessData("btrain.csv")
attributes = raw[0]

data = raw[1:]
target = len(attributes)-1
result = createTree(data, attributes, target)

#######################################################
"""
normalForm returns a list of numerous unique routes
whose target is 1

"""
########################################################

def normalForm(lst):
    routes = []
    while (len(routes) <= 16):
        route = printBooleanForm(result, [])
        if (route[len(route)-1] == "1.0" and route not in routes):
            routes.append(route)
    return routes
    
#######################################################
"""
printNormalForm prints the data from normalForm in
a disjunctive normal form

"""
########################################################

def printNormalForm(lst):
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            if (j < len(lst[i])):
                print(lst[i][j] + " AND ")
            else:
                print(lst[i][j])
        if (i < len(lst[i])):
            print(" OR ")

#######################################################
"""
printBooleanForm iterates the tree to get the pathway to
a leaf

"""
########################################################

def printBooleanForm(model,keys):
    newKeys = copy.deepcopy(keys)
    allAttrNames=['winpercent', ' oppwinpercent', ' weather', ' temperature', ' numinjured', ' oppnuminjured', ' startingpitcher', ' oppstartingpitcher', ' dayssincegame', ' oppdayssincegame', ' homeaway', ' rundifferential', ' opprundifferential']
    if newKeys:
        branch = copy.deepcopy(model)
        for elem in newKeys:
           branch=branch[elem]                
    else:
        branch = copy.deepcopy(model)
    if isinstance(branch,dict):
        k=branch.keys()
        ranNum = random.randrange(0,len(k))
        newKeys.append(k[ranNum])
        return printBooleanForm(model,newKeys)
    else:
        newKeys.append(str(branch))
        return newKeys
    
x = normalForm(printBooleanForm(result,[]))
printNormalForm(x)

#######################################################
"""
pruning takes one of the branches and prunes it

"""
########################################################

def pruning(model):    
    prunedModel=copy.deepcopy(model)
    path=[]
    branch=model
    k=branch.keys()
    r=random.randint(0,len(k))
    prunedModel.pop(k[r])
    return prunedModel
