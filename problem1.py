import csv
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import pylab

# estimate Sigma = SUM(sigma) / 96 (12 * 8)
# Pr(dBm) = Pt(dBm) + K_ref - Eta * 10lg(d / d0) + sigma * N

# estimated_sigma = SUM(est_sigma) / valid_number_of_column
sum_of_sigma = 0
valid_number_of_Pr = 0

def processSingleColSig(col):
    if len(col) != 0:
        mean = np.mean(col)
        sum = 0
        for x in np.nditer(col):
            sum += (x - mean)**2
        return sum
    else:
        return 0

dictRec = {}
with open('/Users/yizhuolu/Desktop/EE597/HW#1/HW1_Data/receiverXY.csv') as file1:
    receiver = csv.reader(file1)
    i = 1
    for row in receiver:
        dictRec[i] = row
        i += 1
# for key in dictRec.keys():
#     print(dictRec.get(key))
dictTrans = {}
with open('/Users/yizhuolu/Desktop/EE597/HW#1/HW1_Data/transmitterXY.csv') as file2:
    transmitter = csv.reader(file2)
    i = 1
    for row in transmitter:
        dictTrans[i] = row
        i += 1

dictData = {}
d = 0

data_set = pd.read_csv('/Users/yizhuolu/Desktop/EE597/HW#1/HW1_Data/wifiExp7.csv', header=None, index_col=False)
# process 8 different est_sigma from 8 different columns in the first experiment.
for column in range(1, 9):
    d = np.sqrt((float(dictTrans[1][1]) - float(dictRec[column][1]))**2 + (float(dictTrans[1][0]) - float(dictRec[column][0]))**2)
    d = 10 * math.log10(d)
    col = np.array(data_set.iloc[:, column])
    invalid = np.where(col == 500.0)
    est_sigma = 0
    if len(invalid[0]) != 0:
        newCol = np.delete(col, invalid)
        est_sigma = processSingleColSig(newCol)
        valid_number_of_Pr += 0 if len(newCol) == 0 else len(newCol)
        if len(newCol) != 0:
            dictData[d] = []
            dictData[d].append(-newCol)
    else:
        est_sigma = processSingleColSig(col)
        valid_number_of_Pr += len(col)
        dictData[d] = []
        dictData[d].append(-col)
    sum_of_sigma += est_sigma
# for key in dictData.keys():
#     print(dictData.get(key)[0][0])

# second experiment
data_set = pd.read_csv('/Users/yizhuolu/Desktop/EE597/HW#1/HW1_Data/wifiExp8.csv', header=None, index_col=False)
for column in range(1, 9):
    d = np.sqrt((float(dictTrans[2][1]) - float(dictRec[column][1]))**2 + (float(dictTrans[2][0]) - float(dictRec[column][0]))**2)
    d = 10 * math.log10(d)
    col = np.array(data_set.iloc[:, column])
    invalid = np.where(col == 500.0)
    est_sigma = 0
    if len(invalid[0]) != 0:
        newCol = np.delete(col, invalid)
        est_sigma = processSingleColSig(newCol)
        valid_number_of_Pr += 0 if len(newCol) == 0 else len(newCol)
        if len(newCol) != 0:
            dictData[d] = []
            dictData[d].append(-newCol)
    else:
        est_sigma = processSingleColSig(col)
        valid_number_of_Pr += len(col)
        dictData[d] = []
        dictData[d].append(-col)
    sum_of_sigma += est_sigma

# third experiment
data_set = pd.read_csv('/Users/yizhuolu/Desktop/EE597/HW#1/HW1_Data/wifiExp9.csv', header=None, index_col=False)
for column in range(1, 9):
    d = np.sqrt((float(dictTrans[3][1]) - float(dictRec[column][1]))**2 + (float(dictTrans[3][0]) - float(dictRec[column][0]))**2)
    d = 10 * math.log10(d)
    col = np.array(data_set.iloc[:, column])
    invalid = np.where(col == 500.0)
    est_sigma = 0
    if len(invalid[0]) != 0:
        newCol = np.delete(col, invalid)
        est_sigma = processSingleColSig(newCol)
        valid_number_of_Pr += 0 if len(newCol) == 0 else len(newCol)
        if len(newCol) != 0:
            dictData[d] = []
            dictData[d].append(-newCol)
    else:
        est_sigma = processSingleColSig(col)
        valid_number_of_Pr += len(col)
        dictData[d] = []
        dictData[d].append(-col)
    sum_of_sigma += est_sigma

# fourth experiment
data_set = pd.read_csv('/Users/yizhuolu/Desktop/EE597/HW#1/HW1_Data/wifiExp10.csv', header=None, index_col=False)
for column in range(1, 9):
    d = np.sqrt((float(dictTrans[4][1]) - float(dictRec[column][1]))**2 + (float(dictTrans[4][0]) - float(dictRec[column][0]))**2)
    d = 10 * math.log10(d)
    col = np.array(data_set.iloc[:, column])
    invalid = np.where(col == 500.0)
    est_sigma = 0
    if len(invalid[0]) != 0:
        newCol = np.delete(col, invalid)
        est_sigma = processSingleColSig(newCol)
        valid_number_of_Pr += 0 if len(newCol) == 0 else len(newCol)
        if len(newCol) != 0:
            dictData[d] = []
            dictData[d].append(-newCol)
    else:
        est_sigma = processSingleColSig(col)
        valid_number_of_Pr += len(col)
        dictData[d] = []
        dictData[d].append(-col)
    sum_of_sigma += est_sigma

# fifth experiment
data_set = pd.read_csv('/Users/yizhuolu/Desktop/EE597/HW#1/HW1_Data/wifiExp11.csv', header=None, index_col=False)
for column in range(1, 9):
    d = np.sqrt((float(dictTrans[5][1]) - float(dictRec[column][1]))**2 + (float(dictTrans[5][0]) - float(dictRec[column][0]))**2)
    d = 10 * math.log10(d)
    col = np.array(data_set.iloc[:, column])
    invalid = np.where(col == 500.0)
    est_sigma = 0
    if len(invalid[0]) != 0:
        newCol = np.delete(col, invalid)
        est_sigma = processSingleColSig(newCol)
        valid_number_of_Pr += 0 if len(newCol) == 0 else len(newCol)
        if len(newCol) != 0:
            dictData[d] = []
            dictData[d].append(-newCol)
    else:
        est_sigma = processSingleColSig(col)
        valid_number_of_Pr += len(col)
        dictData[d] = []
        dictData[d].append(-col)
    sum_of_sigma += est_sigma

# sixth experiment
data_set = pd.read_csv('/Users/yizhuolu/Desktop/EE597/HW#1/HW1_Data/wifiExp12.csv', header=None, index_col=False)
for column in range(1, 9):
    d = np.sqrt((float(dictTrans[6][1]) - float(dictRec[column][1]))**2 + (float(dictTrans[6][0]) - float(dictRec[column][0]))**2)
    d = 10 * math.log10(d)
    col = np.array(data_set.iloc[:, column])
    invalid = np.where(col == 500.0)
    est_sigma = 0
    if len(invalid[0]) != 0:
        newCol = np.delete(col, invalid)
        est_sigma = processSingleColSig(newCol)
        valid_number_of_Pr += 0 if len(newCol) == 0 else len(newCol)
        if len(newCol) != 0:
            dictData[d] = []
            dictData[d].append(-newCol)
    else:
        est_sigma = processSingleColSig(col)
        valid_number_of_Pr += len(col)
        dictData[d] = []
        dictData[d].append(-col)
    sum_of_sigma += est_sigma

# seventh experiment
data_set = pd.read_csv('/Users/yizhuolu/Desktop/EE597/HW#1/HW1_Data/wifiExp13.csv', header=None, index_col=False)
for column in range(1, 9):
    d = np.sqrt((float(dictTrans[7][1]) - float(dictRec[column][1]))**2 + (float(dictTrans[7][0]) - float(dictRec[column][0]))**2)
    d = 10 * math.log10(d)
    col = np.array(data_set.iloc[:, column])
    invalid = np.where(col == 500.0)
    est_sigma = 0
    if len(invalid[0]) != 0:
        newCol = np.delete(col, invalid)
        est_sigma = processSingleColSig(newCol)
        valid_number_of_Pr += 0 if len(newCol) == 0 else len(newCol)
        if len(newCol) != 0:
            dictData[d] = []
            dictData[d].append(-newCol)
    else:
        est_sigma = processSingleColSig(col)
        valid_number_of_Pr += len(col)
        dictData[d] = []
        dictData[d].append(-col)
    sum_of_sigma += est_sigma

#eighth experiment
data_set = pd.read_csv('/Users/yizhuolu/Desktop/EE597/HW#1/HW1_Data/wifiExp14.csv', header=None, index_col=False)
for column in range(1, 9):
    d = np.sqrt((float(dictTrans[8][1]) - float(dictRec[column][1]))**2 + (float(dictTrans[8][0]) - float(dictRec[column][0]))**2)
    d = 10 * math.log10(d)
    col = np.array(data_set.iloc[:, column])
    invalid = np.where(col == 500.0)
    est_sigma = 0
    if len(invalid[0]) != 0:
        newCol = np.delete(col, invalid)
        est_sigma = processSingleColSig(newCol)
        valid_number_of_Pr += 0 if len(newCol) == 0 else len(newCol)
        if len(newCol) != 0:
            dictData[d] = []
            dictData[d].append(-newCol)
    else:
        est_sigma = processSingleColSig(col)
        valid_number_of_Pr += len(col)
        dictData[d] = []
        dictData[d].append(-col)
    sum_of_sigma += est_sigma

# nineth experiment
data_set = pd.read_csv('/Users/yizhuolu/Desktop/EE597/HW#1/HW1_Data/wifiExp15.csv', header=None, index_col=False)
for column in range(1, 9):
    d = np.sqrt((float(dictTrans[9][1]) - float(dictRec[column][1]))**2 + (float(dictTrans[9][0]) - float(dictRec[column][0]))**2)
    d = 10 * math.log10(d)
    col = np.array(data_set.iloc[:, column])
    invalid = np.where(col == 500.0)
    est_sigma = 0
    if len(invalid[0]) != 0:
        newCol = np.delete(col, invalid)
        est_sigma = processSingleColSig(newCol)
        valid_number_of_Pr += 0 if len(newCol) == 0 else len(newCol)
        if len(newCol) != 0:
            dictData[d] = []
            dictData[d].append(-newCol)
    else:
        est_sigma = processSingleColSig(col)
        valid_number_of_Pr += len(col)
        dictData[d] = []
        dictData[d].append(-col)
    sum_of_sigma += est_sigma

# tenth experiment
data_set = pd.read_csv('/Users/yizhuolu/Desktop/EE597/HW#1/HW1_Data/wifiExp16.csv', header=None, index_col=False)
for column in range(1, 9):
    d = np.sqrt((float(dictTrans[10][1]) - float(dictRec[column][1]))**2 + (float(dictTrans[10][0]) - float(dictRec[column][0]))**2)
    d = 10 * math.log10(d)
    col = np.array(data_set.iloc[:, column])
    invalid = np.where(col == 500.0)
    est_sigma = 0
    if len(invalid[0]) != 0:
        newCol = np.delete(col, invalid)
        est_sigma = processSingleColSig(newCol)
        valid_number_of_Pr += 0 if len(newCol) == 0 else len(newCol)
        if len(newCol) != 0:
            dictData[d] = []
            dictData[d].append(-newCol)
    else:
        est_sigma = processSingleColSig(col)
        valid_number_of_Pr += len(col)
        dictData[d] = []
        dictData[d].append(-col)
    sum_of_sigma += est_sigma

# eleventh experiment
data_set = pd.read_csv('/Users/yizhuolu/Desktop/EE597/HW#1/HW1_Data/wifiExp17.csv', header=None, index_col=False)
for column in range(1, 9):
    d = np.sqrt((float(dictTrans[11][1]) - float(dictRec[column][1]))**2 + (float(dictTrans[11][0]) - float(dictRec[column][0]))**2)
    d = 10 * math.log10(d)
    col = np.array(data_set.iloc[:, column])
    invalid = np.where(col == 500.0)
    est_sigma = 0
    if len(invalid[0]) != 0:
        newCol = np.delete(col, invalid)
        est_sigma = processSingleColSig(newCol)
        valid_number_of_Pr += 0 if len(newCol) == 0 else len(newCol)
        if len(newCol) != 0:
            dictData[d] = []
            dictData[d].append(-newCol)
    else:
        est_sigma = processSingleColSig(col)
        valid_number_of_Pr += len(col)
        dictData[d] = []
        dictData[d].append(-col)
    sum_of_sigma += est_sigma

# twelfth experiment
data_set = pd.read_csv('/Users/yizhuolu/Desktop/EE597/HW#1/HW1_Data/wifiExp18.csv', header=None, index_col=False)
for column in range(1, 9):
    d = np.sqrt((float(dictTrans[12][1]) - float(dictRec[column][1]))**2 + (float(dictTrans[12][0]) - float(dictRec[column][0]))**2)
    d = 10 * math.log10(d)
    col = np.array(data_set.iloc[:, column])
    invalid = np.where(col == 500.0)
    est_sigma = 0
    if len(invalid[0]) != 0:
        newCol = np.delete(col, invalid)
        est_sigma = processSingleColSig(newCol)
        valid_number_of_Pr += 0 if len(newCol) == 0 else len(newCol)
        if len(newCol) != 0:
            dictData[d] = []
            dictData[d].append(-newCol)
    else:
        est_sigma = processSingleColSig(col)
        valid_number_of_Pr += len(col)
        dictData[d] = []
        dictData[d].append(-col)
    sum_of_sigma += est_sigma

# plot scatter graph
xx = []
yy = []
for key in dictData.keys():
    x = [key] * len(dictData.get(key)[0])
    for each_x in x:
        xx.append(each_x)
    y = dictData.get(key)
    for each_y in y[0]:
        yy.append(each_y)
    plt.scatter(x, y, s=2)
k_b = np.polyfit(xx, yy, 1)

# get the sigma by formula sigma^2 = SUM(Pr - Pr')^2 / n
sum_of_square_diff = 0
for key in dictData.keys():
    Pr_prime = k_b[0] * key + k_b[1]
    for y in dictData.get(key)[0]:
        sum_of_square_diff += (y - Pr_prime)**2
sigma = np.sqrt(sum_of_square_diff / valid_number_of_Pr)
print('Estimated sigma: %f' %sigma)

fig = pylab.gcf()
fig.canvas.set_window_title('Name: Yizhuo Lu  ID: 1383554122  EE597 HW#1')

Eta = -k_b[0]
K = k_b[1] + 27
print('Estimated Eta: %f' %Eta)
print('Estimated K: %f' %K)
xxx = np.arange(0, 18, 0.1)
yyy = xxx * k_b[0] + k_b[1]
plt.plot(xxx, yyy)
plt.title('Received power versus distance')
plt.text(10, -35, 'Estimated sigma: 10.131388\nEstimated Eta: 2.941358\nEstimated K: -2.785128')
plt.xlabel('10lg(d/d0)')
plt.ylabel('Pr-dBm')
plt.show()