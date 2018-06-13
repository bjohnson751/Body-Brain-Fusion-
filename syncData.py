import pandas as pd
import numpy as np
import csv
import math
import nltk
import scipy as sp
from scipy import signal
from scipy.signal import butter, lfilter, freqz, firwin,kaiserord
import matplotlib.pyplot as plt

#.04
#500

# use the normalized cut off for fs
def butter_lowpass_filter(data, cutoffN, order=6):
    b, a = butter(order, cutoffN, btype='low', analog=False)

    y = lfilter(b, a, data)

    '''taps = firwin(order+1, cutoffN)

    # Use lfilter to filter x with the FIR filter.
    y = lfilter(taps, 1.0, data)'''
    return y


def firfilt(data):
    nfreq = 0.04
    taps =  6
    a = 1
    b = signal.firwin(taps, cutoff=nfreq)
    print(b)
    firstpass = signal.lfilter(b, a, data)
    ## second pass to compensate phase delay
    secondpass = signal.lfilter(b, a, firstpass[::-1])[::-1]
    return secondpass

# name the columns of the time data frame
'''columns = ["min","sec","ms","ns"]

# read from time.csv and name the columns
time_datafNIRS = pd.read_csv("time.csv", names=columns)

# transpose the data for calculations
time_datafNIRS.T

# get the size of the data frame
dim = time_datafNIRS.shape

# actual size of the data
size = dim[0]

# calculated time
timeList = []

for i in range(size):
        a = time_datafNIRS.iloc[i,0]
        min = a * (3600)
        b = time_datafNIRS.iloc[i,1]
        sec = b * (60)
        c = time_datafNIRS.iloc[i,2]
        d = time_datafNIRS.iloc[i,3]
        nano = d/1000

        calc = min+sec+c+nano

        timeList +=[calc]

# name the column for new dataframe and create it from a list
column2 = ["time"]
calc_time = pd.DataFrame(np.array(timeList), columns = column2)

#print(df2)

# creating fduration

#perform individual calculations percolumn
val1 = (time_datafNIRS.iloc[size-1,0] - time_datafNIRS.iloc[0,0]) *(3600)

val2 = (time_datafNIRS.iloc[size-1,1] - time_datafNIRS.iloc[0,1]) *(60)

val3 =  (time_datafNIRS.iloc[size-1,2] - time_datafNIRS.iloc[0,2])

val4 =  (time_datafNIRS.iloc[size-1,3] - time_datafNIRS.iloc[0,3]) / 1000

totalVal = val1 + val2 + val3 + val4

# total time it took for study to complete
fDuration = int(math.floor(totalVal))

# this uses dataframe 2 for the calculations
# getting a zeroed out dataframe
zeroing = []
for i in range(size):
    zero = calc_time.iloc[i,0]-calc_time.iloc[0,0]
    #print(zero)
    zeroing += [zero]

#create zeroed data frame
column3 = ["seconds"]
zeroed_data = pd.DataFrame(np.array(zeroing), columns = column3)

# every second of the data is calculated
fxx = []
for k in range(size):
    fx = []
    for j in range(fDuration):
        f = abs(j - zeroed_data.iloc[k,0])
        fx += [f]
    fxx += [fx]

# turn multidimentional list into dataframe
x = np.array(fxx)
study_data = pd.DataFrame(np.array(fxx))

#clear row of unnessecary information
study_data.drop(study_data.index[0], inplace=True)

# gather minimums of each columns for upsampling
mins = []
for i in range(fDuration):
    if i == 0:
        mins += [study_data[i].idxmin()]
    else:
        mins += [study_data[i].idxmin() + 1]

oxy = np.genfromtxt('oxy.csv', delimiter=',')


oxy_df = pd.DataFrame(np.array(oxy))

oxysh = oxy_df.shape

oxyDuration = oxysh[1]

upsampy = pd.DataFrame()
for j in range(fDuration-1):

    x = []
    for i in range(oxyDuration):
        interDF = oxy_df.loc[mins[j]-1:mins[j+1]-1, :]
        arr = np.array(interDF.loc[:,i].values)
        z = signal.resample(arr, 50)
        x += [z]


    sample = pd.DataFrame(np.array(x))

    upsampy = upsampy.append(sample.T, ignore_index = True)



#upsampy.to_csv("upOxy.csv", sep=',')'''


''' MOCAP CODE '''
# name the columns of the time data frame
columns = ["min","sec","ms","ns"]

# read from timeM.csv and name the columns
time_dataMoCap = pd.read_csv("timeM.csv", names=columns)

# transpose the data for calculations
time_dataMoCap.T

# get the size of the data frame
dimM = time_dataMoCap.shape

# actual size of the data
sizeM = dimM[0]

# calculated time
timeListM = []

for i in range(sizeM):
        a = time_dataMoCap.iloc[i,0]
        minM = a * (3600)
        b = time_dataMoCap.iloc[i,1]
        secM = b * (60)
        cM = time_dataMoCap.iloc[i,2]
        d = time_dataMoCap.iloc[i,3]
        nanoM = d/1000

        calc = minM+secM+cM+nanoM

        timeListM +=[calc]

column2M = ["time"]
calc_timeM = pd.DataFrame(np.array(timeListM), columns = column2M)

#perform individual calculations percolumn
val1M = (time_dataMoCap.iloc[sizeM-1,0] - time_dataMoCap.iloc[0,0]) *(3600)

val2M = (time_dataMoCap.iloc[sizeM-1,1] - time_dataMoCap.iloc[0,1]) *(60)

val3M =  (time_dataMoCap.iloc[sizeM-1,2] - time_dataMoCap.iloc[0,2])

val4M =  (time_dataMoCap.iloc[sizeM-1,3] - time_dataMoCap.iloc[0,3]) / 1000

totalValM = val1M + val2M + val3M + val4M

# total time of the study
mDuration = int(math.floor(totalValM))

# this uses dataframe 2/calc_timeM for the calculations
# getting a zeroed out dataframe
zeroingM = []
for i in range(sizeM):
    zero = calc_timeM.iloc[i,0]-calc_timeM.iloc[0,0]
    zeroingM += [zero]

# creating new dataframe with zeroed data
column3M = ["seconds"]
zeroed_dataM = pd.DataFrame(np.array(zeroingM), columns = column3M)

# creating dataframe for every second of the study
fxxM = []
for k in range(sizeM):
    fx = []
    for j in range(mDuration):
        f = abs(j - zeroed_dataM.iloc[k,0])
        fx += [f]
    fxxM += [fx]

xM = np.array(fxxM)
study_dataM = pd.DataFrame(np.array(fxxM))

# getting rid of unnessecary information
study_dataM.drop(study_dataM.index[0], inplace=True)


#finding the minimum of each column
minsM = []
for i in range(mDuration):
    if i == 0:
        minsM += [study_dataM[i].idxmin()]
    else:
        minsM += [study_dataM[i].idxmin()+1]

# read in the data as a np array
mocap = np.genfromtxt('mocap.csv', delimiter=',')

# convert data to data frame
mocap_df = pd.DataFrame(np.array(mocap))

# get the shape (dimensions)
mosh = mocap_df.shape

# get rid of the uncessary column
mocap_df.drop(mocap_df.columns[mosh[1]-1],axis=1, inplace=True)

print(mocap_df)

# adjust dimensions
moshDuration = mosh[1]-1
'''
filteredDF = pd.DataFrame()
for i in range(moshDuration):
    interSD = mocap_df.loc[:,i]
    arr = np.array(interSD.values)
    k = []
    for j in range(len(arr)):
        if arr[j] <= 0.1:
            k +=[arr[j]]
    sample = pd.DataFrame(np.array(k))
    #y = butter_lowpass_filter(arr, 0.1)
    filteredDF = filteredDF.append(sample.T, ignore_index = True)

'''
filteredDF = pd.DataFrame()
for i in range(moshDuration):
    interSD = mocap_df.loc[:,i]
    arr = np.array(interSD.values)
    y = butter_lowpass_filter(arr, 0.04)
    #y = firfilt(arr)
    sample = pd.DataFrame(np.array(y))
    filteredDF = filteredDF.append(sample.T, ignore_index = True)

#each column is a seperate channel - in the homeoglobin

print(filteredDF)

filtDF = filteredDF.T

filtDF.to_csv("upMocapFilB.csv", sep=',')

# apply filter on the data
print(filtDF)
# calculate upsample
upsampyM = pd.DataFrame()
for j in range(mDuration-1):
    x = []
    # take the chunk of information
    interDFM = filtDF.loc[minsM[j]-1:minsM[j+1]-1, :]
    # for each column
    for i in range(moshDuration):
        # conver to np array
        arr = np.array(interDFM.loc[:,i].values)
        # upsample
        z = signal.resample(arr, 500)
        #80
        #z = signal.resample_poly(arr, 50, 75)
        # store in array of arrays
        x += [z]

    # convert array of arrays to
    sample = pd.DataFrame(np.array(x))

    upsampyM = upsampyM.append(sample.T, ignore_index = True)

print(upsampyM)


upsampyM.to_csv("upMocapFilUpB.csv", sep=',')
