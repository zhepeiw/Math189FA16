import os
import pandas as pd
import numpy as np
import math

companyList = [
	['B365H','B365D','B365A'],
	['BWH','BWD','BWA'],
	['IWH','IWD','IWA'],
	['LBH','LBD','LBA'],
	['PSH','PSD','PSA'],
	['WHH','WHD','WHA'],
	['VCH','VCD','VCA'],
	['PSCH','PSCD','PSCA']
]

dirname = "data"
years = [os.path.join(dirname, o) for o in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, o))][5:6]

def generateScoreData(dataSet):
	FTHomeScore = dataSet[['FTHG']].as_matrix()
	FTAwayScore = dataSet[['FTAG']].as_matrix()

	FTScoreDiff = [FTHomeScore.item(i) - FTAwayScore.item(i) for i in range(len(FTHomeScore))]
	FTResult = []
	for s in FTScoreDiff:
		if s > 0:
			FTResult.append(1)
		elif s < 0:
			FTResult.append(-1)
		else:
			FTResult.append(0)

	return FTScoreDiff

def generateBetData(dataSet, betH, betD, betA):
	BH = dataSet[[betH]].as_matrix()
	BD = dataSet[[betD]].as_matrix()
	BA = dataSet[[betA]].as_matrix()
	return BH, BD, BA

def generateXData(dataSet):
	hSum = None
	dSum = None
	aSum = None

	count_valid = 0
	for company in companyList:
		if company[0] in dataSet and company[1] in dataSet and company[2] in dataSet:
			count_valid += 1
			currH, currD, currA = generateBetData(dataSet, company[0], company[1], company[2])
			if hSum is None:
				hSum = np.matrix([0.] * len(currH)).reshape(-1,1)
				dSum = np.matrix([0.] * len(currD)).reshape(-1,1)
				aSum = np.matrix([0.] * len(currA)).reshape(-1,1)
			# print hSum
			hSum += currH
			dSum += currD
			aSum += currA
	
	return hSum / count_valid, dSum / count_valid, aSum / count_valid

# Load and parse data
def generate():
	XData = [[], [], []]
	YData = []

	for year in years:	
		filenames = sorted([os.path.join(year, fn) for fn in os.listdir(year) if fn.endswith(".csv")])
		for filename in filenames:
			print "Read dataset: {}".format(filename)

			df_train = None
			# try:
			df_train = pd.read_csv(filename, sep=',', parse_dates=True, dayfirst=True, index_col=0, error_bad_lines=False)
			df_train.describe()
			# except ValueError:
			# 	print "Error occurred while parsing! Skip this"

			if df_train is not None:
				y = generateScoreData(df_train)
				h, d, a = generateXData(df_train)

				for i in range(len(h)):
					if math.isnan(h.item(i)) or math.isnan(d.item(i)) or math.isnan(a.item(i)) or math.isnan(y[i]):
						continue
					XData[0].append(h.item(i))
					XData[1].append(d.item(i))
					XData[2].append(a.item(i))				
					YData.append(y[i])
	return XData, YData
# print len(YData)
