import pandas as pd
import matplotlib.pyplot as plt
import math
import operator
#This is my first hack at this--first we remove the test listings that have no
#date first booking which implies that they are NDF. Then we can do further
#analysis.
def returnfinal(thelist): #This takes the predictions and formats them for production.
	finallist=[]
	for item in thelist:
		for country in item[1]:
			finallist.append([item[0],country])
	return finallist
def weighting(row): #This is a non-ML way to pull out relationships in data.
	finaldict={'US':5,'other':4,'FR':3,'IT':2,'GB':1,'ES':0,'DE':0,'CA':0,'NL':0,'AU':0,'PT':0}
	keys=[]
	if row.signup_method=="basic":
		finaldict['US']=finaldict['US']+5
		finaldict['other']=finaldict['other']+4
		finaldict['FR']=finaldict['FR']+3
		finaldict['IT']=finaldict['IT']+2
		finaldict['GB']=finaldict['GB']+1
	elif row.signup_method=="facebook":
		finaldict['US']=finaldict['US']+5
		finaldict['other']=finaldict['other']+4
		finaldict['FR']=finaldict['FR']+3
		finaldict['IT']=finaldict['IT']+2
		finaldict['ES']=finaldict['ES']+3
	else:	
		finaldict['US']=finaldict['US']+5
		finaldict['other']=finaldict['other']+4
		finaldict['FR']=finaldict['FR']+3
		finaldict['IT']=finaldict['IT']+2
		finaldict['GB']=finaldict['GB']+1
	if row.language=="es":
		finaldict['US']=finaldict['US']+5
		finaldict['other']=finaldict['other']+4
		finaldict['ES']=finaldict['ES']+4
		finaldict['FR']=finaldict['FR']+2
		finaldict['IT']=finaldict['IT']+2
	elif row.language=="it":
		finaldict['US']=finaldict['US']+5
		finaldict['IT']=finaldict['IT']+6
		finaldict['FR']=finaldict['FR']+2
		finaldict['other']=finaldict['other']+2
		finaldict['GB']=finaldict['GB']+2
	finaldict = sorted(finaldict.items(), key=operator.itemgetter(1),reverse=True)[0:5]
	for newitem in finaldict:
		keys.append(newitem[0])
	print(keys)
	return keys
test=pd.read_csv("test_users.csv") #Let's import the test data.
firstscreen=test[test['date_first_booking']==test['date_first_booking']] #We're gonna focus on anything that has a booking.
ndfscreen=test[test['date_first_booking']!=test['date_first_booking']]
ndffinal=[[iden,["NDF","US"]] for iden in ndfscreen['id']] #We're assigning NDF and US to any row that doesn't have a booking.
finalscreen=[]
for index, row in firstscreen.iterrows():
	finalscreen.append([row['id'],weighting(row)]) #Let's apply the weighting.
finalscreen=finalscreen+ndffinal
final=returnfinal(finalscreen) #Now we prepare for submission.
frame=pd.DataFrame(final,columns=('id','country'))
print(frame)
frame.to_csv('test.csv') #Now we print to CSV for submission.
