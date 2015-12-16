import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
def returnfinal(thelist): #A method to parse the finallist for submission.
	finallist=[]
	for item in thelist:
		for country in item[1]:	
			finallist.append([item[0],country])
	return finallist
def dateParseMonth(date): #Thanks HulkBulk
	try:
		spl=date.split('-')
		return int(spl[1])
	except:
		return 0
def numberToClass(listofprobs): #Converting a dict number to a country code
	listofprobs[0]=thecountries[listofprobs[0]]
	return listofprobs
def fixAge(age): #There are obviously some outliers in the data-set, let's fix that.
	try:
		if (age<16) or (age>95):
			return np.nan
	except:
		return np.nan
	return age
test=pd.read_csv("test_users.csv")
train=pd.read_csv("train_users.csv")
firsttest=test[test['date_first_booking']==test['date_first_booking']]
firsttrain=train[train['date_first_booking']==train['date_first_booking']]
ndftest=test[test['date_first_booking']!=test['date_first_booking']]

firsttrain['month_c']=firsttrain['date_account_created'].apply(dateParseMonth)
firsttest['month_c']=firsttest['date_account_created'].apply(dateParseMonth)

firsttest=firsttest.drop(['date_account_created'],axis=1)
firsttrain=firsttrain.drop(['date_account_created'],axis=1)
#House cleaning
firsttrain['month_f']=firsttrain['date_first_booking'].apply(dateParseMonth)
firsttest['month_f']=firsttest['date_first_booking'].apply(dateParseMonth)

firsttrain.apply(fixAge)
firsttest.apply(fixAge)

firsttest=firsttest.drop(['date_first_booking'],axis=1)
firsttrain=firsttrain.drop(['date_first_booking'],axis=1)

firsttest=firsttest.drop(['timestamp_first_active'],axis=1) #I doubt there's any relevant correlation between the timestamp of first activity and final destination. I've opted to disclude this specific information in the learning model.
firsttrain=firsttrain.drop(['timestamp_first_active'],axis=1)
#Ceruleus developed a servicable method for preprocessing the data. I've adapted it here but the concept is theirs. You can find their script here: https://www.kaggle.com/ceruleus/airbnb-recruiting-new-user-bookings/airbnb-notebook/notebook
x_train=firsttrain.drop(["country_destination","id"],axis=1)
y_train=firsttrain["country_destination"]
x_test=firsttest.copy()
x_test=x_test.drop(['id'],axis=1)
for f in x_train.columns:
	if x_train[f].dtype=='object':
		lbl=preprocessing.LabelEncoder()
		if f not in x_test.columns:
			lbl.fit(np.unique(list(x_train[f].values)))
			x_train[f]=lbl.transform(list(x_train[f].values))
		else:
			lbl.fit(np.unique(list(x_train[f].values)+list(x_test[f].values)))
			x_train[f]=lbl.transform(list(x_train[f].values))
			x_test[f]=lbl.transform(list(x_test[f].values))
imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
x_train_nonan=imp.fit_transform(x_train)
x_test_nonan=imp.fit_transform(x_test)
model=AdaBoostClassifier(DecisionTreeClassifier(class_weight={'US':.7033,'other':.1117,'FR':.0574,'IT':.0313,'GB':.0262,'ES':.02,'CA':.01,'DE':.008,'NL':.006,'AU':.005,'PT':.001}),n_estimators=300)
#I've found the AdaBoostClassifier paired with decision trees prevents overfitting and has the best scoring.
model.fit(x_train_nonan,y_train)
print(model.score(x_train_nonan,y_train))
y_probs=model.predict_proba(x_test_nonan)
thecountries=model.classes_
finalappend=[]
for t,trying in enumerate(y_probs):
	new=sorted([[coun,prob] for coun,prob in enumerate(trying)],key=lambda x: x[1],reverse=True)
	del new[5:]
	new=[numberToClass(g)[0] for g in new]
	#print(new)
	finalappend.append(new) #Let's pick out the 5 highest and add them in
#print(firsttest["id"])
#print(y_test)
ndffinal=[[iden,["NDF","US"]] for iden in ndftest['id']]
out=[[testing,finalappend[j]] for j,testing in enumerate(firsttest["id"])]
#print(out)
finalscreen=out+ndffinal
final=returnfinal(finalscreen)
frame=pd.DataFrame(final,columns=('id','country'))
frame.to_csv('test.csv') #and submit
