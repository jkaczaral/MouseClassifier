import sys
import os
import numpy as np
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

arg1 = sys.argv[1]


#model for classifying with cross_validation k=10
def Voting_model(x,y):
	clf1 = svm.SVC(kernel='poly')
	clf2 = ExtraTreesClassifier(n_estimators=250, bootstrap=True)
	clf3 = svm.SVC(kernel='linear')
	clf = VotingClassifier(estimators=[('linear', clf3), ('Extra', clf2)], voting='hard')
	clf = clf.fit(x,y)

	return clf

def cross_validation(clf, x, y):
	scores = cross_val_score(clf, x, y, cv=10)

	return scores

def get_data_train(filename):
	with open(filename, 'r') as data:
		names_temp = []
		features_temp = []
		for line in data:
			line = line.rstrip('\n')
			line = line.split(',',1)
			name = line[0]
			data_points = line[1]

			#add names to list
			names_temp.append(name)


				#add features to test to a list of lists
			single_feature = []
			data_points = data_points.split(',')
			for point in data_points:
				single_feature.append(float(point))
			features_temp.append(single_feature)


			#make lists into arrays for testing
		Names_Array = np.array(names_temp)
		Features_array = np.array(features_temp)
	with open('Mouse_Classification_Dataset.txt', 'r') as types_file:
		types = dict()
		for line in types_file:
			line  = line.rstrip('\n')
			line  = line.split('\t')
			mouse_type = line[1]
			name = line[0].replace(' ', ' ')
			types.update({name:mouse_type})

	types_temp = []
	for value in names_temp:
		types_temp.append(types[value])
	types_array = np.array(types_temp)

	return types_array, Features_array, Names_Array

def get_data_test(filename):
	with open(filename, 'r') as data:
		names_temp = []
		features_temp = []
		for line in data:
			line = line.rstrip('\n')
			line = line.split(',',1)
			name = line[0]
			data_points = line[1]

			#add names to list
			names_temp.append(name)

			#add features to test to a list of lists
			single_feature = []
			data_points = data_points.split(',')
			for point in data_points:
				single_feature.append(float(point))
			features_temp.append(single_feature)


			#make lists into arrays for testing
		Names_Array = np.array(names_temp)
		Features_array = np.array(features_temp)

	return Features_array, Names_Array

Train_types, Train_Features, Train_Names = get_data_train('All_features.csv')
Test_Features, Test_Names = get_data_test(arg1)
model = Voting_model(Train_Features, Train_types)

classifier = model.predict(Test_Features)

Cross_Vals = cross_validation(model, Train_Features, Train_types)


with open('Data_Voting_all_features_Extra_poly_linear_hard.txt', 'w') as out_file:
	out_file.write('Cross_Validation Accuracy Scores: ' + repr(Cross_Vals) + '\n') 
	out_file.write('95 percent Confidence Interval :  %0.2f (+/- %0.2f)' % (Cross_Vals.mean(), Cross_Vals.std() * 2) + '\n')
	for index, value in enumerate(classifier):
		out_file.write(Test_Names[index] + '\t' + value + '\n')
print 'ALL DONE!'