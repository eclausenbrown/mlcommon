from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelBinarizer
import numpy as np

class TrainClassifier(object):

	def __init__(self, X, y, clf):
		self.X = X
		self.y = y
		self.clf = clf

	def holdout_split(self, test_size=0.2, random_state=None):
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
			self.X, self.y, test_size=test_size, random_state=random_state)

	def cross_val(self, cv=3, scoring="accuracy", is_shuffle_split=False, **kwargs):
		if is_shuffle_split:
			test_size = kwargs.get("test_size", 0.3)
			random_state = kwargs.get("random_state", 0)
			cv = ShuffleSplit(n_splits=cv, test_size=test_size, random_state=random_state)
		scores = cross_val_score(self.clf, self.X_train, self.y_train, cv=cv, 
			scoring=scoring)
		print("cross validation scores = ", scores)

	def fit(self):
		self.clf.fit(self.X_train, self.y_train)

	def test_set_evaluation(self, average="weighted"):
		# run after running the fit() method
		y_pred = self.clf.predict(self.X_test)
		y_score = self.clf.predict_proba(self.X_test)
		is_multiclass = len(np.unique(self.y)) > 2
		if is_multiclass:
			lb = LabelBinarizer()
			lb.fit(self.y)
			y_test_mat = lb.transform(self.y_test)
			#y_pred_mat = lb.transform(y_pred)
			print(np.shape(y_test_mat), np.shape(y_score))
			auc = roc_auc_score(y_test_mat, y_score, average=average)
			print("Multiclass using a '{}' averaging strategy: ROC AUC score = {}".format(average, str(auc)))
		else:
			auc = roc_auc_score(y_pred, self.y_test)
			print("Binary ROC AUC score = {}".format(str(auc)))
		print("accuracy = {}".format(str(accuracy_score(self.y_test, y_pred))))
		print("classification report:")
		print(classification_report(self.y_test, y_pred))


def main():
	from sklearn.datasets import fetch_20newsgroups 
	from sklearn.pipeline import Pipeline
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.linear_model import LogisticRegression
	from sklearn.preprocessing import Normalizer

	# #############################################################################
	# Load some categories from the training set

	remove = ('headers', 'footers', 'quotes')
	categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
	]

	print("Loading 20 newsgroups dataset for categories:")
	print(categories)

	
	data = fetch_20newsgroups(subset = 'train', categories = categories,
	                        shuffle = True, random_state = 42,
	                        remove = remove)

	#data = fetch_20newsgroups(subset='train', categories=categories)
	print("%d documents" % len(data.filenames))
	print("%d categories" % len(data.target_names))
	print()

	# #############################################################################

	vect = CountVectorizer(max_features=30000, ngram_range=(1,2))
	lf = LogisticRegression(C=1.)
	#clf = Pipeline([("vect", vect), ("norm", Normalizer()), ("clf", lf)])
	clf = Pipeline([("vect", vect), ("clf", lf)])
	myclf = TrainClassifier(X=data.data, y=data.target, clf=clf)
	myclf.holdout_split(random_state=42)
	myclf.cross_val()
	myclf.fit()
	myclf.test_set_evaluation()

if __name__ == '__main__':
	main()







		