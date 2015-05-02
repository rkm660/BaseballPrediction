Readme

-The function to get pre-processed data in a nested list format that takes in a CSV fileName is preProcessedData()
	EX. data=preProcessedData(‘btrain.csv’)

-Once you have pre-processed data, run createTree() with the return value from the previous function to create a ID3 decision tree model where the return value is a nested dictionary.
	EX model = createTree(data)

-Run pruning() with your tree model to return a pruned version of your tree.
	EX pruneModel= pruning(model)

-Run printNormalForm(printBooleanForm()) with the model and and empty list as inputs to print out a disjunctive normal format of the decision tree model.
	EX printNormalForm(printBooleanForm(model,[]))

-Run testAccuracy() with a decision tree and validation set to test the accuracy of a model given a validation set.
	EX testAccuracy(model,’bvalidate.csv’)


NOTE: Unfortunately, our validation takes several hours to run, therefore we were unable to label the target value of the test set.
