import os

if 'kaggle' in os.getcwd():
    from training.training_template import DS_Classifier
else:
    from training_template import DS_Classifier

from configs.dsCls_agrs import TestArgs


ds_agrs = TestArgs()
args = ds_agrs.parser()

ds_classifier = DS_Classifier(opts=args)
ds_classifier.test()









