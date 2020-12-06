Emotional Classifier.ipynb inputs text_emotion-ori.csv. This dataset isn't reclassified manually, so the accuracy is not high. The program's output is the accuracy and text2.csv. We use such test results to manually reclassify the emotions. 
Emotional Classifier-Filter.ipynb inputs text_emotion-ori.csv and outputs dataafterfilter.csv.
Emotional Classifier-Divider.ipynb inputs dataafterfilternew.csv and outputs the accuracy.
text2.csv: where the predicted emotions of the first 3000 comments are.
text_emotion-ori.csv: where 40000 dataset is. This is original dataset that isn't manually reclassified.
dataafterfilter.csv: which is used for manually reclassifying and as the input of Emotional Classifier-Divider.
dataafterfilternew.csv: which is manually reclassified.