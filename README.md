# EAFT
Emotion Analysier for Text (EAFT)

Authors: Sheng Zhao(Leader), Yuanzhe Liu, Shuwei Zheng

## 1. Introduction: 
A movie comments classifier and predictor where criticisms and applause are clearly presented. This classifier has unique features, which can be used to predict the specific score of particular comments from 0 to 10 (which are 'Reject','','Trash(0-2)','','Mediocre(2-4)','','Indifferent(4-6)','','Good(6-8)','','Masterwork(8-10)') and to identify people's feelings and emotions out of 4 (which are 'Fierce','Sadness','Happiness','Reject'). 

The accuracy is above 70% according to the tfscore.

![Result70PercentAccuracy](https://github.com/JasonJarvan/EAFT/blob/master/Result%20Example/Result72PercentAccuracy.png)

We crawled 50000 comments from IMDB, and we manually cleaned and relabeled the. With high volume samples and accurate labels, our project reachs a high accuracy.

You can use the GUI to classify the emotion and score of a comment to movie.

![GUI]([https://github.com/JasonJarvan/EAFT/blob/master/Result%20Example/Result72PercentAccuracy.png](https://github.com/JasonJarvan/EAFT/blob/master/Result%20Example/GUI.png))

## 2. Environment and Commands: 	
a.Download Anaconda for python 3.7 which contains jupyter notebook pre-installed. Open jupyter notebook, create a new python3 note, and make sure you have installed nltk, or input the following code to download nltk packages:
```
	import nltk
	nltk.download()
```

	Then the nltk's downloader will run. Download all the packages to the default filepath.
 
b. Unzip EAFT\Demo\Score Classifier\aclImdb.zip into 'EAFT\Demo\Score Classifier\aclImdb' folder.

c. Then you can put our program into the directory of your jupyter notebook's workspace (usually be your C:\Users\Your User) and see them in jupyter notebook. Run them with jupyter notebook.

Hint: Anaconda includes all the packages we need, so install it and set it as your main python's interpreter is important.

## 3. How to run EAFT: 

a. How to run the core classifier (model learner): 

Run Score Classifier.ipynb and Emotion Classifier.ipynb in jupyter notebook. By doing this, you will see the tf scores of different algorithms.

b. How to run the GUI: 

As you have anaconda correctly installed, simply open the CMD in the GUI's directory, and input the command: "python main.py", then you will see the GUI. You can use it to calculate how is the text being satisfied with the movie, which is the score from 0 to 10 (which are 'Reject','','Trash(0-2)','','Mediocre(2-4)','','Indifferent(4-6)','','Good(6-8)','','Masterwork(8-10)') , and the emotion out of 4 (which are 'Fierce','Sadness','Happiness','Reject').

## 4. Directories
├──Readme.txt  
├──Demo: The classifiers runned by jupyter notebook. These show the accuracy of our classifier.  
│   └──Emotion Classifier   
│       ├──Emotional Classifier.ipynb inputs text_emotion-ori.csv. This dataset isn't reclassified manually, so the accuracy is not high. The program's output is the accuracy and text2.csv. We use such test results to manually 		reclassify the emotions.   
│       ├──Emotional Classifier-Filter.ipynb inputs text_emotion-ori.csv and outputs dataafterfilter.csv.  
│       ├──Emotional Classifier-Divider.ipynb inputs dataafterfilternew.csv and outputs the accuracy.  
│       ├──text2.csv: where the predicted emotions of the first 3000 comments are.  
│       ├──text_emotion-ori.csv: where 40000 dataset is. This is original dataset that isn't manually reclassified.  
│       ├──dataafterfilter.csv: which is used for manually reclassifying and as the input of Emotional Classifier-Divider.  
│       └──dataafterfilternew.csv: which is manually reclassified.  
│   └──Score Classifier  
│       ├──score.ipynb:Inputs the acllmdb and output the accuracy.  
│       ├──Score Classifier Filter: Generate the scoreallafterfilter.csv, which can shorten our debuging time.  
│       ├──acllmdb: where 20000 dataset is.  
│       └──scoreallafterfilter.csv: includes the score and content of the 20000 dataset.  
├──GUI: The classifiers runned by python. This is our GUI which allows you to input your custom text and get output.  
│   ├──debug.py： which is used for debugging.  
│   ├──Classifier.ui: the GUI generated by QTDesigner.  
│   ├──Classifier_ui.py: Translated from Classifier.ui to python language by QTDesigner.  
│   ├──GUI.py: The functional code of GUI where triggers are.  
│   ├──main.py: The main method of GUI.  
│   ├──EmotionClassifier.py and ScoreClassifier.py: The classifiers.  
│   └──dataafterfilternew.csv and scoreallafterfilter.csv: will be used by classifiers.  
  
Hint: Each childfolder contains a Readme file to explain what's the use of the files.
