from PyQt5.QtWidgets import QMainWindow,QDialog,QFileDialog
from PyQt5.QtCore import pyqtSignal

from Classifier_ui import Ui_MainWindow

from threading import Thread

import pandas as pd


from EmotionClassifier import eAnalyse
from ScoreClassifier import sAnalyse

class Window(QMainWindow):

    ui = Ui_MainWindow()#transfer Classifier_ui.py(the GUI)
    __signalComplet = pyqtSignal(int,int)#(dataEmotion, dataScore)

    def __init__(self):#Methods inherited from the QMainWindow class
        super().__init__()
        self.ui.setupUi(self)



        self.ui.btnAnalyse.clicked.connect(self.analyse)#Analyze method is triggered when Analysis button is pressed


        self.__signalComplet.connect(self.setResult)#When the signalComplet in the getResult method is received, the setResult method is triggered

        #self.setWindowTitle("55")


    def analyse(self):#Pass the input parameters to getResult




        dataEmotion={
            "tweet_id": [self.ui.txtTweetId.text()],
            "sentiment":[self.ui.txtSentiment.text()],
            "author":[self.ui.txtAuthor.text()],
            "content":[self.ui.txtEmotionContent.toPlainText()]
        }


        dataScore={
            "lable": [self.ui.txtLable.text()],
            "content": [self.ui.txtScoreContent.toPlainText()]
        }




        t=Thread(target=self.getResult,args=(dataEmotion,dataScore))#Create a new child thread to handle getResult. If you use the main thread to process data, the interface will not respond.
        t.start()
        self.setEnabled(False)#Lock the interface while processing data.

    def setResult(self,eResult,sResult):#Used to output data

        # self.ui.txtEmotion.setPlainText(emotion)
        # self.ui.txtScore.setPlainText( score)

        eArr=['Fierce','Sadness','Happiness','Reject']

        self.ui.txtEResult.setText(eArr[eResult])


        sArr=['Reject','','Trash(0-2)','','Mediocre(2-4)','','Indifferent(4-6)','','Good(6-8)','','Masterwork(8-10)']


        self.ui.txtSResult.setText(sArr[sResult])

        self.setEnabled(True)#Unlock the interface

    def getResult(self,dataEmotion,dataScore):#Used to process results


        columns = ['tweet_id', 'sentiment', 'author', 'content']#Turn data into DataFrame
        eData = pd.DataFrame(dataEmotion,columns=columns)



        #dataframe.to_csv("emotion_data.csv", index=False, sep=',',columns=columns)

        columns = ['content', 'label']
        eResult = eAnalyse(eData)#eAnalyse is EmotionalClassifier.

        sData = pd.DataFrame(dataScore,columns=columns)
        #dataframe.to_csv("score_data.csv", index=False, sep=',')

        sResult=sAnalyse(sData)#sAnalyse is ScoreClassifier.



        self.__signalComplet.emit(eResult,sResult)#trigger signalComplet
