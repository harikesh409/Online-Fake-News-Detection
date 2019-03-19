# Fake News Detection

Fake News Detection in Python

In this project, we have used various natural language processing techniques and machine learning algorithms to classify fake news articles using sci-kit libraries from python. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

1. Python 3.6 
   - This setup requires that your machine has python 3.6 installed on it. you can refer to this url https://www.python.org/downloads/ to download python. Once you have python downloaded and installed, you will need to setup PATH variables (if you want to run python program directly, detail instructions are below in *how to run software section*). To do that check this: https://www.pythoncentral.io/add-python-to-path-python-is-not-recognized-as-an-internal-or-external-command/.  
   - Setting up PATH variable is optional as you can also run program without it and more instructon are given below on this topic. 
2. Second and easier option is to download anaconda and use its anaconda prompt to run the commands. To install anaconda check this url https://www.anaconda.com/download/
3. You also need to install the dependent packages. To do that execute the below command either in command prompt or anaconda prompt depending on your environment.
```
pip install -r requirements.txt
```
#### Dataset used
The data source used for this project is LIAR dataset which contains 3 files with .tsv format for test, train and validation. Below is some description about the data files used for this project.

LIAR: A BENCHMARK DATASET FOR FAKE NEWS DETECTION

William Yang Wang, "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection, to appear in Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017), short paper, Vancouver, BC, Canada, July 30-August 4, ACL.

the original dataset contained 13 variables/columns for train, test and validation sets as follows:

* Column 1: the ID of the statement ([ID].json).
* Column 2: the label. (Label class contains: True, Mostly-true, Half-true, Barely-true, FALSE, Pants-fire)
* Column 3: the statement.
* Column 4: the subject(s).
* Column 5: the speaker.
* Column 6: the speaker's job title.
* Column 7: the state info.
* Column 8: the party affiliation.
* Column 9-13: the total credit history count, including the current statement.
* 9: barely true counts.
* 10: false counts.
* 11: half true counts.
* 12: mostly true counts.
* 13: pants on fire counts.
* Column 14: the context (venue / location of the speech or statement).

To make things simple we have chosen only 2 variables from this original dataset for this classification. The other variables can be added later to add some more complexity and enhance the features.

Below are the colomns used to create 3 datasets that have been in used in this project
* Column 1: Statement (News headline or text).
* Column 2: Label (Label class contains: True, False)
 
You will see that newly created dataset has only 2 classes as compared to 6 from original classes. Below is method used for reducing the number of classes.

* Original 	--	New
* True	   	--	True
* Mostly-true	-- 	True
* Half-true	-- 	True
* Barely-true	-- 	False
* False		-- 	False
* Pants-fire	-- 	False

The dataset used for this project were in csv format named train.csv, test.csv and valid.csv and can be found in repo. The original datasets are in "liar" folder in tsv format.

### File descriptions
There are 2 folders Notebooks and Code
#### Notebooks
This folder contains the jupyter notebook files. It contains 3 notebooks
1 DataPreProcessing
2 FeatureSelection
3 Classifier

-To run these notebooks you need to install jupyter notebook inside anaconda.
-After installing jupyter notebook use the below command to run the notebook.
```
cd <your cloned project folder path goes here>/Notebooks
jupyter notebook
```
- After successfully running the command the jupyter notebook will run in your default browser. Open the corresponding notebook that you want to visit.

#### Code
This folder contains the source code of the project. If you want to run the code directly instead of web application follow the following steps.

 - Open the command prompt and change the directory to project folder as mentioned in above by running below command
    ```
    cd <your cloned project folder path goes here>/Code
    ```
    - run below command
    ```
    python <your cloned project folder path goes here>/Code/prediction.py
    ```
    - After hitting the enter, program will ask for an input which will be a piece of information or a news headline that you 	    	   want to verify. Once you paste or type news headline, then press enter.

    - Once you hit the enter, program will take user input (news headline) and will be used by model to classify in one of  categories of "True" and "False". Along with classifying the news headline, model will also provide a probability of truth associated with it.

#### Files

##### Data Preprocessing
This file contains all the pre processing functions needed to process all input documents and texts. First we read the train, test and validation data files then performed some pre processing like tokenizing, stemming etc. There are some exploratory data analysis is performed like response variable distribution and data quality checks like null or missing values etc.

##### Feature Selection
In this file we have performed feature extraction and selection methods from sci-kit learn python libraries. For feature selection, we have used methods like simple bag-of-words and n-grams and then term frequency like tf-tdf weighting. we have also used word2vec and POS tagging to extract the features, though POS tagging and word2vec has not been used at this point in the project.

##### Classifier
Here we have build all the classifiers for predicting the fake news detection. The extracted features are fed into different classfiers. We have used Naive-bayes, Logistic Regression, Linear SVM, Stochastic gradient decent and Random forest classifiers from sklearn. Each of the extracted featues were used in all of the classifiers. Once fitting the model, we compared the f1 score and checked the confusion matrix. After fitting all the classifiers, 2 best peforming models were selected as candidate models for fake news classification. We have performed parameter tuning by implementing GridSearchCV methos on these candidate models and chosen best performing paramters for these classifier. Finally selected model was used for fake news detection with the probability of truth. In Addition to this, We have also extracted the top 50 features from our term-frequency tfidf vectorizer to see what words are most and important in each of the classes. We have also used Precision-Recall and learning curves to see how training and test set performs when we increase the amount of data in our classifiers.

##### Prediction
Our finally selected and best performnig classifer was ```Logistic Regression``` which was then saved on disk with name ```final_model.sav```. Once you close this repository, this model will be copied to user's machine and will be used by prediction.py file to classify the fake news. It takes an news article as input from user then model is used for final classification output that is shown to user along with probability of truth.

##### Server
This is the flask server used as backend for the web application.

### Installing and steps to run the software

A step by step series of examples that tell you have to get a development env running

1.The first step would be to clone this repo in a folder in your local machine. To do that you need to run following command in command prompt or in git bash

```
git clone https://github.com/harikesh409/Online-Fake-News-Detection.git
```
2.This will copy all the data source file, program files and model into your machine.

3.
- If you have chosen to install anaconda then follow below instructions
	- After all the files are saved in a folder in your machine.If you chosen to install anaconda from the steps given in ```Prerequisites``` sections then open the anaconda prompt, change the directory to the folder where this project is saved in     your machine and type below command and press enter.
	```
	cd <your cloned project folder path goes here>
	```
	- Once your are inside the directory type the below command to change the directory to web app
	```
	cd web app
	```
	- Once you are inside the web app directory call the ```server.py``` file; To do this, run the below command in anaconda prompt.
	```
	python server.py
	```
	- After hitting enter the server will run in the default port 5000
	- Now open ```index.html``` file in the browser and enter the input(news headline) and click the check button. This input will be used by model to provide a probability of truth associated with it.

4.If you have chosen to install python (and did not set up PATH variable for it) then follow below instructions:
- After you clone the project in a folder in your machine. Open command prompt and change the directory to project directory by running below command.
    ```
    cd <your cloned project folder path goes here>
    ```
    - Locate ```python.exe``` in your machine. you can search this in window explorer search bar. 
    - Once you locate the ```python.exe``` path, you need to write whole path of it and then entire path of project folder with ```prediction.py``` at the end. For example if your ```python.exe``` is located at ```c:/Python36/python.exe``` and project folder is at ```c:/users/user_name/desktop/Online-Fake-News-Detection/```, then your command to run program will be as below:
    ```
    c:/Python36/python.exe C:/users/user_name/desktop/fake_news_detection/web app/server.py
    ```
    - After hitting enter the server will run in the default port 5000
	- Now open ```index.html``` file in the browser and enter the input(news headline) and click the check button. This input will be used by model to provide a probability of truth associated with it.

5. If you have chosen to install python (and already setup PATH variable for ```python.exe```) then follow instructions:
    - Open the command prompt and change the directory to project folder as mentioned in above by running below command
    ```
    cd <your cloned project folder path goes here>
    ```
    - Once your are inside the directory type the below command to change the directory to web app
	```
	cd web app
	```
    - run below command
    ```
    python <your cloned project folder path goes here>/web app/server.py
    ```
    - After hitting enter the server will run in the default port 5000
	- Now open ```index.html``` file in the browser and enter the input(news headline) and click the check button. This input will be used by model to provide a probability of truth associated with it.