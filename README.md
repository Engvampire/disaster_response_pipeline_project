# Disaster Response Pipeline Project

## Project Description
In this project, we will build a model to classify messages that are sent during disasters. There are 36 pre-defined categories.
By classifying these messages, we can allow these messages to be sent to the appropriate disaster relief agency.

This project will involve the building of ETL and Machine Learning pipelines to perform the task. Then, a web app where you can input a message and get classification results.

![Screenshot of App](Intro.PNG)

## File Description

        disaster_response_pipeline
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data
                |-- disaster_message.csv
                |-- disaster_categories.csv
                |-- DisasterResponse.db
                |-- process_data.py
          |-- models
                |-- classifier.zip   --> to be unzipped to have classifier.pkl
                |-- train_classifier.py
          |-- Preparation
                |-- categories.csv
                |-- ETL Pipeline Preparation.ipynb
                |-- ETL_Preparation.db
                |-- messages.csv
                |-- ML Pipeline Preparation.ipynb
                |-- README
          |-- README


1. App folder including the templates folder and "run.py" for the web application
2. Data folder containing "DisasterResponse.db", "disaster_categories.csv", "disaster_messages.csv" and "process_data.py" for data cleaning and transfering.
3. Models folder including "classifier.pkl" and "train_classifier.py" for the Machine Learning model.
4. README file
5. Preparation folder containing 6 different files, which were used for the project building. (Please note: this folder is not necessary for this project to run.)

## Installation
Must runing with Python 3 with libraries of numpy, pandas, sqlalchemy, re, NLTK, pickle, Sklearn, plotly and flask libraries.

## Instructions
1. Unzip the file : disaster_response_pipeline_project/models/classifier.zip
2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements
Feel free to use this app as long as it is for the greater good.

