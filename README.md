# Language-Identification-from-Text-Documents
Predicting language of sentence's using different machine learning algorithm

Problem : 
DSL dataset that contains sentences written in 14 different language. Some
machine learning technique was applied to train a system with this dataset so that it can predict the language
of a particular sentence.

Data Collection : 
Data were collected from the official repository for the Disciminating between Similar Language
(DSL) Shared Task 2015.
Link : https://github.com/Simdiva/DSL-Task

This repository contains the following :

• DSL Corpus Collection (DSLCC) version 2.0 (training, dev, test and gold data included)
• DSL Shared Task submissions from participating teams
• The script to blind Named Entities (NEs) for the Test Set B in DSL-2015 (blindNE.py)
• The evaluation script to evaluate outputs.
• The evaluation script to evaluate all submitted system.
Only train.txt file from train-dev folder was used for language identification task. This file
contains 2,52,000 sentences in 14 different language.

Experiment and Performance : 
I used Multinomial Naive Bayes and Logistic Regression for both character n-gram and word
n-gram features.
So, The following code has 4 different parts.

• Feature : Character n-grams, Algorithm : Logistic Regression
• Feature : Character n-grams, Algorithm : Multinomial Naive Bayes
• Feature : Word n-grams, Algorithm : Logistic Regression
• Feature : Word n-grams, Algorithm : Multinomial Naive Bayes

70% of total data set was used for training purpose and 30% of data set was used for testing
purpose. The implementation code is given in the next section.
