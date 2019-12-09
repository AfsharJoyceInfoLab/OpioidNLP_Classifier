@author Brihat Sharma

Introduction

Opioid classifier to identify opioid misuse from the Electronic Health Record of Emergency Department and Hospitalized Patients. The first 24hr of clinical notes are needed as an input, which should be first processed using Apache cTAKES to concept map the raw with UMLS into Concept Unique Identifiers (CUIs).

Dependencies Library: Pandas, os, pickle, numpy, keras, tensorflow

Steps:

cTAKES:

Download cTAKES from https://ctakes.apache.org/downloads.cgi
cTAKES comes with default dictionary, this dictionary can also be cutomized creating own version. Our dictionary consists of rxnorms and snomedCT but default dictionary also works well
Process the input data using cTAKES, this will crete .txt files with CUIs which will be input data to the model
Model:

Open the Opioid_Predict.py script and change the input and output directory
Run the sript as python3 Opioid_predict.py
The result will be inside the output directory inside a csv file, first column represents the files, second column represents predicted labels and the third column represents predict probability. 1 as current opioid misuse and 0 as no opioid misuse for the second column.
