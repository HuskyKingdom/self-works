# Comp 337 Assignment 1
&nbsp; This file shows clear instruction of how to run the code. The assignment contains mainly 2 part
of python files.

### &nbsp; perceptron.py & perceptron_1vsR.py
&nbsp; The ealier python file contains the implementation of perceptrons required by the assignment specification 2) and 3), the later one contains the implementation for question 4). Beacause they follows the same clear commend line user interaction menu, this page would only brifly introduce the useage of one of them.</br>
&nbsp; Run the python file , the following menu would then be displayed:

    Multi-Class Perceptron -- 
    Enter one of the following option to continue ..

    1. View datasets 
    2. Training & Accuracy Report 
    3. Terminate : 

&nbsp; Choose one of the option from above with corresponding option mark to continue. Option 1 would display all the dataset read from the local file; option 2 will train the dataset and report the accracy. Let`s choose option 2:

    Choose a dataset in training dataset to train. With fixed 20 iterations.
     class 1 & 2 
     class 2 & 3 
     class 1 & 3 :

&nbsp; Note that in this binary training perceptron, the datasets are been splited into 3 kinds as shown. Choose one of them to learn with and applie the learned weight and bias to one of exsisting datasets later:

Parameters Learned : b = 0.0 ; w =  [15.49999999999999, -0.19999999999999396, -23.300000000000008, -20.2]  

    Choose a dataset in dataset to test the result and report the accurracy.
    Trainingdata - class 1 & 2 
    Trainingdata - class 2 & 3 
    Trainingdata - class 1 & 3 
    Testdata - class 1 & 2 
    Testdata - class 2 & 3 
    Testdata - class 1 & 3 
    : 5
    
&nbsp; Choose option 5, the result of prediction and accracy is then shown:

    Output classes :  [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    Accracy :  50.0%