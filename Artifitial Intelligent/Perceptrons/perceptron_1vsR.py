from numpy import True_
import pandas as pd
from torch import float32

# reading files
train_data = pd.read_csv("./CA1data/train.data",header=None)
test_data = pd.read_csv("./CA1data/test.data",header=None)
# prepering training data
data_1 = train_data.copy(deep=True)
data_2 = train_data.copy(deep=True)
data_3 = train_data.copy(deep=True)

def preprocess(df,positive_label):
    # class label
    if df[4] == positive_label:
        df[4] = 1.0
    else:
        df[4] = -1.0
    return df

def test_preprocess(df):
    if df[4] == "class-1":
        df[4] = 0
    elif df[4] == "class-2":
        df[4] = 1
    else:
        df[4] = 2
    return df

    
data_1 = data_1.apply(lambda x:preprocess(x,"class-1"),axis = 1)
data_2 = data_2.apply(lambda x:preprocess(x,"class-2"),axis = 1)
data_3 = data_3.apply(lambda x:preprocess(x,"class-3"),axis = 1)

test_data = test_data.apply(lambda x:test_preprocess(x),axis = 1)
train_data = train_data.apply(lambda x:test_preprocess(x),axis = 1)



# perceptron training
def perceptron_train(data,learning_rate,MaxInte):
    b = 0
    w = [0.0 for i in range(data.shape[1] - 1)]
    for i in range(MaxInte):
        for row in range(data.shape[0]):
            # compute output y
            sum = 0
            y = 0
            for col in range(len(w)): # compute some of products
                sum += w[col] * data[col][row] 
            sum = sum + b

            if sum > 0 : # compute activation function
                y = 1
            else:
                y = -1
            
            # justify if to do update 
            if y * data[4][row] <= 0:
                # update w
                for index in range(len(w)):
                    w[index] = w[index] + learning_rate * data[4][row] * data[index][row]
                b = b + data[4][row]
    
    return b,w


def perceptron_application(data,b,w):
    return_val = []
    for row in range(data.shape[0]):
            # compute output y
            sum = 0
            y = 0
            for col in range(len(w)): # compute some of products
                sum += w[col] * data[col][row] 
            sum = sum + b

            if sum > 0 : # compute activation function
                y = 1
            else:
                y = -1

            return_val.append(y)

    return return_val

def calculate_accur(predicted,actual_dataset):

    totoal_correct = 0
    for i in range(len(predicted)):
        if predicted[i] == actual_dataset[4][i]:
            totoal_correct += 1

    return str(totoal_correct / len(predicted) * 100) + "%"

def Multi_class_voting(result): # the result parameter consists of 3 arraies of different prediction value for 3 different classifiers

    return_val = []

    for individual_result in range(len(result[0])): # comparing all 3 results

        if result[0][individual_result] == 1 :
            return_val.append(0)
        elif result[1][individual_result] == 1:
            return_val.append(1)
        elif result[2][individual_result] == 1:
            return_val.append(2)
        else:
            return_val.append(-1) # indicating classification falier


    return return_val


            
while True :
    print("\n Multi-Class Perceptron -- \n")
    print(">>Enter one of the following option to continue ..\n")
    usr_input = input("1. View datasets \n2. Training & Accuracy Report \n3. Terminate : ")
    if usr_input == "1":

        sets = [train_data,test_data]
        print("\n>> --- The dataset is in the order of :  ---\n")
        print(">> Training set  , then Testing set  \n")
        for set in sets :
            print(set,"\n ---------- \n")
        print("\n>> --- The dataset is in the order of :  ---\n")
        print(">> Training set  , then Testing set \n")

    elif usr_input == "2":

        print("\n>>Training 3 classifiers...\n") # train
        b1,w1 = perceptron_train(data_1,1,20)
        b2,w2 = perceptron_train(data_2,1,20)
        b3,w3 = perceptron_train(data_3,1,20)
        print(">>Training completed...\n")
        
        print(">>Choose a dataset in dataset to test the result and report the accurracy.") # test
        usr_input = input(" 1. Trainingdata \n 2. Testdata  \n : ")
        if usr_input == "1":

            result = []
            result.append(perceptron_application(train_data,b1,w1))
            result.append(perceptron_application(train_data,b2,w2))
            result.append(perceptron_application(train_data,b3,w3))
            final_result = Multi_class_voting(result)
            print("Output classes : ", final_result)
            print("Accracy : " ,calculate_accur(final_result,train_data))

       

        elif usr_input == "2":

            result = []
            result.append(perceptron_application(test_data,b1,w1))
            result.append(perceptron_application(test_data,b2,w2))
            result.append(perceptron_application(test_data,b3,w3))
            final_result = Multi_class_voting(result)
            print("Output classes : ", final_result)
            print("Accracy : " ,calculate_accur(final_result,test_data))

        else:
            print(" Invalid input . \n")
            continue

    elif usr_input == "3":
        print("\nBye.")
        break
    else:
        print(" Invalid input . \n")
            
   



