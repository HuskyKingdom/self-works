import pandas as pd
from torch import float32

# reading files
train_data = pd.read_csv("./CA1data/train.data",header=None)
test_data = pd.read_csv("./CA1data/test.data",header=None)
# prepering training data
data_1 = train_data[(train_data[4] == "class-1") | (train_data[4] == "class-2")]
data_1.reset_index(drop=True,inplace=True)
data_2 = train_data[(train_data[4] == "class-2") | (train_data[4] == "class-3")]
data_2.reset_index(drop=True,inplace=True)
data_3 = train_data[(train_data[4] == "class-1") | (train_data[4] == "class-3")]
data_3.reset_index(drop=True,inplace=True)

t_data_1 = test_data[(test_data[4] == "class-1") | (test_data[4] == "class-2")]
t_data_1.reset_index(drop=True,inplace=True)
t_data_2 = test_data[(test_data[4] == "class-2") | (test_data[4] == "class-3")]
t_data_2.reset_index(drop=True,inplace=True)
t_data_3 = test_data[(test_data[4] == "class-1") | (test_data[4] == "class-3")]
t_data_3.reset_index(drop=True,inplace=True)

def preprocess(df,positive_label):
    # class label
    if df[4] == positive_label:
        df[4] = 1.0
    else:
        df[4] = -1.0
    return df

    


data_1 = data_1.apply(lambda x:preprocess(x,data_1[4][0]),axis = 1)
data_2 = data_2.apply(lambda x:preprocess(x,data_2[4][0]),axis = 1)
data_3 = data_3.apply(lambda x:preprocess(x,data_3[4][0]),axis = 1)

t_data_1 = t_data_1.apply(lambda x:preprocess(x,t_data_1[4][0]),axis = 1)
t_data_2 = t_data_2.apply(lambda x:preprocess(x,t_data_2[4][0]),axis = 1)
t_data_3 = t_data_3.apply(lambda x:preprocess(x,t_data_3[4][0]),axis = 1)


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


            
while True :
    print("\n Perceptron -- \n")
    print(">>Enter one of the following option to continue ..\n")
    usr_input = input("1. View datasets \n2. Training & Accuracy Report \n3. Terminate : ")
    if usr_input == "1":
        sets = [data_1,data_2,data_3,t_data_1,t_data_2,t_data_3]
        print("\n>> --- The dataset is in the order of :  ---\n")
        print(">> Training set (class1&2 , 2&3 , 3&1) , then Testing set  (class1&2 , 2&3 , 3&1) \n")
        for set in sets :
            print(set,"\n ---------- \n")
        print("\n>> --- The dataset is in the order of :  ---\n")
        print(">> Training set (class1&2 , 2&3 , 3&1) , then Testing set  (class1&2 , 2&3 , 3&1) \n")
    elif usr_input == "2":

        print(">>Choose a dataset in training dataset to train. With fixed 20 iterations.") # train
        usr_input = input("1. class 1 & 2 \n2. class 2 & 3 \n3. class 1 & 3 : ")
        if usr_input == "1":
            b,w = perceptron_train(data_1,1,20)
            print("Parameters Learned : b = " + str(b) + " ; w = ",w," \n")
        elif usr_input == "2":
            b,w = perceptron_train(data_2,1,20)
            print("Parameters Learned : b = " + str(b) + " ; w = ",w," \n")
        elif usr_input == "3":
            b,w = perceptron_train(data_3,1,20)
            print("Parameters Learned : b = " + str(b) + " ; w = ",w," \n")
        else:
            print(" Invalid input . \n")
            continue
        
        print(">>Choose a dataset in dataset to test the result and report the accurracy.") # test
        usr_input = input(" 1. Trainingdata - class 1 & 2 \n 2. Trainingdata - class 2 & 3 \n 3. Trainingdata - class 1 & 3 \n 4. Testdata - class 1 & 2 \n 5. Testdata - class 2 & 3 \n 6. Testdata - class 1 & 3 \n : ")
        if usr_input == "1":
            result =  perceptron_application(data_1,b,w)
            print("Output classes : ", result)
            print("Accracy : " ,calculate_accur(result,data_1))
        elif usr_input == "2":
            result =  perceptron_application(data_2,b,w)
            print("Output classes : ", result)
            print("Accracy : " ,calculate_accur(result,data_2))
        elif usr_input == "3":
            result =  perceptron_application(data_3,b,w)
            print("Output classes : ", result)
            print("Accracy : " ,calculate_accur(result,data_3))
        elif usr_input == "4":
            result =  perceptron_application(t_data_1,b,w)
            print("Output classes : ", result)
            print("Accracy : " ,calculate_accur(result,t_data_1))
        elif usr_input == "5":
            result =  perceptron_application(t_data_2,b,w)
            print("Output classes : ", result)
            print("Accracy : " ,calculate_accur(result,t_data_2))         
        elif usr_input == "6":
            result =  perceptron_application(t_data_3,b,w)
            print("Output classes : ", result)
            print("Accracy : " ,calculate_accur(result,t_data_3))
        else:
            print(" Invalid input . \n")
            continue

    elif usr_input == "3":
        print("\nBye.")
        break
    else:
        print(" Invalid input . \n")
            
   



