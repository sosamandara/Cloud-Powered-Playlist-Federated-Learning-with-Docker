import os
import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import json
import logging
import time
import numpy as np

global_model = None
main_server_url = "http://master:80"

def send_model_to_main_server(weights, server_url):
    logger.debug('Sending Model to server...')
    try:
        url = f"{server_url}/get_model"
        headers = {'Content-Type': 'application/json'}
        
        # Convert NumPy arrays to Python lists
        weights['coefficients'] = weights['coefficients'].tolist()
        weights['intercept'] = weights['intercept'].tolist()
        
        data = json.dumps(weights)
        
        response = requests.post(url, headers=headers, data=data)

        if response.status_code == 200:
            logger.debug('Request successful!')
            return 200
        else:
            logger.debug(f"Received a non-200 status code: {response.status_code}")
            return response.status_code
    except Exception as e:
        logger.debug(f"Error in send_model_to_main_server: {e}")
        return 500  # You can return an appropriate error code for exceptions


# Get the container name
def get_partition():
    name =  os.environ["PARTITION_ID"]
    return name    


def train_on_slave_node(server_url,X_train,Y_train):
    logger.debug('Training started...')
    # Load the local training data on the slave node
    
    global_model.fit(X_train,Y_train)

    weights = {'coefficients': global_model.coef_,'intercept':global_model.intercept_}
    logger.debug(weights)
    
    del X_train,Y_train
    return send_model_to_main_server(weights,server_url)



def manual_predict(X, coefficients, intercept):
    # Convert coefficients to a NumPy array
    coefficients = np.array(coefficients)
    
    # Calculate the linear combination of features and coefficients
    linear_combination = np.dot(X, coefficients.T) + intercept

    # Apply the softmax function to calculate class probabilities
    exp_scores = np.exp(linear_combination)
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Get the class with the highest probability as the predicted class
    predicted_classes = np.argmax(probabilities, axis=1)

    return predicted_classes



def get_model_from_server(server_url,X_test,Y_test,logger):
    global global_model
    global id
    try:
        url = f"{server_url}/get_model"
        response = requests.get(url)
        data = response.json()
        # Check the HTTP status code to ensure a successful response
        if response.status_code == 200:
            if 'message' in data:

                id = int(data['id'])
                global_model  = LogisticRegression(solver='saga',max_iter=1000,random_state=23, multi_class='multinomial')
                return 200
            else:
                
                data = response.json()
                # Extract model parameters from the received JSON data
                model_params = data['model']
                #logger.debug('Received this params')
                #logger.debug(model_params['coefficients'])
                #logger.debug(model_params['intercept'])
                
                # Create a new LogisticRegression model and set its parameters
                new_model = LogisticRegression(solver='saga',max_iter=1000,random_state=23, multi_class='multinomial')
                new_model.coef_ = model_params['coefficients']
                new_model.intercept_ = model_params['intercept']
                # Update the global model with the new model
                
                logger.debug(new_model.coef_)
                logger.debug(new_model.intercept_)
                logger.debug('Updated Model received!')
                # logger.debug(X_test_scaled.head())
                # logger.debug(Y_test.head())
                try:
                    label = LabelEncoder()
                    Y_test = label.fit_transform(Y_test)
                except Exception as e:
                    logger.debug(f"Label encoding error: {e}")
                logger.debug(X_test.head(5))
                logger.debug(Y_test[:5])
                # Perform prediction using the coefficients and intercept
                #time.sleep(30)
                #try:
                #    Y_pred = new_model.predict_proba(X_test)
                #    Y_pred = np.argmax(Y_pred, axis=1)
                #except Exception as e:
                #    logger.debug(f"Error during prediction: {e}")
                model_coefficients = new_model.coef_
                model_intercept = new_model.intercept_
                predicted_classes = manual_predict(X_test, model_coefficients, model_intercept)
                # Now, you can calculate accuracy
                accuracy = accuracy_score(Y_test, predicted_classes)
                #logger.debug(accuracy)
                return accuracy,200
                #return 0,200
                #return 200
        else:
            logger.debug(f"Received a non-200 status code: {response.status_code}")
            return response.status_code
    except Exception as e:
        logger.debug(f"Error in get_model_from_server: {e}")
        return 500  # You can return an appropriate error code for exceptions
    
def send_accuracy(main_server_url,accuracy):
    logger.debug('Sending accuracy to master..')
    try:
        url  = f'{main_server_url}/test_model'
        headers = {'Content-Type': 'application/json'}
        data = json.dumps({'accuracy':accuracy})
        response = requests.post(url, headers=headers, data=data)
        logger.debug('Accuracy: '+str(accuracy))
        if response.status_code == 200:
            return 200
        else:
            logger.debug(f"Received a non-200 status code: {response.status_code}")
            return response.status_code
    except Exception as e:
        logger.debug(f"Error in test_new_model: {e}")
        return 500  # You can return an appropriate error code for exceptions

def prepare_train_test(logger):
    logger.debug('Splitting the dataset')
    if id < 0:
        return 'Error id not set'
    logger.debug(f'/app/dataset/partition{id}.csv')
    dataset_path = f'/app/dataset/partition{id}.csv'

    df = pd.read_csv(dataset_path)
    logger.debug(len(df))

    Y = df['decade']
    X = df.drop(columns=['decade'])
    logger.debug(Y.head(5))
    logger.debug(X.head(5))
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Fit and transform the training set
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
#
    ## Transform the testing set using the same scaler
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    
    #X_test.to_csv(f'X_test{id}.csv',index=False)
    #Y_test.to_csv(f'Y_test{id}.csv',index=False)
    # Split the data into training and testing sets
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    logger.debug(X_test.head(5))
    logger.debug(Y_test.head(5))

    del X,Y
    return X_train, Y_train, X_test, Y_test

def check_server(server_url):
    try:
        url = f"{server_url}/check_server"
        response = requests.get(url)
        data  = response.json()
        logger.info("Updates: "+str(data['updates']))
        if response.status_code==200:
            #logger.debug("Server IS ready...")
            return 200
    except Exception as e:
        #logger.debug(f"Server NOT ready... {e}")
        return 500  # You can return an appropriate error code for exceptions


if __name__ == '__main__':
    logging.basicConfig(
        filename='slave.log',
        level=logging.DEBUG,
        filemode='w'
    )
    logger = logging.getLogger()
    logger.debug('Slave Started')
    label = LabelEncoder()
    #Get Model from Master node
    sc=  get_model_from_server(main_server_url,None,None,logger)
    time.sleep(10)
    if(sc == 200 and global_model!=None):
        #Get dataset partition and split in train and test
        X_train, Y_train, X_test, Y_test = prepare_train_test(logger)


        Y_train = label.fit_transform(Y_train)
        Y_test = label.transform(Y_test)
        #Train the model on the partition and send the coefficients to the Master node
        sc = train_on_slave_node(main_server_url,X_train,Y_train)
        logger.debug(sc)
        time.sleep(10)
        check=False
        while(check==False):
            sc = check_server(main_server_url)
            if sc==200:
                check=True
                break
            time.sleep(20)

        if(check):
            logger.debug('Getting Updated Model..')
            time.sleep(10)
            accuracy, sc = get_model_from_server(main_server_url,X_test,Y_test,logger)
            logger.debug(sc)
            logger.debug(accuracy)

        if sc==200:
            send_accuracy(main_server_url,accuracy)
    
    
        
    
    
        


