from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
import os
import datetime
import logging
import numpy as np
import json
import threading
from collections import deque

num_slaves = int(os.environ['NUM_SLAVES'])
partitions = deque(list(range(num_slaves,0,-1)))

global_model = None

coeff_list = {
    'counter': 0,
    'coefficients': None,
    'intercept': None
}
counter  =0 
master_accuracy  = 0
mutex = threading.Lock()
# num_slave_nodes = 0 environment variable

app = Flask(__name__)

@app.route('/get_model', methods=['POST'])
def get_model():
    logger.debug(f'POST request from {request.remote_addr}')
    global global_model
    # Process the data received from the slave server
    data = request.get_json()

    # Check if the received data contains coefficients and an intercept
    if 'coefficients' not in data or 'intercept' not in data:
        return jsonify({'error': 'Invalid data format'}), 400

    #logger.debug(coeff_list['coefficients'])
    logger.debug( data['coefficients'])
    def process_data(data):
        logger.debug('Processing data...')
        try:
            with mutex:

                global coeff_list
                # Create or update the global model

                if coeff_list['coefficients'] is None:
                    coeff_list['coefficients'] =  np.zeros_like(data['coefficients'])
                else:
                    # Add the 'coefficients' element-wise using NumPy
                    coeff_list['coefficients'] = np.add(coeff_list['coefficients'], data['coefficients'])

                if coeff_list['intercept'] is None:
                    coeff_list['intercept'] = np.zeros_like(data['intercept'])
                else:
                    # Add the 'intercept' element-wise using NumPy
                    coeff_list['intercept'] = np.add(coeff_list['intercept'], data['intercept'])

                # Increment the counter
                coeff_list['counter'] += 1

        except Exception as e:
            return 'Fail'
        finally:
            return 'Success'
        

    result = process_data(data)
    logger.debug('Result: ' + result)
    n_slaves = int(os.environ['NUM_SLAVES'])

    logger.debug('number of updates: '+str(coeff_list['counter']))
    logger.debug(type(coeff_list['coefficients']))
    logger.debug(np.array(coeff_list['coefficients']).shape)
    if n_slaves == coeff_list['counter']:
        coeff_list['intercept'] = np.divide(coeff_list['intercept'], n_slaves)
        coeff_list['coefficients'] = np.divide(coeff_list['coefficients'], n_slaves)

        global_model = LogisticRegression(random_state=23)
        global_model.coef_ = coeff_list['coefficients']
        global_model.intercept_ = coeff_list['intercept']
        logger.debug('Model update successfully!')


    return jsonify({'message': 'Model updated successfully'}), 200


@app.route('/get_model', methods=['GET'])
def send_model():
    logger.debug(f'GET request from {request.remote_addr} ')
    global global_model
    if global_model is None:
        with mutex:
            id = partitions.pop()
        logger.debug('Model Initialized! ' + str(datetime.datetime.now().timestamp()) + ', Partiton '+str(id))
        return jsonify({'message': 'initialize model','id' : id}), 200
    else:
        weights = {
            'coefficients': global_model.coef_.tolist(),
            'intercept': global_model.intercept_.tolist(),
        }
        return jsonify({'model':weights}), 200


@app.route('/test_model', methods=['POST'])
def test_model():
    logger.debug(f'POST request from {request.remote_addr} ')
    global master_accuracy
    global counter
    try:
        data = request.get_json()
        if 'accuracy' not in data:
            return jsonify({'error': 'Invalid data format'}), 400
        else:
            with mutex:
                accuracy  =  data['accuracy']
                master_accuracy += accuracy
                counter += 1
                n_slaves = int(os.environ['NUM_SLAVES'])
                if(counter == n_slaves):
                    master_accuracy = master_accuracy / n_slaves
                    logger.info('Federated Learning Accuracy: '+str(master_accuracy))
                return jsonify({'Success': 'Accuracy arrived correctly'}), 200
    except Exception as e:
        logger.debug(f"Error in updating accuracy: {e}")
        return 500


@app.route('/check_server', methods=['GET'])
def check_server():
    global coeff_list
    #logger.debug(f'GET request from {request.remote_addr} ')
    if (coeff_list['counter']==num_slaves):
        return jsonify({'updates':coeff_list['counter']}),200
    else:
        return jsonify({'updates':coeff_list['counter']}),500
    

if __name__ == '__main__':
    logging.basicConfig(
        filename='master.log',
        level=logging.DEBUG,
        filemode='w'
    )

    logger = logging.getLogger()
    print('Master started')
    app.run(host='0.0.0.0', port=80)
