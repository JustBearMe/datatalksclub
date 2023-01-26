#!/usr/bin/env python
# coding: utf-8

import os
import grpc

import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from keras_image_helper import create_preprocessor

from flask import Flask
from flask import request
from flask import jsonify

from proto import np_to_protobuf

host = os.getenv('TF_SERVING_HOST', 'localhost:8500')

channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

preprocessor = create_preprocessor('xception', target_size=(299, 299))


def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()

    pb_request.model_spec.name = 'garbage-model'
    pb_request.model_spec.signature_name = 'serving_default'

    pb_request.inputs['input_44'].CopyFrom(np_to_protobuf(X))
    return pb_request


classes = [
    'empty', 
    'full', 
    'halfempty'
]

def prepare_response(pb_response):
    preds = pb_response.outputs['dense_32'].float_val
    return dict(zip(classes, preds))


def predict(path):
    X = preprocessor.from_url(path)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    response = prepare_response(pb_response)
    return response


app = Flask('gateway')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    path = data['url']
    result = predict(path)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
