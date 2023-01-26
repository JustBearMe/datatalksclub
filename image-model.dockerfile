FROM tensorflow/serving:2.10.1

COPY garbage-model /models/garbage-model/1
ENV MODEL_NAME="garbage-model"