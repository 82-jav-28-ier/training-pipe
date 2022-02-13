ARG region

# SageMaker PyTorch image
FROM 171803786209.dkr.ecr.us-east-1.amazonaws.com/python:3.8-slim-local

ENV PATH="/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
COPY train.py /opt/ml/code/train.py

# Define script entrypoint 
ENV SAGEMAKER_PROGRAM train.py