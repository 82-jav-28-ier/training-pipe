ARG region

FROM 171803786209.dkr.ecr.us-east-1.amazonaws.com/python:3.8-slim-local

ENV PATH="${PATH}:/opt/ml/code"

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
COPY train.py /opt/ml/code/train.py
COPY train /opt/ml/code/train
COPY requirements.txt /opt/ml/code/requirements.txt

WORKDIR /opt/ml/code/

RUN pip install --no-cache-dir -r requirements.txt

# Define script entrypoint 
ENV SAGEMAKER_PROGRAM train.py
ENTRYPOINT ["python3.8", "/opt/ml/code/train.py"]