FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Set the working directory
WORKDIR /workspace
# Copy the requirements file into the container
COPY requirements.txt /workspace/requirements.txt
# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade keras>3.9.2

# Copy the rest of the application code into the container
COPY src /workspace/src
# Expose the port the app runs on
EXPOSE 8008
# Set the environment variable for TensorFlow
ENV TF_CPP_MIN_LOG_LEVEL=2
# Set the environment variable for TensorFlow to use GPU
ENV NVIDIA_VISIBLE_DEVICES=all


# Set the command to run the application
CMD ["/workspace/start.sh"]