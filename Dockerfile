FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Set the working directory
WORKDIR /workspace
# Copy the requirements file into the container
COPY requirements.txt /workspace/requirements.txt
# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade keras>3.9.2


# Copy the rest of the application code into the container
COPY . /workspace
# Expose the port the app runs on
EXPOSE 8008
# Set the environment variable for TensorFlow
ENV TF_CPP_MIN_LOG_LEVEL=2
# Set the environment variable for TensorFlow to use GPU
ENV NVIDIA_VISIBLE_DEVICES=all

ADD https://web.archive.org/web/20220328223413if_/http://opihi.cs.uvic.ca/sound/genres.tar.gz#expand /workspace/data/
RUN tar -xvzf /workspace/data/genres.tar.gz -C /workspace/data/

# Set the command to run the application
CMD ["/workspace/start.sh"]