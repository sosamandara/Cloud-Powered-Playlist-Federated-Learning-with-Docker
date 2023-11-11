# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Add the requirements.txt file to the container
ADD requirements /app

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements

# Copy the main_server.py file to the container
ADD master_node.py /app


# Expose port 80 for communication
EXPOSE 80

# Run the main_server.py script when the container launches
CMD ["python", "master_node.py"]
