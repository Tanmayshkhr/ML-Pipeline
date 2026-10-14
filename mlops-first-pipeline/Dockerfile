#Pull the Python image from Docker Hub 
#Use a lightweight Python image to reduce the size of the final image
FROM python:3.14-slim

#Set the working directory in the container to /app
WORKDIR /app

#Copy the requirements.txt file from the host machine to the working directory in the container
COPY requirements.txt . 

#Install the dependencies specified in the requirements.txt file
RUN pip install -r requirements.txt

#Copy application code and saved model inside /app 
COPY . .

#Document the port used by the FastAPI application
EXPOSE 8000

#Start the API server when container is ready
#0.0.0.0 = allows connection through any container's network interface, not just localhost
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]