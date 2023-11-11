# Cloud-Powered-Playlist-Federated-Learning-with-Docker

This repository presents an innovative solution for **era-based song classification**, employing federated learning within a **master-slave architecture** to address challenges related to *data privacy* and **security**. The key focus is on leveraging **Docker** for efficient weight distribution and isolation and **AWS** for **scalable deployment**.
This repository introduces a comprehensive **federated learning framework** designed with a focus on **flexibility, scalability, and security**. Our primary goal was to establish a versatile solution using Docker and AWS within a master-slave architecture, providing an easily replaceable infrastructure for datasets and models.

**Federated Learning Architecture:**

- **Master-Slave Architecture**: The system orchestrates federated learning using a **master node**, initiating the **logistic regression model**, and coordinating updates from **multiple slave nodes**.
  - Decentralized Collaboration: Enables collaborative model building while preserving data privacy and adhering to regulatory constraints.

    <img width="300" alt="plot1" src="https://user-images.githubusercontent.com/113529675/282260910-1b33bf11-20b4-47c9-a899-1d89d856d3dd.jpeg">

- **Dockerization for Enhanced Security**:
  - Docker Containers: Utilizes Docker to encapsulate and isolate the federated learning process, providing enhanced security and preventing unintended data leaks.
  - Efficient Weight Distribution: Docker efficiently transfers model weights between the master and slave nodes, ensuring consistency and reproducibility.
    
- **AWS Deployment for Scalability**:
  - Cloud Environment: The system is deployed on AWS for scalability and accessibility, allowing efficient collaboration among client nodes while adhering to stringent data security requirements.
  - AWS EC2 Instances: Multiple instances are launched, each running the federated learning application, showcasing scalability with varying numbers of slave nodes.

    <img width="900" alt="plot1" src="https://user-images.githubusercontent.com/113529675/282261076-82cfc045-12bf-427c-8c78-5b94c47b5212.png">

- **Implementation Details:**
  - RESTful API: Implemented using Flask to facilitate communication between master and slave nodes.
  - Coordination Point: The master node serves as the central coordination point for the federated learning process.

- **Slave Node Functionality:**
  - Python Client Script: The slave node communicates with the master node using well-defined RESTful API endpoints.
  - Logistic Regression Model: Each slave node uses a custom-built logistic regression model implemented in Python.

- **Dockerization and AWS Deployment:**
  - Docker Containers:
  - Isolated Environments: Docker ensures isolated environments for both master and slave nodes, enhancing security and reproducibility.
  - Efficient Distribution: Docker efficiently transfers model weights, encapsulating the process and preventing unauthorized access.

- **AWS Deployment Script:**
  - Scalability: The deployment script launches AWS EC2 instances with varying numbers of slave nodes, demonstrating the system's scalability.
  - CloudWatch Monitoring: Utilizes AWS CloudWatch for monitoring CPU and credit usage, ensuring efficient resource utilization.

- **Future Developments:**
  - Secondary Web Service: Planned future development includes a user-friendly web service on the master node, allowing users to predict song decades based on the trained model.
  - Explore this repository for a comprehensive solution that effectively combines federated learning, Dockerization, and AWS deployment for era-based song classification with a strong emphasis on data privacy and security.
