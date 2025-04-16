# ENS492-Truck-Drone-Project
Truck-Drone Project

#  Integrating Truck-Drone Systems for Enhanced Last-Mile Delivery

This project aims to revolutionize the last-mile delivery process by developing a hybrid logistics model that integrates trucks and unmanned aerial vehicles (drones). The proposed system leverages the high carrying capacity of trucks and the flexibility of drones to create an efficient, sustainable, and cost-effective delivery network.

While trucks operate as mobile bases carrying multiple packages across long distances, drones are used to reach final destinations—especially those in remote or congested urban areas. By coordinating these two delivery modes through advanced optimization algorithms, the system ensures:

- Faster and more reliable deliveries  
- Reduced greenhouse gas emissions  
- Lower operational costs  
- Better access to geographically challenging areas

The model is formulated as a Mixed-Integer Linear Program (MILP) that optimizes truck routes and drone flights under real-world constraints such as delivery time windows, payload limits, and battery capacity.

This project was developed as part of the ENS491 Graduation Project at Sabancı University and reflects a novel approach toward building more sustainable and scalable last-mile logistics systems.

---


You can access gurobi solution of the project from "ENS492 Truck-Drone Project/gurobi_solution.py". 

Here are all the libraries you need to install for this code:
Copypip install gurobipy
pip install pandas
pip install matplotlib
pip install numpy
pip install openpyxl  # For Excel file support with pandas
You'll need to run these commands in your terminal or command prompt. Note that Gurobi requires a license
