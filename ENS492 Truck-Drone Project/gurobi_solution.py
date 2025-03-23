import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from gurobipy import Model, GRB, quicksum
import pandas as pd

# Read data from Excel
file_path = "final_customer_data.xlsx"
customer_data = pd.read_excel(file_path, sheet_name="Customer Data")
distance_matrix = pd.read_excel(file_path, sheet_name="Distance Matrix", index_col=0)

# Problem parameters
Depot = 0
N = list(customer_data["Customer_ID"])
Z = [1, 2, 3]  # Three trucks
D = [1, 2]  # Two drones
truck_routes = {z: [] for z in Z}

# Convert distance matrix to dictionary format
dist = {(i, j): distance_matrix.at[i, j] for i in [Depot] + N for j in [Depot] + N}

# Vehicle capacity parameters
Q_truck = 60  # Truck capacity
Q_drone = 15  # Drone capacity
R_max = 40    # Maximum drone range

# Customer demands
demand = dict(zip(customer_data["Customer_ID"], customer_data["Demand"]))

# Model definition
model = Model("Truck-Drone Routing")

# Decision variables
x = model.addVars(N + [Depot], N + [Depot], Z, vtype=GRB.BINARY, name="x")  # Truck routes
y = model.addVars(D, N, vtype=GRB.BINARY, name="y")  # Drone delivery assignments
s = model.addVars(D, N, Z, vtype=GRB.BINARY, name="s")  # Drone launch points
r = model.addVars(D, N, Z, vtype=GRB.BINARY, name="r")  # Drone-truck assignments

# Objective function: minimize total distance
model.setObjective(
    quicksum(dist[i, j] * x[i, j, z] for i in N + [Depot] for j in N + [Depot] for z in Z if i != j) +
    quicksum((quicksum(s[d, k, z] * dist[z, k] for z in Z)) * y[d, k] for d in D for k in N),
    GRB.MINIMIZE
)

# Every customer must be visited exactly once (by truck or drone)
for k in N:
    model.addConstr(quicksum(x[i, k, z] for i in N + [Depot] for z in Z if i != k) +
                    quicksum(y[d, k] for d in D) == 1, name=f"Customer_Served_{k}")

# Flow conservation for trucks
for k in N:
    for z in Z:
        model.addConstr(
            quicksum(x[i, k, z] for i in N + [Depot] if i != k) ==
            quicksum(x[k, j, z] for j in N + [Depot] if j != k),
            name=f"Truck_Flow_{k}_{z}"
        )

# MTZ constraints to eliminate subtours
u = model.addVars(N, Z, name="u")
for i in N:
    for j in N:
        if i != j:
            for z in Z:
                model.addConstr(
                    u[i, z] - u[j, z] + len(N) * x[i, j, z] <= len(N) - 1, 
                    name=f"MTZ_{i}_{j}_{z}"
                )

# Drone must launch from a truck route
for d in D:
    for k in N:
        model.addConstr(
            quicksum(s[d, k, z] for z in Z) == y[d, k],
            name=f"Drone_Assigned_To_Truck_{d}_{k}"
        )

# Drone range constraint
for d in D:
    for k in N:
        model.addConstr(
            quicksum(s[d, k, z] * dist[z, k] for z in Z) <= R_max * y[d, k],
            name=f"Drone_Range_{d}_{k}"
        )

# Drone-truck relationship
for d in D:
    for k in N:
        model.addConstr(
            quicksum(r[d, k, z] for z in Z) == y[d, k],
            name=f"Drone_Launch_Assignment_{d}_{k}"
        )

# Truck capacity constraint
for z in Z:
    model.addConstr(
        quicksum(demand[j] * x[i, j, z] for i in N + [Depot] for j in N if i != j) <= Q_truck,
        name=f"Truck_Capacity_{z}"
    )

# Drone capacity constraint
for d in D:
    model.addConstr(
        quicksum(demand[k] * y[d, k] for k in N) <= Q_drone,
        name=f"Drone_Capacity_{d}"
    )

# Each truck must serve at least one customer
for z in Z:
    model.addConstr(
        quicksum(x[i, j, z] for i in N + [Depot] for j in N if i != j and j != Depot) >= 1,
        name=f"Min_Truck_{z}_Customers"
    )

# Each drone must serve at least one customer
for d in D:
    model.addConstr(
        quicksum(y[d, k] for k in N) >= 1,
        name=f"Min_Drone_{d}_Usage"
    )

# Maximum customers per truck (for load balancing)
max_customers_per_truck = 15
for z in Z:
    model.addConstr(
        quicksum(x[i, j, z] for i in N + [Depot] for j in N if i != j and j != Depot) <= max_customers_per_truck,
        name=f"Max_Truck_{z}_Customers"
    )

# Ensure all customers are visited
model.addConstr(
    quicksum(x[i, j, z] for i in N + [Depot] for j in N for z in Z if i != j and j != Depot) +
    quicksum(y[d, k] for d in D for k in N) == len(N),
    name="Total_Customers_Visited"
)

# All trucks must start and end at the depot
for z in Z:
    model.addConstr(quicksum(x[Depot, j, z] for j in N) >= 1, name=f"Truck_Start_{z}")
    model.addConstr(quicksum(x[i, Depot, z] for i in N) >= 1, name=f"Truck_Return_{z}")

# Solver settings
model.setParam('OutputFlag', 1)
model.setParam('TimeLimit', 300)       # 5 minutes
model.setParam('MIPGap', 0.05)         # 5% optimality gap
model.setParam('MIPFocus', 1)          # Focus on finding feasible solutions
model.setParam('Threads', 0)           # Use maximum threads
model.setParam('Method', 3)            # Concurrent optimization
model.setParam('Cuts', 2)              # Aggressive cutting planes

model.optimize()

# Output results
if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
    print(f"Objective function value: {model.objVal}")
    print(f"Solution status: {model.status}")
    print(f"Solution quality (MIP Gap): {model.MIPGap:.2%}")
    
    visited = set()

    for z in Z:
        print(f"\nTruck {z} route:")
        route = [Depot]
        current = Depot
        
        while True:
            next_node = None
            for j in N + [Depot]:
                if j != current and x[current, j, z].x > 0.5:
                    next_node = j
                    route.append(j)
                    if j != Depot:
                        visited.add(j)
                    current = j
                    break
            if next_node is None or current == Depot:
                break
        if route[-1] != Depot:
            route.append(Depot)
        
        truck_routes[z] = route
        print(f"Route: {' -> '.join(map(str, route))}")
        
        total_load = sum(demand.get(j, 0) for j in route if j != Depot)
        print(f"Total load: {total_load}/{Q_truck}")
        print(f"Customer count: {len(route) - 2}")

    for d in D:
        print(f"\nDrone {d} route:")
        
        drone_customers = []
        
        for k in N:
            if y[d, k].x > 0.5:
                drone_customers.append(k)
                visited.add(k)
                
                launch_truck = None
                for z in Z:
                    if r[d, k, z].x > 0.5:
                        launch_truck = z
                        break
                
                if launch_truck is not None:
                    min_dist = float('inf')
                    closest_node = None
                    
                    for node in truck_routes[launch_truck]:
                        if node != Depot and dist[node, k] < min_dist:
                            min_dist = dist[node, k]
                            closest_node = node
                    
                    if closest_node is not None:
                        print(f"Customer {k}: Launch from Truck {launch_truck}, Node {closest_node}, Distance {min_dist}")
                    else:
                        print(f"Customer {k}: Launch from Truck {launch_truck} (launch node not found)")
                else:
                    print(f"Customer {k}: Launch point not found")
        
        if not drone_customers:
            print("Drone not used.")
        else:
            print(f"Total customers: {len(drone_customers)}")
            
            drone_load = sum(demand.get(k, 0) for k in drone_customers)
            print(f"Total load: {drone_load}/{Q_drone}")
    
    print("\nSummary:")
    print(f"Visited customers: {len(visited)}")
    print(f"Total customers: {len(N)}")
    
    unvisited = set(N) - visited
    if unvisited:
        print(f"Unvisited customers: {sorted(unvisited)}")
    else:
        print("All customers visited.")
    
    total_truck_distance = sum(dist[i, j] * x[i, j, z].x 
                           for i in N + [Depot] for j in N + [Depot] for z in Z if i != j)
    total_drone_distance = sum(sum(s[d, k, z].x * dist[z, k] 
                              for z in Z) * y[d, k].x for d in D for k in N)
    
    print(f"\nTotal truck distance: {total_truck_distance:.2f}")
    print(f"Total drone distance: {total_drone_distance:.2f}")
    print(f"Total distance: {total_truck_distance + total_drone_distance:.2f}")
else:
    print("Model could not find optimal solution. Status code:", model.status)

# Visualization
if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
    np.random.seed(42)
    coordinates = {0: (50, 50)}
    
    for i in N:
        coordinates[i] = (np.random.randint(0, 100), np.random.randint(0, 100))
    
    plt.figure(figsize=(15, 10))
    
    plt.scatter(*coordinates[0], s=200, c='black', marker='s', label='Depot')
    
    for i in N:
        plt.scatter(*coordinates[i], s=100, c='blue', alpha=0.5)
        plt.text(*coordinates[i], f" {i}", fontsize=9)
    
    truck_colors = ['red', 'green', 'orange']
    
    for z, color in zip(Z, truck_colors):
        route = truck_routes[z]
        if len(route) > 1:
            for i in range(len(route) - 1):
                from_node, to_node = route[i], route[i+1]
                plt.plot([coordinates[from_node][0], coordinates[to_node][0]],
                         [coordinates[from_node][1], coordinates[to_node][1]],
                         c=color, linewidth=2, alpha=0.7)
            
            plt.text(*coordinates[route[1]], f" Truck {z}", fontsize=10, 
                     color=color, fontweight='bold')
    
    drone_colors = ['purple', 'brown']
    
    for d, color in zip(D, drone_colors):
        for k in N:
            if y[d, k].x > 0.5:
                launch_truck = None
                for z in Z:
                    if r[d, k, z].x > 0.5:
                        launch_truck = z
                        break
                
                if launch_truck is not None:
                    min_dist = float('inf')
                    closest_node = None
                    
                    for node in truck_routes[launch_truck]:
                        if node != Depot and dist[node, k] < min_dist:
                            min_dist = dist[node, k]
                            closest_node = node
                    
                    if closest_node is not None:
                        plt.plot([coordinates[closest_node][0], coordinates[k][0]],
                                 [coordinates[closest_node][1], coordinates[k][1]],
                                 c=color, linestyle='--', linewidth=1.5, alpha=0.8)
                        
                        plt.scatter(*coordinates[k], s=120, c=color, marker='^')
    
    custom_lines = [
        plt.Line2D([0], [0], color='black', marker='s', linestyle='', markersize=10),
        plt.Line2D([0], [0], color='blue', marker='o', linestyle='', markersize=10, alpha=0.5),
    ]
    
    for z, color in zip(Z, truck_colors):
        custom_lines.append(plt.Line2D([0], [0], color=color, lw=2))
    
    for d, color in zip(D, drone_colors):
        custom_lines.append(plt.Line2D([0], [0], color=color, linestyle='--', lw=2))
        custom_lines.append(plt.Line2D([0], [0], color=color, marker='^', linestyle='', markersize=10))
    
    labels = ['Depot', 'Customer']
    for z in Z:
        labels.append(f'Truck {z} Route')
    for d in D:
        labels.append(f'Drone {d} Route')
        labels.append(f'Drone {d} Customer')
    
    plt.legend(custom_lines, labels, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.title('Truck-Drone Routing Solution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.savefig('truck_drone_routes.png', dpi=300, bbox_inches='tight')
    plt.show()