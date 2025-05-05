import gurobipy as gp
from gurobipy import GRB
import math
import numpy as np
import matplotlib.pyplot as plt
import time

# Parse the VRP file
def parse_vrp_file(filename):
    coordinates = {}
    demands = {}
    capacity = 0
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        # Find the capacity
        for line in lines:
            if line.strip().startswith('CAPACITY'):
                capacity = int(line.strip().split(':')[1].strip())
                break
        
        # Find coordinates and demands
        coord_section = False
        demand_section = False
        
        for line in lines:
            line = line.strip()
            
            if line == 'NODE_COORD_SECTION':
                coord_section = True
                continue
            elif line == 'DEMAND_SECTION':
                coord_section = False
                demand_section = True
                continue
            elif line == 'DEPOT_SECTION':
                demand_section = False
                continue
            
            if coord_section:
                parts = line.split()
                if len(parts) == 3:
                    node = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coordinates[node] = (x, y)
            
            if demand_section:
                parts = line.split()
                if len(parts) == 2:
                    node = int(parts[0])
                    demand = int(parts[1])
                    demands[node] = demand
    
    return coordinates, demands, capacity

# Calculate Euclidean distances
def calculate_distances(coordinates):
    nodes = list(coordinates.keys())
    distances = {}
    
    for i in nodes:
        for j in nodes:
            if i != j:
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[j]
                distances[(i, j)] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    return distances

# Adjust coordinates for better visualization
def adjust_coordinates(coordinates):
    adjusted = {}
    nodes = list(coordinates.keys())
    
    # Get the minimum and maximum values of x and y
    min_x = min(coordinates[node][0] for node in nodes)
    max_x = max(coordinates[node][0] for node in nodes)
    min_y = min(coordinates[node][1] for node in nodes)
    max_y = max(coordinates[node][1] for node in nodes)
    
    # Calculate current ranges
    x_range = max_x - min_x
    y_range = max_y - min_y
    
    # Scale factor for more spread
    scale_factor = 1.2
    
    # Adjust coordinates
    for node in nodes:
        x, y = coordinates[node]
        
        # Normalize to [0,1] range
        norm_x = (x - min_x) / x_range
        norm_y = (y - min_y) / y_range
        
        # Apply spacing adjustment
        adjusted_x = min_x + norm_x * x_range * scale_factor
        adjusted_y = min_y + norm_y * y_range * scale_factor
        
        # Extra adjustment to spread overlapping points
        for other_node in nodes:
            if other_node != node:
                other_x, other_y = coordinates[other_node]
                if abs(x - other_x) < 2 and abs(y - other_y) < 2:
                    # Small random offset for overlapping points
                    adjusted_x += np.random.uniform(-1.5, 1.5)
                    adjusted_y += np.random.uniform(-1.5, 1.5)
        
        adjusted[node] = (adjusted_x, adjusted_y)
    
    return adjusted

# Solve the Truck-Drone VRP
def solve_truck_drone_vrp(filename, max_time=300):
    start_time = time.time()
    
    # Parse the VRP file
    coordinates, demands, capacity = parse_vrp_file(filename)
    
    # Calculate distances
    distances = calculate_distances(coordinates)
    
    # Set up nodes
    nodes = list(coordinates.keys())
    depot = 1  # Depot node is 1
    customers = [node for node in nodes if node != depot]
    
    # Problem parameters
    num_trucks = 4  # Use 3 trucks
    num_drones = 3  # Use 2 drones
    
    # Truck and drone capacities
    truck_capacity = capacity  # Capacity from VRP file
    drone_capacity = 150        # Drone capacity
    max_drone_range = 100       # Maximum drone range
    
    # Find customers with small demands suitable for drone delivery
    customer_demands = [(c, demands[c]) for c in customers]
    sorted_customers = sorted(customer_demands, key=lambda x: x[1])
    
    # Select suitable customers for drones (lowest demand and within capacity)
    drone_candidates = []
    for c, d in sorted_customers:
        if d <= drone_capacity and len(drone_candidates) < num_drones * 3:  # Max 3 customers per drone
            drone_candidates.append(c)
    
    # Customers not suitable for drones will be assigned to trucks
    truck_customers = [c for c in customers if c not in drone_candidates]
    
    # Balance truck loads
    truck_clusters = [[] for _ in range(num_trucks)]
    truck_loads = [0] * num_trucks
    
    # Sort by demand (largest first) and use greedy assignment
    sorted_truck_customers = sorted([(c, demands[c]) for c in truck_customers], key=lambda x: x[1], reverse=True)
    
    for customer, demand in sorted_truck_customers:
        # Find the least loaded truck
        min_load_truck = min(range(num_trucks), key=lambda t: truck_loads[t])
        
        # If capacity is not exceeded, add to this truck
        if truck_loads[min_load_truck] + demand <= truck_capacity:
            truck_clusters[min_load_truck].append(customer)
            truck_loads[min_load_truck] += demand
        else:
            # Find first truck where customer fits
            for t in range(num_trucks):
                if truck_loads[t] + demand <= truck_capacity:
                    truck_clusters[t].append(customer)
                    truck_loads[t] += demand
                    break
            else:
                # If no truck has capacity, add to least loaded truck anyway
                truck_clusters[min_load_truck].append(customer)
                truck_loads[min_load_truck] += demand
    
    # Solve TSP for each truck
    truck_routes = []
    truck_total_distance = 0
    
    for t in range(num_trucks):
        if not truck_clusters[t]:
            continue
            
        # Create TSP model
        cluster_nodes = [depot] + truck_clusters[t]
        
        tsp_model = gp.Model(f"TSP_Truck_{t}")
        tsp_model.setParam('OutputFlag', 0)  # Suppress output messages
        
        # Variables
        x = {}
        for i in cluster_nodes:
            for j in cluster_nodes:
                if i != j:
                    x[i, j] = tsp_model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
        
        # MTZ subtour elimination variables
        u = {}
        for i in cluster_nodes:
            if i != depot:
                u[i] = tsp_model.addVar(lb=0, ub=len(cluster_nodes)-1, name=f'u_{i}')
        
        # Objective function
        tsp_model.setObjective(
            gp.quicksum(distances[i, j] * x[i, j] for i in cluster_nodes for j in cluster_nodes if i != j),
            GRB.MINIMIZE
        )
        
        # Each customer must be visited exactly once
        for j in cluster_nodes:
            if j != depot:
                tsp_model.addConstr(
                    gp.quicksum(x[i, j] for i in cluster_nodes if i != j) == 1,
                    name=f"visit_to_{j}"
                )
                
                tsp_model.addConstr(
                    gp.quicksum(x[j, i] for i in cluster_nodes if i != j) == 1,
                    name=f"visit_from_{j}"
                )
        
        # Start and end at depot
        tsp_model.addConstr(
            gp.quicksum(x[depot, j] for j in cluster_nodes if j != depot) == 1,
            name="depot_out"
        )
        tsp_model.addConstr(
            gp.quicksum(x[i, depot] for i in cluster_nodes if i != depot) == 1,
            name="depot_in"
        )
        
        # Subtour elimination constraints (MTZ formulation)
        for i in cluster_nodes:
            if i != depot:
                for j in cluster_nodes:
                    if j != depot and i != j:
                        tsp_model.addConstr(
                            u[i] - u[j] + len(cluster_nodes) * x[i, j] <= len(cluster_nodes) - 1,
                            name=f"mtz_{i}_{j}"
                        )
        
        # Limit solution time
        remaining_time = max(1, max_time - (time.time() - start_time))
        tsp_model.setParam('TimeLimit', remaining_time / num_trucks)  # Equal time for each truck
        
        # Solve
        tsp_model.optimize()
        
        # Process results
        if tsp_model.Status == GRB.OPTIMAL or tsp_model.Status == GRB.TIME_LIMIT:
            route = [depot]
            current = depot
            
            # Construct route
            while True:
                next_node = None
                for j in cluster_nodes:
                    if j != current and (current, j) in x and x[current, j].X > 0.5:
                        next_node = j
                        route.append(j)
                        current = j
                        break
                        
                if next_node is None or current == depot:
                    break
                    
            # Return to depot if route doesn't end there
            if route[-1] != depot:
                route.append(depot)
            
            # Store route information
            route_demand = sum(demands.get(node, 0) for node in route if node != depot)
            route_distance = sum(distances.get((route[i], route[i+1]), 0) for i in range(len(route)-1))
            truck_total_distance += route_distance
            
            truck_routes.append((t, route, route_demand, route_distance))
    
    # Assign customers to drones with updated drone routing approach
    drone_assignments = []
    drone_total_distance = 0
    
    # Get position of each node in truck routes (for synchronization)
    node_positions = {}
    for t, route, _, _ in truck_routes:
        for pos, node in enumerate(route):
            if node not in node_positions:
                node_positions[node] = []
            node_positions[node].append((t, pos))
    
    for d in range(num_drones):
        drone_load = 0
        drone_customers = []
        
        # Select customers within range and capacity for each drone
        for c in drone_candidates[:]:
            if drone_load + demands[c] <= drone_capacity:
                # Find best launch and rendezvous points for this customer
                best_dist = float('inf')
                best_launch = None
                best_launch_truck = None
                best_rendezvous = None
                best_rendezvous_truck = None
                
                # Try all possible truck nodes as launch and rendezvous points
                for launch_truck, launch_route, _, _ in truck_routes:
                    for launch_idx, launch_node in enumerate(launch_route):
                        if launch_node == depot:  # Skip depot as launch point
                            continue
                            
                        launch_dist = distances.get((launch_node, c), float('inf'))
                        
                        # Check if customer is within drone range from this launch point
                        if launch_dist <= max_drone_range / 2:  # Need to reserve range for return trip
                            # Try each possible rendezvous point (could be same truck or different)
                            for rendez_truck, rendez_route, _, _ in truck_routes:
                                for rendez_idx, rendez_node in enumerate(rendez_route):
                                    if rendez_node == depot:  # Skip depot as rendezvous
                                        continue
                                        
                                    rendez_dist = distances.get((c, rendez_node), float('inf'))
                                    
                                    # Check if rendezvous point is within range
                                    if rendez_dist <= max_drone_range / 2:
                                        # If same truck, ensure rendezvous comes after launch
                                        if launch_truck == rendez_truck and rendez_idx <= launch_idx:
                                            continue
                                            
                                        total_trip = launch_dist + rendez_dist
                                        
                                        # Update if this is the best option so far
                                        if total_trip < best_dist:
                                            best_dist = total_trip
                                            best_launch = launch_node
                                            best_launch_truck = launch_truck
                                            best_rendezvous = rendez_node
                                            best_rendezvous_truck = rendez_truck
                
                # If valid launch and rendezvous points are found, assign to this drone
                if best_launch is not None and best_rendezvous is not None:
                    drone_customers.append((c, best_launch_truck, best_launch, best_rendezvous_truck, best_rendezvous, best_dist))
                    drone_load += demands[c]
                    drone_candidates.remove(c)
                    drone_total_distance += best_dist
                    
                    # Limit to 3 customers per drone
                    if len(drone_customers) >= 3:
                        break
        
        drone_assignments.append((d, drone_customers, drone_load))
    
    # Print results
    print(f"\nTotal solution time: {time.time() - start_time:.2f} seconds")
    print(f"Total distance: {truck_total_distance + drone_total_distance:.2f}")
    
    print("\nTruck Routes:")
    for t, route, demand, distance in truck_routes:
        print(f"Truck {t+1}: {route} - Demand: {demand}/{truck_capacity} - Distance: {distance:.2f}")
    
    print("\nDrone Assignments:")
    for d, customers, load in drone_assignments:
        print(f"Drone {d+1} - Demand: {load}/{drone_capacity}")
        for c, launch_truck, launch_node, rendez_truck, rendez_node, dist in customers:
            print(f"  Customer {c}: Launch from Truck {launch_truck+1} at node {launch_node}, "
                  f"Rendezvous with Truck {rendez_truck+1} at node {rendez_node}, "
                  f"Total flight distance: {dist:.2f}")
    
    # Adjust coordinates for better visualization
    vis_coordinates = adjust_coordinates(coordinates)
    
    # Visualization
    plt.figure(figsize=(14, 10))
    
    # Draw nodes with improved spacing
    for node, (x, y) in vis_coordinates.items():
        if node == depot:  # Depot
            plt.scatter(x, y, s=250, c='red', marker='s', edgecolor='black', zorder=100)
            plt.text(x+1.5, y+1.5, f"Depot", fontsize=14, fontweight='bold')
        else:  # Customers
            is_drone_customer = any(c[0] == node for _, customers, _ in drone_assignments for c in customers)
            if is_drone_customer:
                plt.scatter(x, y, s=120, c='lime', alpha=0.8, edgecolor='black', zorder=90)
            else:
                plt.scatter(x, y, s=120, c='royalblue', alpha=0.8, edgecolor='black', zorder=90)
            
            # Add node labels with slight offset to avoid overlap
            offset_x = 1.0
            offset_y = 1.0
            if node in [23, 24, 31, 13, 14, 18, 10, 7]:  # Manually adjust overlapping labels
                offset_y += 1.0
            if node in [11, 28, 3, 21]:
                offset_x -= 2.0
            if node in [2, 16, 20, 15, 25]:
                offset_y -= 2.0
            
            plt.text(x+offset_x, y+offset_y, f"{node}", fontsize=12, fontweight='bold')
    
    # Draw truck routes
    truck_colors = ['darkorange', 'darkblue', 'darkgreen', 'brown', 'purple']
    
    for i, (t, route, _, _) in enumerate(truck_routes):
        color = truck_colors[i % len(truck_colors)]
        for j in range(len(route)-1):
            node1, node2 = route[j], route[j+1]
            x1, y1 = vis_coordinates[node1]
            x2, y2 = vis_coordinates[node2]
            plt.plot([x1, x2], [y1, y2], c=color, linewidth=2.5, alpha=0.8, zorder=80)
            
            # Add direction arrows
            arrow_x = x1 + 0.6*(x2-x1)
            arrow_y = y1 + 0.6*(y2-y1)
            dx = (x2-x1) * 0.1
            dy = (y2-y1) * 0.1
            plt.arrow(arrow_x, arrow_y, dx, dy, head_width=1.5, head_length=1.5, 
                      fc=color, ec=color, zorder=85)
    
    # Draw drone connections (dashed lines)
    drone_colors = ['magenta', 'darkviolet', 'cyan', 'orange', 'lime']
    
    for i, (d, customers, _) in enumerate(drone_assignments):
        color = drone_colors[i % len(drone_colors)]
        for c, launch_truck, launch_node, rendez_truck, rendez_node, _ in customers:
            # Draw the complete drone trip: launch point -> customer -> rendezvous point
            launch_x, launch_y = vis_coordinates[launch_node]
            cust_x, cust_y = vis_coordinates[c]
            rendez_x, rendez_y = vis_coordinates[rendez_node]
            
            # Draw first leg: launch -> customer
            plt.plot([launch_x, cust_x], [launch_y, cust_y], c=color, linestyle='--', 
                     linewidth=2, alpha=0.9, zorder=85)
            
            # Add direction arrow
            arrow_x = launch_x + 0.6*(cust_x-launch_x)
            arrow_y = launch_y + 0.6*(cust_y-launch_y)
            dx = (cust_x-launch_x) * 0.1
            dy = (cust_y-launch_y) * 0.1
            plt.arrow(arrow_x, arrow_y, dx, dy, head_width=1.5, head_length=1.5, 
                      fc=color, ec=color, zorder=85)
            
            # Draw second leg: customer -> rendezvous
            plt.plot([cust_x, rendez_x], [cust_y, rendez_y], c=color, linestyle='-.', 
                     linewidth=2, alpha=0.9, zorder=85)
            
            # Add direction arrow
            arrow_x = cust_x + 0.6*(rendez_x-cust_x)
            arrow_y = cust_y + 0.6*(rendez_y-cust_y)
            dx = (rendez_x-cust_x) * 0.1
            dy = (rendez_y-cust_y) * 0.1
            plt.arrow(arrow_x, arrow_y, dx, dy, head_width=1.5, head_length=1.5, 
                      fc=color, ec=color, zorder=85)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=15, 
                  markeredgecolor='black', label='Depot'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', markersize=15, 
                  markeredgecolor='black', label='Truck Customer'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=15, 
                  markeredgecolor='black', label='Drone Customer')
    ]
    
    for i in range(len(truck_routes)):
        legend_elements.append(plt.Line2D([0], [0], color=truck_colors[i % len(truck_colors)], 
                                         linewidth=3, label=f'Truck {i+1}'))
    
    for i in range(len(drone_assignments)):
        legend_elements.append(plt.Line2D([0], [0], color=drone_colors[i % len(drone_colors)], 
                                         linestyle='--', linewidth=2.5, label=f'Drone {i+1} (Launch → Customer)'))
        legend_elements.append(plt.Line2D([0], [0], color=drone_colors[i % len(drone_colors)], 
                                         linestyle='-.', linewidth=2.5, label=f'Drone {i+1} (Customer → Rendezvous)'))
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.title(f"Truck-Drone VRP Solution - Total Distance: {truck_total_distance + drone_total_distance:.2f}", 
              fontsize=16, fontweight='bold', pad=15)
    plt.grid(alpha=0.3)
    plt.axis('equal')
    
    # Add margins around the plot
    plt.tight_layout()
    plt.margins(0.1)
    
    plt.savefig('truck_drone_vrp_solution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return truck_total_distance + drone_total_distance, truck_routes, drone_assignments

# Main program
if __name__ == "__main__":
    filename = "A-n32-k5.vrp"  
    solve_truck_drone_vrp(filename, max_time=60) 