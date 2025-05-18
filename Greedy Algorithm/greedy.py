import math
import matplotlib.pyplot as plt

# Parse VRP file
def parse_vrp_file(filename):
    coordinates = {}
    demands = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
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
    
    return coordinates, demands

# Calculate distance between two points
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Simple Greedy Truck-Drone VRP
def simple_greedy_truck_drone_vrp(filename):
    coordinates, demands = parse_vrp_file(filename)
    
    # Problem parameters
    depot = 1
    truck_capacity = 100
    drone_capacity = 15  # Increased to match more customers
    max_drone_customers = 3  # Each truck can serve 3 drone customers
    
    # Get customers
    customers = [node for node in coordinates.keys() if node != depot]
    
    # Sort customers by demand (smallest first)
    customers_by_demand = sorted(customers, key=lambda c: demands[c])
    
    # Assign customers to trucks or drones
    truck_customers = []
    drone_customers = []
    
    for customer in customers_by_demand:
        if demands[customer] <= drone_capacity and len(drone_customers) < max_drone_customers * 3:
            drone_customers.append(customer)
        else:
            truck_customers.append(customer)
    
    unserved = truck_customers.copy()
    
    # Greedy truck routing
    total_distance = 0
    truck_routes = []
    
    while unserved:
        # New truck route
        current_load = 0
        route = [depot]
        current_location = depot
        
        # While truck has capacity and customers remain
        while unserved and current_load < truck_capacity:
            # Find nearest unserved customer
            nearest = None
            min_dist = float('inf')
            
            for customer in unserved:
                if demands[customer] + current_load <= truck_capacity:
                    dist = distance(coordinates[current_location], coordinates[customer])
                    if dist < min_dist:
                        min_dist = dist
                        nearest = customer
            
            if nearest is None:
                break
                
            # Add to route
            route.append(nearest)
            unserved.remove(nearest)
            current_load += demands[nearest]
            current_location = nearest
            total_distance += min_dist
        
        # Return to depot
        route.append(depot)
        total_distance += distance(coordinates[current_location], coordinates[depot])
        truck_routes.append((route, current_load))
    
    # Greedy drone assignment
    drone_assignments = []
    remaining_drones = drone_customers.copy()
    
    for i, (truck_route, _) in enumerate(truck_routes):
        # Each truck gets assigned up to max_drone_customers drones
        assigned_drones = []
        
        for _ in range(min(max_drone_customers, len(remaining_drones))):
            if not remaining_drones:
                break
                
            # Find nearest drone customer to any point on truck route
            best_customer = None
            min_total_dist = float('inf')
            best_launch_point = None
            
            for drone_customer in remaining_drones:
                for route_point in truck_route:
                    dist = distance(coordinates[route_point], coordinates[drone_customer])
                    if dist < min_total_dist:
                        min_total_dist = dist
                        best_customer = drone_customer
                        best_launch_point = route_point
            
            if best_customer:
                assigned_drones.append((best_customer, best_launch_point, min_total_dist))
                remaining_drones.remove(best_customer)
                total_distance += 2 * min_total_dist  # Round trip
        
        drone_assignments.append(assigned_drones)
    
    # Print solution
    print(f"Total distance: {total_distance:.2f}")
    print(f"Number of trucks used: {len(truck_routes)}")
    print(f"Drone customers: {len(drone_customers)}")
    
    print("\nTruck Routes:")
    for i, (route, load) in enumerate(truck_routes):
        route_distance = sum(distance(coordinates[route[j]], coordinates[route[j+1]]) 
                           for j in range(len(route)-1))
        print(f"Truck {i+1}: {route} - Load: {load}/{truck_capacity} - Distance: {route_distance:.2f}")
    
    print("\nDrone Assignments:")
    for i, drones in enumerate(drone_assignments):
        print(f"Truck {i+1} drones:")
        if not drones:
            print("  No drone assignments")
        for customer, launch_point, dist in drones:
            print(f"  Drone to customer {customer} from {launch_point} - Distance: {dist:.2f}")
    
    # Visualization
    plt.figure(figsize=(14, 10))
    
    # Plot nodes
    for node, (x, y) in coordinates.items():
        if node == depot:
            plt.scatter(x, y, s=250, c='red', marker='s', edgecolor='black', zorder=100)
            plt.text(x+1, y+1, "Depot", fontsize=12, fontweight='bold')
        elif node in drone_customers:
            plt.scatter(x, y, s=150, c='lime', alpha=0.9, edgecolor='black', zorder=90)
            plt.text(x+1, y+1, str(node), fontsize=10, fontweight='bold')
        else:
            plt.scatter(x, y, s=150, c='lightblue', alpha=0.9, edgecolor='black', zorder=90)
            plt.text(x+1, y+1, str(node), fontsize=10)
    
    # Plot truck routes
    truck_colors = ['orange', 'green', 'purple', 'brown', 'pink']
    for i, (route, _) in enumerate(truck_routes):
        color = truck_colors[i % len(truck_colors)]
        for j in range(len(route)-1):
            x1, y1 = coordinates[route[j]]
            x2, y2 = coordinates[route[j+1]]
            plt.plot([x1, x2], [y1, y2], c=color, linewidth=2.5, alpha=0.7, zorder=80)
    
    # Plot drone routes (MORE VISIBLE)
    drone_colors = ['magenta', 'darkviolet', 'cyan']
    for i, drones in enumerate(drone_assignments):
        if drones:  # Only plot if there are drones
            color = drone_colors[i % len(drone_colors)]
            for customer, launch_point, _ in drones:
                x1, y1 = coordinates[launch_point]
                x2, y2 = coordinates[customer]
                # Draw outbound (thick solid line)
                plt.plot([x1, x2], [y1, y2], c=color, linestyle='-', linewidth=3, alpha=0.9, zorder=95)
                # Draw return (thick dashed line)
                plt.plot([x2, x1], [y2, y1], c=color, linestyle='--', linewidth=3, alpha=0.9, zorder=95)
                
                # Add arrows
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                dx, dy = (x2 - x1) * 0.1, (y2 - y1) * 0.1
                plt.arrow(mid_x, mid_y, dx, dy, head_width=1.5, head_length=1.5, 
                         fc=color, ec=color, alpha=0.9, zorder=96)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=15, label='Depot'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=15, label='Truck Customer'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=15, label='Drone Customer'),
        plt.Line2D([0], [0], color='black', linewidth=2.5, label='Truck Route'),
        plt.Line2D([0], [0], color='magenta', linewidth=3, label='Drone Route (Round Trip)')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    plt.title(f"Simple Greedy Truck-Drone VRP - Total Distance: {total_distance:.2f}", 
              fontsize=16, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.margins(0.1)
    plt.savefig('simple_greedy_truck_drone.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run the algorithm
if __name__ == "__main__":
    simple_greedy_truck_drone_vrp("B-n31-k5.vrp")