import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import gurobipy as gp
from gurobipy import GRB
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import os

class TruckDroneVRPGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Truck-Drone VRP Optimizer")
        self.root.geometry("1400x900")
        
        # Variables
        self.coordinates = {}
        self.demands = {}
        self.capacity = 0
        self.filename = ""
        
        # Create the interface
        self.create_interface()
        
    def create_interface(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel for controls (COMPACT)
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # SOLVE BUTTON AT TOP!
        solve_container = ttk.LabelFrame(left_panel, text="🚀 OPTIMIZATION", padding="8")
        solve_container.pack(fill='x', pady=(0, 8))
        
        self.solve_button = tk.Button(solve_container, 
                                     text="START SOLVING", 
                                     command=self.solve_vrp,
                                     font=('Arial', 14, 'bold'),
                                     bg='#4CAF50',
                                     fg='white',
                                     height=2)
        self.solve_button.pack(fill='x', pady=3)
        
        self.solve_status = tk.Label(solve_container, text="Ready!", 
                                   font=('Arial', 9), fg='green')
        self.solve_status.pack()
        
        # File selection
        file_frame = ttk.LabelFrame(left_panel, text="File", padding="5")
        file_frame.pack(fill='x', pady=(0, 5))
        
        self.file_label = ttk.Label(file_frame, text="No file selected", font=('Arial', 8))
        self.file_label.pack(pady=(0, 2))
        
        file_buttons = ttk.Frame(file_frame)
        file_buttons.pack(fill='x')
        ttk.Button(file_buttons, text="📁 Select", command=self.select_file).pack(side='left', fill='x', expand=True, padx=(0,1))
        ttk.Button(file_buttons, text="📂 Example", command=self.load_example).pack(side='left', fill='x', expand=True, padx=(1,0))
        
        # Vehicles
        vehicle_frame = ttk.LabelFrame(left_panel, text="Vehicles", padding="5")
        vehicle_frame.pack(fill='x', pady=(0, 5))
        
        # Trucks
        truck_frame = ttk.Frame(vehicle_frame)
        truck_frame.pack(fill='x')
        ttk.Label(truck_frame, text="Trucks:", font=('Arial', 8)).pack(side='left')
        self.num_trucks = tk.IntVar(value=4)
        truck_scale = ttk.Scale(truck_frame, from_=1, to=20, variable=self.num_trucks, orient='horizontal', length=120)
        truck_scale.pack(side='left', fill='x', expand=True, padx=5)
        self.truck_label = ttk.Label(truck_frame, text="4", font=('Arial', 8), width=2)
        self.truck_label.pack(side='right')
        truck_scale.configure(command=lambda v: self.truck_label.config(text=str(int(float(v)))))
        
        # Drones
        drone_frame = ttk.Frame(vehicle_frame)
        drone_frame.pack(fill='x')
        ttk.Label(drone_frame, text="Drones:", font=('Arial', 8)).pack(side='left')
        self.num_drones = tk.IntVar(value=3)
        drone_scale = ttk.Scale(drone_frame, from_=1, to=20, variable=self.num_drones, orient='horizontal', length=120)
        drone_scale.pack(side='left', fill='x', expand=True, padx=5)
        self.drone_label = ttk.Label(drone_frame, text="3", font=('Arial', 8), width=2)
        self.drone_label.pack(side='right')
        drone_scale.configure(command=lambda v: self.drone_label.config(text=str(int(float(v)))))
        
        # Capacity
        capacity_frame = ttk.LabelFrame(left_panel, text="Capacity", padding="5")
        capacity_frame.pack(fill='x', pady=(0, 5))
        
        # Truck capacity
        tcap_frame = ttk.Frame(capacity_frame)
        tcap_frame.pack(fill='x')
        ttk.Label(tcap_frame, text="Truck ×:", font=('Arial', 8)).pack(side='left')
        self.truck_capacity_mult = tk.DoubleVar(value=1.0)
        tcap_scale = ttk.Scale(tcap_frame, from_=0.5, to=2.0, variable=self.truck_capacity_mult, orient='horizontal', length=100)
        tcap_scale.pack(side='left', fill='x', expand=True, padx=5)
        self.truck_cap_label = ttk.Label(tcap_frame, text="1.0x", font=('Arial', 8), width=4)
        self.truck_cap_label.pack(side='right')
        tcap_scale.configure(command=lambda v: self.truck_cap_label.config(text=f"{float(v):.1f}x"))
        
        # Drone capacity
        dcap_frame = ttk.Frame(capacity_frame)
        dcap_frame.pack(fill='x')
        ttk.Label(dcap_frame, text="Drone:", font=('Arial', 8)).pack(side='left')
        self.drone_capacity = tk.IntVar(value=150)
        dcap_scale = ttk.Scale(dcap_frame, from_=50, to=300, variable=self.drone_capacity, orient='horizontal', length=100)
        dcap_scale.pack(side='left', fill='x', expand=True, padx=5)
        self.drone_cap_label = ttk.Label(dcap_frame, text="150", font=('Arial', 8), width=4)
        self.drone_cap_label.pack(side='right')
        dcap_scale.configure(command=lambda v: self.drone_cap_label.config(text=str(int(float(v)))))
        
        # Range
        range_frame = ttk.LabelFrame(left_panel, text="Range", padding="5")
        range_frame.pack(fill='x', pady=(0, 5))
        
        range_config = ttk.Frame(range_frame)
        range_config.pack(fill='x')
        ttk.Label(range_config, text="Drone km:", font=('Arial', 8)).pack(side='left')
        self.max_drone_range = tk.IntVar(value=100)
        range_scale = ttk.Scale(range_config, from_=50, to=200, variable=self.max_drone_range, orient='horizontal', length=100)
        range_scale.pack(side='left', fill='x', expand=True, padx=5)
        self.range_label = ttk.Label(range_config, text="100", font=('Arial', 8), width=4)
        self.range_label.pack(side='right')
        range_scale.configure(command=lambda v: self.range_label.config(text=str(int(float(v)))))
        
        # Algorithm
        algo_frame = ttk.LabelFrame(left_panel, text="Algorithm", padding="5")
        algo_frame.pack(fill='x', pady=(0, 5))
        
        # Time limit
        time_config = ttk.Frame(algo_frame)
        time_config.pack(fill='x')
        ttk.Label(time_config, text="Time sec:", font=('Arial', 8)).pack(side='left')
        self.time_limit = tk.IntVar(value=300)
        time_scale = ttk.Scale(time_config, from_=60, to=600, variable=self.time_limit, orient='horizontal', length=100)
        time_scale.pack(side='left', fill='x', expand=True, padx=5)
        self.time_label = ttk.Label(time_config, text="300", font=('Arial', 8), width=4)
        self.time_label.pack(side='right')
        time_scale.configure(command=lambda v: self.time_label.config(text=str(int(float(v)))))
        
        # Max customers per drone
        cust_config = ttk.Frame(algo_frame)
        cust_config.pack(fill='x')
        ttk.Label(cust_config, text="Max/drone:", font=('Arial', 8)).pack(side='left')
        self.max_customers_per_drone = tk.IntVar(value=3)
        cust_scale = ttk.Scale(cust_config, from_=1, to=5, variable=self.max_customers_per_drone, orient='horizontal', length=100)
        cust_scale.pack(side='left', fill='x', expand=True, padx=5)
        self.cust_label = ttk.Label(cust_config, text="3", font=('Arial', 8), width=4)
        self.cust_label.pack(side='right')
        cust_scale.configure(command=lambda v: self.cust_label.config(text=str(int(float(v)))))
        
        # Display options
        vis_frame = ttk.LabelFrame(left_panel, text="Display", padding="5")
        vis_frame.pack(fill='x', pady=(0, 5))
        
        self.show_arrows = tk.BooleanVar(value=True)
        ttk.Checkbutton(vis_frame, text="Arrows", variable=self.show_arrows).pack(side='left')
        
        self.show_labels = tk.BooleanVar(value=True)
        ttk.Checkbutton(vis_frame, text="Labels", variable=self.show_labels).pack(side='left')
        
        self.save_plot = tk.BooleanVar(value=False)
        ttk.Checkbutton(vis_frame, text="Save PNG", variable=self.save_plot).pack(side='left')
        
        ttk.Button(vis_frame, text="🗑️ Clear", command=self.clear_results).pack(side='right')
        
        # Status
        status_frame = ttk.LabelFrame(left_panel, text="Status", padding="5")
        status_frame.pack(fill='both', expand=True, pady=(5, 0))
        
        self.status_text = tk.Text(status_frame, height=6, width=35, wrap='word', font=('Arial', 8))
        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        self.status_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Right panel for visualization
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side='right', fill='both', expand=True)
        
        self.vis_frame = ttk.LabelFrame(right_panel, text="Solution Visualization", padding="10")
        self.vis_frame.pack(fill='both', expand=True)
        
        self.canvas = None
        self.root.after(100, self.show_welcome_message)
        
    def show_welcome_message(self):
        """Show welcome message in visualization area"""
        for widget in self.vis_frame.winfo_children():
            widget.destroy()
            
        welcome_frame = tk.Frame(self.vis_frame, bg='white')
        welcome_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        title_label = tk.Label(welcome_frame, text="🚛 + 🚁 TRUCK-DRONE VRP OPTIMIZER", 
                              font=('Arial', 16, 'bold'), bg='white', fg='navy')
        title_label.pack(pady=(20, 10))
        
        instructions = [
            "Welcome! To get started:",
            "",
            "1.  Select a VRP file (A-n32-k5.vrp, B-n31-k5.vrp, etc.)",
            "   OR click 'Example' for a quick demo",
            "",
            "2.  Adjust parameters using the sliders on the left",
            "",
            "3.  Click 'START SOLVING' to begin optimization",
            "",
            "4.  View results here with interactive visualization",
            "",
            "Ready to optimize your delivery routes?"
        ]
        
        for instruction in instructions:
            label = tk.Label(welcome_frame, text=instruction, 
                           font=('Arial', 11), bg='white', justify='left')
            label.pack(pady=2, anchor='w', padx=50)
        
        if hasattr(self, 'filename') and self.filename:
            status_label = tk.Label(welcome_frame, 
                                  text=f"✅ File loaded: {os.path.basename(self.filename)}",
                                  font=('Arial', 12, 'bold'), bg='white', fg='green')
            status_label.pack(pady=(20, 10))
            
            start_label = tk.Label(welcome_frame, 
                                 text="Ready to solve! Click 'START SOLVING' button.",
                                 font=('Arial', 12, 'italic'), bg='white', fg='blue')
            start_label.pack(pady=5)
        
        self.welcome_frame = welcome_frame
    
    def load_example(self):
        example_files = []
        for file in os.listdir('.'):
            if file.endswith('.vrp'):
                example_files.append(file)
        
        if example_files:
            example_file = example_files[0]
            self.filename = example_file
            self.file_label.config(text=os.path.basename(example_file))
            
            try:
                self.coordinates, self.demands, self.capacity = self.parse_vrp_file(example_file)
                if hasattr(self, 'status_text'):
                    self.status_text.insert('end', f"Loaded example: {example_file}\n")
                    self.status_text.insert('end', f"Example loaded: {len(self.coordinates)} nodes, capacity: {self.capacity}\n")
                    self.status_text.insert('end', "Ready to solve! Click 'START SOLVING' to start.\n")
                    self.status_text.see('end')
                self.show_welcome_message()
            except Exception as e:
                if hasattr(self, 'status_text'):
                    self.status_text.insert('end', f"Error loading example: {str(e)}\n")
                    self.status_text.see('end')
        else:
            if hasattr(self, 'status_text'):
                self.status_text.insert('end', "No .vrp files found in current directory.\n")
                self.status_text.insert('end', "Please use 'Select' to choose a file.\n")
                self.status_text.see('end')

    def log_status(self, message):
        self.status_text.insert('end', f"{message}\n")
        self.status_text.see('end')
        self.root.update()
    
    def clear_results(self):
        self.status_text.delete(1.0, 'end')
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        for widget in self.vis_frame.winfo_children():
            widget.destroy()
        self.show_welcome_message()
        if hasattr(self, 'status_text'):
            self.status_text.insert('end', "Results cleared. Ready for new optimization.\n")
            self.status_text.see('end')
    
    def select_file(self):
        filename = filedialog.askopenfilename(
            title="Select VRP File",
            filetypes=[("VRP files", "*.vrp"), ("All files", "*.*")]
        )
        
        if filename:
            self.filename = filename
            self.file_label.config(text=os.path.basename(filename))
            print(f"Selected file: {os.path.basename(filename)}")
            
            try:
                self.coordinates, self.demands, self.capacity = self.parse_vrp_file(filename)
                print(f"Loaded {len(self.coordinates)} nodes, capacity: {self.capacity}")
                if hasattr(self, 'status_text'):
                    self.status_text.insert('end', f"Selected file: {os.path.basename(filename)}\n")
                    self.status_text.insert('end', f"Loaded {len(self.coordinates)} nodes, capacity: {self.capacity}\n")
                    self.status_text.see('end')
                
                self.show_welcome_message()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error reading file: {str(e)}")
    
    def parse_vrp_file(self, filename):
        coordinates = {}
        demands = {}
        capacity = 0
        
        with open(filename, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                if line.strip().startswith('CAPACITY'):
                    capacity = int(line.strip().split(':')[1].strip())
                    break
            
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
    
    def calculate_distances(self, coordinates):
        nodes = list(coordinates.keys())
        distances = {}
        
        for i in nodes:
            for j in nodes:
                if i != j:
                    x1, y1 = coordinates[i]
                    x2, y2 = coordinates[j]
                    distances[(i, j)] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        return distances
    
    def adjust_coordinates(self, coordinates):
        adjusted = {}
        nodes = list(coordinates.keys())
        
        min_x = min(coordinates[node][0] for node in nodes)
        max_x = max(coordinates[node][0] for node in nodes)
        min_y = min(coordinates[node][1] for node in nodes)
        max_y = max(coordinates[node][1] for node in nodes)
        
        x_range = max_x - min_x
        y_range = max_y - min_y
        
        scale_factor = 1.2
        
        for node in nodes:
            x, y = coordinates[node]
            
            norm_x = (x - min_x) / x_range if x_range > 0 else 0
            norm_y = (y - min_y) / y_range if y_range > 0 else 0
            
            adjusted_x = min_x + norm_x * x_range * scale_factor
            adjusted_y = min_y + norm_y * y_range * scale_factor
            
            for other_node in nodes:
                if other_node != node:
                    other_x, other_y = coordinates[other_node]
                    if abs(x - other_x) < 2 and abs(y - other_y) < 2:
                        adjusted_x += np.random.uniform(-1.5, 1.5)
                        adjusted_y += np.random.uniform(-1.5, 1.5)
            
            adjusted[node] = (adjusted_x, adjusted_y)
        
        return adjusted
    
    def solve_vrp(self):
        if not self.filename:
            messagebox.showwarning("Warning", "Please select a VRP file first!")
            return
        
        self.solve_button.config(text="⏳ SOLVING...", state='disabled', bg='orange')
        self.solve_status.config(text="Optimization in progress...", fg='orange')
        self.root.update()
        
        try:
            num_trucks = self.num_trucks.get()
            num_drones = self.num_drones.get()
            truck_capacity = int(self.capacity * self.truck_capacity_mult.get())
            drone_capacity = self.drone_capacity.get()
            max_drone_range = self.max_drone_range.get()
            max_time = self.time_limit.get()
            max_customers_per_drone = self.max_customers_per_drone.get()
            
            self.log_status(f"Configuration:")
            self.log_status(f"- Trucks: {num_trucks}, Drones: {num_drones}")
            self.log_status(f"- Truck capacity: {truck_capacity}")
            self.log_status(f"- Drone capacity: {drone_capacity}")
            self.log_status(f"- Drone range: {max_drone_range}")
            
            start_time = time.time()
            
            distances = self.calculate_distances(self.coordinates)
            
            nodes = list(self.coordinates.keys())
            depot = 1
            customers = [node for node in nodes if node != depot]
            
            customer_demands = [(c, self.demands[c]) for c in customers]
            sorted_customers = sorted(customer_demands, key=lambda x: x[1])
            
            drone_candidates = []
            for c, d in sorted_customers:
                if d <= drone_capacity and len(drone_candidates) < num_drones * max_customers_per_drone:
                    drone_candidates.append(c)
            
            truck_customers = [c for c in customers if c not in drone_candidates]
            
            self.log_status(f"Drone candidates: {len(drone_candidates)}")
            self.log_status(f"Truck customers: {len(truck_customers)}")
            
            truck_clusters = [[] for _ in range(num_trucks)]
            truck_loads = [0] * num_trucks
            
            sorted_truck_customers = sorted([(c, self.demands[c]) for c in truck_customers], 
                                           key=lambda x: x[1], reverse=True)
            
            for customer, demand in sorted_truck_customers:
                min_load_truck = min(range(num_trucks), key=lambda t: truck_loads[t])
                
                if truck_loads[min_load_truck] + demand <= truck_capacity:
                    truck_clusters[min_load_truck].append(customer)
                    truck_loads[min_load_truck] += demand
                else:
                    for t in range(num_trucks):
                        if truck_loads[t] + demand <= truck_capacity:
                            truck_clusters[t].append(customer)
                            truck_loads[t] += demand
                            break
                    else:
                        truck_clusters[min_load_truck].append(customer)
                        truck_loads[min_load_truck] += demand
            
            truck_routes = []
            truck_total_distance = 0
            
            self.log_status("Solving truck routes...")
            
            for t in range(num_trucks):
                if not truck_clusters[t]:
                    continue
                
                self.log_status(f"Solving TSP for truck {t+1}...")
                
                cluster_nodes = [depot] + truck_clusters[t]
                
                tsp_model = gp.Model(f"TSP_Truck_{t}")
                tsp_model.setParam('OutputFlag', 0)
                
                x = {}
                for i in cluster_nodes:
                    for j in cluster_nodes:
                        if i != j:
                            x[i, j] = tsp_model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
                
                u = {}
                for i in cluster_nodes:
                    if i != depot:
                        u[i] = tsp_model.addVar(lb=0, ub=len(cluster_nodes)-1, name=f'u_{i}')
                
                tsp_model.setObjective(
                    gp.quicksum(distances[i, j] * x[i, j] for i in cluster_nodes for j in cluster_nodes if i != j),
                    GRB.MINIMIZE
                )
                
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
                
                tsp_model.addConstr(
                    gp.quicksum(x[depot, j] for j in cluster_nodes if j != depot) == 1,
                    name="depot_out"
                )
                tsp_model.addConstr(
                    gp.quicksum(x[i, depot] for i in cluster_nodes if i != depot) == 1,
                    name="depot_in"
                )
                
                for i in cluster_nodes:
                    if i != depot:
                        for j in cluster_nodes:
                            if j != depot and i != j:
                                tsp_model.addConstr(
                                    u[i] - u[j] + len(cluster_nodes) * x[i, j] <= len(cluster_nodes) - 1,
                                    name=f"mtz_{i}_{j}"
                                )
                
                remaining_time = max(1, max_time - (time.time() - start_time))
                tsp_model.setParam('TimeLimit', remaining_time / num_trucks)
                
                tsp_model.optimize()
                
                if tsp_model.Status == GRB.OPTIMAL or tsp_model.Status == GRB.TIME_LIMIT:
                    route = [depot]
                    current = depot
                    
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
                            
                    if route[-1] != depot:
                        route.append(depot)
                    
                    route_demand = sum(self.demands.get(node, 0) for node in route if node != depot)
                    route_distance = sum(distances.get((route[i], route[i+1]), 0) for i in range(len(route)-1))
                    truck_total_distance += route_distance
                    
                    truck_routes.append((t, route, route_demand, route_distance))
            
            self.log_status("Assigning drones...")
            
            drone_assignments = []
            drone_total_distance = 0
            
            for d in range(num_drones):
                drone_load = 0
                drone_customers = []
                
                for c in drone_candidates[:]:
                    if drone_load + self.demands[c] <= drone_capacity:
                        best_dist = float('inf')
                        best_launch = None
                        best_launch_truck = None
                        best_rendezvous = None
                        best_rendezvous_truck = None
                        
                        for launch_truck, launch_route, _, _ in truck_routes:
                            for launch_idx, launch_node in enumerate(launch_route):
                                if launch_node == depot:
                                    continue
                                    
                                launch_dist = distances.get((launch_node, c), float('inf'))
                                
                                if launch_dist <= max_drone_range / 2:
                                    for rendez_truck, rendez_route, _, _ in truck_routes:
                                        for rendez_idx, rendez_node in enumerate(rendez_route):
                                            if rendez_node == depot:
                                                continue
                                                
                                            rendez_dist = distances.get((c, rendez_node), float('inf'))
                                            
                                            if rendez_dist <= max_drone_range / 2:
                                                if launch_truck == rendez_truck and rendez_idx <= launch_idx:
                                                    continue
                                                    
                                                total_trip = launch_dist + rendez_dist
                                                
                                                if total_trip < best_dist:
                                                    best_dist = total_trip
                                                    best_launch = launch_node
                                                    best_launch_truck = launch_truck
                                                    best_rendezvous = rendez_node
                                                    best_rendezvous_truck = rendez_truck
                        
                        if best_launch is not None and best_rendezvous is not None:
                            drone_customers.append((c, best_launch_truck, best_launch, 
                                                  best_rendezvous_truck, best_rendezvous, best_dist))
                            drone_load += self.demands[c]
                            drone_candidates.remove(c)
                            drone_total_distance += best_dist
                            
                            if len(drone_customers) >= max_customers_per_drone:
                                break
                
                drone_assignments.append((d, drone_customers, drone_load))
            
            self.log_status(f"\nSolution completed in {time.time() - start_time:.2f} seconds")
            self.log_status(f"Total distance: {truck_total_distance + drone_total_distance:.2f}")
            
            self.log_status("\nTruck Routes:")
            for t, route, demand, distance in truck_routes:
                self.log_status(f"Truck {t+1}: {route}")
                self.log_status(f"  Demand: {demand}/{truck_capacity}, Distance: {distance:.2f}")
            
            self.log_status("\nDrone Assignments:")
            for d, customers, load in drone_assignments:
                self.log_status(f"Drone {d+1} - Load: {load}/{drone_capacity}")
                for c, lt, ln, rt, rn, dist in customers:
                    self.log_status(f"  Customer {c}: Truck {lt+1}→{ln} to Truck {rt+1}→{rn}, Dist: {dist:.2f}")
            
            self.visualize_solution(truck_routes, drone_assignments, 
                                  truck_total_distance + drone_total_distance)
            
        except Exception as e:
            if hasattr(self, 'status_text'):
                self.status_text.insert('end', f"Error: {str(e)}\n")
                self.status_text.see('end')
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.solve_button.config(text="START SOLVING", state='normal', bg='#4CAF50')
            self.solve_status.config(text="Ready!", fg='green')
    
    def visualize_solution(self, truck_routes, drone_assignments, total_distance):
        for widget in self.vis_frame.winfo_children():
            widget.destroy()
        
        self.canvas = None
        
        vis_coordinates = self.adjust_coordinates(self.coordinates)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        depot = 1
        for node, (x, y) in vis_coordinates.items():
            if node == depot:
                ax.scatter(x, y, s=250, c='red', marker='s', edgecolor='black', zorder=100)
                if self.show_labels.get():
                    ax.text(x+1.5, y+1.5, f"Depot", fontsize=12, fontweight='bold')
            else:
                is_drone_customer = any(c[0] == node for _, customers, _ in drone_assignments for c in customers)
                if is_drone_customer:
                    ax.scatter(x, y, s=120, c='lime', alpha=0.8, edgecolor='black', zorder=90)
                else:
                    ax.scatter(x, y, s=120, c='royalblue', alpha=0.8, edgecolor='black', zorder=90)
                
                if self.show_labels.get():
                    ax.text(x+1.0, y+1.0, f"{node}", fontsize=10, fontweight='bold')
        
        truck_colors = ['darkorange', 'darkblue', 'darkgreen', 'brown', 'purple', 'red', 'cyan', 'magenta']
        
        for i, (t, route, _, _) in enumerate(truck_routes):
            color = truck_colors[i % len(truck_colors)]
            for j in range(len(route)-1):
                node1, node2 = route[j], route[j+1]
                x1, y1 = vis_coordinates[node1]
                x2, y2 = vis_coordinates[node2]
                ax.plot([x1, x2], [y1, y2], c=color, linewidth=2.5, alpha=0.8, zorder=80)
                
                if self.show_arrows.get():
                    arrow_x = x1 + 0.6*(x2-x1)
                    arrow_y = y1 + 0.6*(y2-y1)
                    dx = (x2-x1) * 0.1
                    dy = (y2-y1) * 0.1
                    ax.arrow(arrow_x, arrow_y, dx, dy, head_width=1.5, head_length=1.5, 
                            fc=color, ec=color, zorder=85)
        
        drone_colors = ['magenta', 'darkviolet', 'cyan', 'orange', 'lime', 'pink', 'yellow', 'gray']
        
        for i, (d, customers, _) in enumerate(drone_assignments):
            color = drone_colors[i % len(drone_colors)]
            for c, launch_truck, launch_node, rendez_truck, rendez_node, _ in customers:
                launch_x, launch_y = vis_coordinates[launch_node]
                cust_x, cust_y = vis_coordinates[c]
                rendez_x, rendez_y = vis_coordinates[rendez_node]
                
                ax.plot([launch_x, cust_x], [launch_y, cust_y], c=color, linestyle='--', 
                       linewidth=2, alpha=0.9, zorder=85)
                
                ax.plot([cust_x, rendez_x], [cust_y, rendez_y], c=color, linestyle='-.', 
                       linewidth=2, alpha=0.9, zorder=85)
                
                if self.show_arrows.get():
                    arrow_x = launch_x + 0.6*(cust_x-launch_x)
                    arrow_y = launch_y + 0.6*(cust_y-launch_y)
                    dx = (cust_x-launch_x) * 0.1
                    dy = (cust_y-launch_y) * 0.1
                    ax.arrow(arrow_x, arrow_y, dx, dy, head_width=1.5, head_length=1.5, 
                            fc=color, ec=color, zorder=85)
        
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
            if drone_assignments[i][1]:
                legend_elements.append(plt.Line2D([0], [0], color=drone_colors[i % len(drone_colors)], 
                                                 linestyle='--', linewidth=2.5, label=f'Drone {i+1}'))
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        ax.set_title(f"Truck-Drone VRP Solution - Total Distance: {total_distance:.2f}", 
                    fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.axis('equal')
        
        self.canvas = FigureCanvasTkAgg(fig, self.vis_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        if self.save_plot.get():
            fig.savefig('truck_drone_vrp_solution.png', dpi=300, bbox_inches='tight')
            self.log_status("Plot saved as 'truck_drone_vrp_solution.png'")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = TruckDroneVRPGUI(root)
    root.mainloop()