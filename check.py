import pandas as pd
import numpy as np
from simanneal import Annealer
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
import urllib.parse

# Load CSV data
data = pd.read_csv('locations_updated.txt')

# Extract location names, latitudes, and longitudes
product_ID = data['Product_ID'].tolist()
customer_Name = data['Customer_Name'].tolist()
Phone_No = data['Phone_No'].tolist()
location_names = data['Address'].tolist()
latitudes = data['Latitude'].tolist()
longitudes = data['Longitude'].tolist()
# Number of locations
n_locations = len(location_names)

# Define a function to calculate the Haversine distance between two points
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lat1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# Calculate distance matrix
distance_matrix = np.zeros((n_locations, n_locations))
for i in range(n_locations):
    for j in range(n_locations):
        if i != j:
            distance_matrix[i, j] = haversine_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])

# Function to set the start node
def set_start_node(route, start_node):
    """Set the start node of the route and move it to the beginning."""
    if start_node != 0:
        route.remove(start_node)
        route.insert(0, start_node)
    return route

# Define a function to calculate the total distance for a given route
def calculate_distance(route):
    distance = 0
    for i in range(len(route) - 1):
        distance += distance_matrix[route[i], route[i + 1]]
    distance += distance_matrix[route[-1], route[0]]
    return distance

# Define a custom simulated annealing class for TSP
class TSPAnnealer(Annealer):
    def __init__(self, initial_route):
        self.route = initial_route
        super().__init__(initial_route)

    def move(self):
        # Swap two cities in the route
        i = np.random.randint(1, len(self.route) - 1)
        j = np.random.randint(1, len(self.route) - 1)
        self.route[i], self.route[j] = self.route[j], self.route[i]

    def energy(self):
        # Return the total distance of the current route
        return calculate_distance(self.route)

# Set the initial route
initial_route = list(range(n_locations))

# Specify the start node
start_node = 0

# Update the initial route to start from the specified start node
initial_route = set_start_node(initial_route, start_node)

# Create an instance of the TSPAnnealer with the initial route
annealer = TSPAnnealer(initial_route)

# Run the simulated annealing algorithm
annealer.auto(minutes=5, steps=1000)

# Get the optimized route and distance
optimized_route = annealer.route
optimized_distance = calculate_distance(optimized_route)

# Define a function to generate a Google Maps URL for the optimized route
def generate_google_maps_url(optimized_route, location_names, latitudes, longitudes):
    base_url = "https://www.google.com/maps/dir/?api=1"
    travel_mode = "driving"
    waypoints = []

    # Start point
    start_point = f"{latitudes[optimized_route[0]]},{longitudes[optimized_route[0]]}"

    # Collect waypoints
    for idx in optimized_route:
        waypoint = f"{latitudes[idx]},{longitudes[idx]}"
        waypoints.append(waypoint)

    # Join the waypoints using the pipe ('|') delimiter
    waypoints_str = '|'.join(waypoints)

    # Create the URL
    url = f"{base_url}&origin={start_point}&destination={start_point}&waypoints={urllib.parse.quote(waypoints_str)}&travelmode={travel_mode}"

    return url

# Generate the Google Maps URL
google_maps_url = generate_google_maps_url(optimized_route, location_names, latitudes, longitudes)

# Print the optimized route and distance
print("Optimized Route:")
for idx, route_idx in enumerate(optimized_route):
    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Create the sequence number starting from 1 and pad with zeros for a 3-digit number (e.g., 001)
    sequence_number = f"{idx + 1:03d}"

    # Combine the current date with the sequence number
    current_date_sequence_number = f"{current_date}-{sequence_number}"

    # Print the desired output with the current date-sequence number, customer name, phone number, and location name
    print(f"{current_date_sequence_number} : {customer_Name[route_idx]} : {Phone_No[route_idx]} : {location_names[route_idx]}")

print("Total Distance:", optimized_distance)


# Print the Google Maps URL
print("\nGoogle Maps URL:")
print(google_maps_url)
