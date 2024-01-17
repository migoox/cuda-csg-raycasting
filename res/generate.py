import json
import random

def generate_json(num_elements, num_primitives, box_dimensions):
    json_data = {
        "scene": {
            "max_id": num_elements + num_primitives,
            "objects": [
                {"id": i + 1, "type": random.choice(["union", "diff", "inter"])}
                for i in range(num_elements)
            ]
        }
    }

    # Append spheres to the "objects" array with random radius and center
    for i in range(num_elements + 1, num_elements + 1 + num_primitives):
        radius = random.uniform(1.0, 1.5)  # Adjust the range based on your requirements
        center = {
            "x": random.uniform(min_x, max_x),
            "y": random.uniform(min_y, max_y),
            "z": random.uniform(min_z, max_z),
        }
        color = {
            "r": random.uniform(0.0, 1.0),
            "g": random.uniform(0.0, 1.0),
            "b": random.uniform(0.0, 1.0),
        }
        json_data["scene"]["objects"].append({
            "id": i,
            "type": "sphere",
            "radius": radius,
            "center": center,
            "color": color
        })

    return json_data

def save_to_json(json_data, filename="output.json"):
    with open(filename, "w") as json_file:
        json.dump(json_data, json_file, indent=2)
    print(f"JSON data saved to {filename}")


if __name__ == "__main__":
    tree_height = int(input("Enter the tree height: "))

    num_primitives = 2**(tree_height - 1)    
    num_elements = num_primitives - 1 
    print("primitives: " + str(num_primitives))
    print("operations: " + str(num_elements))

    # Get box dimensions from the user
    min_x = -4.0
    max_x = -2.0
    min_y = -1.0
    max_y = 1.0
    min_z = -3.0
    max_z = 1.0
    
    box_dimensions = (min_x, max_x, min_y, max_y, min_z, max_z)

    json_data = generate_json(num_elements, num_primitives, box_dimensions)
    save_to_json(json_data)