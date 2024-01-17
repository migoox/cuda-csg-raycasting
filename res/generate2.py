import json
import random
import math

def generate_json(num_elements, num_primitives):
    json_data = {
        "scene": {
            "max_id": num_elements + num_primitives,
            "objects": [
                {"id": fix(i + 1), "type": random.choice(["union"])}
                for i in range(num_elements)
            ]
        }
    }

    # Append spheres to the "objects" array with random radius and center
    for i in range(0, num_primitives):
        center = {
            "x": 0.866 * math.cos(2.0 * math.pi * float(i) / float(num_primitives)),
            "y": 0.866 * math.sin(2.0 * math.pi * float(i) / float(num_primitives)),
            "z": -5.5,
        }
        color = {
            "r": random.uniform(0.0, 1.0),
            "g": random.uniform(0.0, 1.0),
            "b": random.uniform(0.0, 1.0),
        }
        json_data["scene"]["objects"].append({
            "id": fix(i + num_elements + 1),
            "type": "sphere",
            "radius": 0.1,
            "center": center,
            "color": color
        })

    return json_data

def save_to_json(json_data, filename="output2.json"):
    with open(filename, "w") as json_file:
        json.dump(json_data, json_file, indent=2)
    print(f"JSON data saved to {filename}")


def get_depth(index):
    depth = 0
    while index > 0:
        index //= 2
        depth += 1
    return depth

def fix(index):
    return index + 2**get_depth(index)


if __name__ == "__main__":
    tree_height = int(input("Enter the tree height: "))

    num_primitives = 2**(tree_height - 1)    
    num_elements = num_primitives - 1 
    print("primitives: " + str(num_primitives))
    print("operations: " + str(num_elements))


    json_data = generate_json(num_elements, num_primitives)
    save_to_json(json_data)