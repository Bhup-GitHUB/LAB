mixed_dict = {
    "name": "John",
    "age": 25,
    "height": 5.9,
    "is_student": True,
    "grades": [85, 90, 78],
    "address": {"street": "123 Main St", "city": "New York"}
}

print("Original dictionary:", mixed_dict)

print("\nAccessing elements:")
print("Name:", mixed_dict["name"])
print("Age:", mixed_dict["age"])
print("Height:", mixed_dict["height"])
print("Is student:", mixed_dict["is_student"])

print("\nAccessing with get method:")
print("Name (get):", mixed_dict.get("name"))
print("Phone (get with default):", mixed_dict.get("phone", "Not available"))

print("\nAccessing nested dictionary:")
print("Street:", mixed_dict["address"]["street"])
print("City:", mixed_dict["address"]["city"])

print("\nAccessing list in dictionary:")
print("First grade:", mixed_dict["grades"][0])
print("All grades:", mixed_dict["grades"])

print("\nAdding new elements:")
mixed_dict["phone"] = "123-456-7890"
mixed_dict["email"] = "john@email.com"
print("After adding phone and email:", mixed_dict)

print("\nUpdating existing elements:")
mixed_dict["age"] = 26
print("After updating age:", mixed_dict)

print("\nRemoving elements:")
removed_value = mixed_dict.pop("height")
print("Removed height:", removed_value)
print("After removing height:", mixed_dict)

del mixed_dict["is_student"]
print("After deleting is_student:", mixed_dict)

print("\nDictionary methods:")
print("Keys:", list(mixed_dict.keys()))
print("Values:", list(mixed_dict.values()))
print("Items:", list(mixed_dict.items()))

print("\nChecking if key exists:")
print("'name' in dictionary:", "name" in mixed_dict)
print("'salary' in dictionary:", "salary" in mixed_dict)

print("\nDictionary comprehension:")
squares_dict = {x: x**2 for x in range(1, 6)}
print("Squares dictionary:", squares_dict)
