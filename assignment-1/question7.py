mixed_tuple = (1, "hello", 3.14, True, [1, 2, 3], {"key": "value"})

print("Original tuple:", mixed_tuple)
print("Data types in tuple:")
for i, item in enumerate(mixed_tuple):
    print(f"Index {i}: {item} ({type(item).__name__})")

print("\nAccessing elements:")
print("First element:", mixed_tuple[0])
print("Second element:", mixed_tuple[1])
print("Last element:", mixed_tuple[-1])

print("\nSlicing:")
print("First 3 elements:", mixed_tuple[:3])
print("Elements from index 2 to end:", mixed_tuple[2:])
print("Elements from index 1 to 4:", mixed_tuple[1:5])

print("\nTuple operations:")
print("Length:", len(mixed_tuple))
print("Count of 1:", mixed_tuple.count(1))
print("Index of 'hello':", mixed_tuple.index("hello"))

print("\nTuple unpacking:")
a, b, c, d, e, f = mixed_tuple
print("Unpacked values:")
print("a =", a)
print("b =", b)
print("c =", c)

print("\nNested tuple:")
nested_tuple = ((1, 2), (3, 4), (5, 6))
print("Nested tuple:", nested_tuple)
print("First inner tuple:", nested_tuple[0])
print("First element of first inner tuple:", nested_tuple[0][0])

print("\nCreating dictionary with tuple keys:")
tuple_dict = {
    (1, 2): "point1",
    (3, 4): "point2",
    (5, 6): "point3"
}

print("Dictionary with tuple keys:", tuple_dict)
print("Value for key (1, 2):", tuple_dict[(1, 2)])
print("Value for key (3, 4):", tuple_dict[(3, 4)])

print("\nTuple as dictionary keys:")
coordinates = {
    ("x", 0): 10,
    ("y", 0): 20,
    ("x", 1): 30,
    ("y", 1): 40
}

print("Coordinates dictionary:", coordinates)
print("All keys:", list(coordinates.keys()))
print("All values:", list(coordinates.values()))

print("\nTuple comparison:")
tuple1 = (1, 2, 3)
tuple2 = (1, 2, 4)
tuple3 = (1, 2, 3)

print(f"{tuple1} == {tuple3}:", tuple1 == tuple3)
print(f"{tuple1} < {tuple2}:", tuple1 < tuple2)
print(f"{tuple1} > {tuple2}:", tuple1 > tuple2)
