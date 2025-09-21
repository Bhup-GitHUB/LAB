str1 = "Hello"
str2 = "World"

print("String 1:", str1)
print("String 2:", str2)

print("\nConcatenation:")
result = str1 + " " + str2
print("Concatenated:", result)

print("\nCapitalize:")
print("Capitalized str1:", str1.capitalize())

print("\nUppercase:")
print("Uppercase str1:", str1.upper())

print("\nRight-justify (width 20):")
print("Right-justified:", repr(str1.rjust(20)))

print("\nCenter (width 20):")
print("Centered:", repr(str1.center(20)))

print("\nReplace:")
text = "Hello Hello Hello"
print("Original:", text)
print("Replaced:", text.replace("Hello", "Hi"))

print("\nAccessing substring:")
text = "Python Programming"
print("Original string:", text)
print("Characters 0-5:", text[0:6])
print("Characters 7-18:", text[7:19])
print("Last 5 characters:", text[-5:])

print("\nString methods:")
print("Length:", len(text))
print("Find 'gram':", text.find("gram"))
print("Count 'o':", text.count("o"))
print("Starts with 'Python':", text.startswith("Python"))
print("Ends with 'ing':", text.endswith("ing"))
