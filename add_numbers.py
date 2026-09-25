import sys


def add(a, b):
    return a + b


def main():
    if len(sys.argv) == 3:
        a, b = sys.argv[1], sys.argv[2]
    else:
        a = input("Enter the first number: ")
        b = input("Enter the second number: ")

    try:
        result = add(float(a), float(b))
    except ValueError:
        sys.exit("Error: both inputs must be numbers.")

    print(f"{a} + {b} = {result:g}")


if __name__ == "__main__":
    main()
