"""
Testing Module

Simple test script to verify email file reading functionality.
"""

def main():
    """Read and display contents of a test email file."""
    try:
        with open("./email/email.txt", "r", encoding="utf-8") as f:
            contents = f.read()
            print(contents)
    except FileNotFoundError:
        print("Error: Test email file not found")
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    main()