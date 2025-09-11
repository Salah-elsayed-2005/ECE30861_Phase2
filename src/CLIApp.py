# src/CLIApp.py
# THIS CODE WILL HANDLE THE HIGH LEVEL LOGIC OF THE APP
import sys

from .Parser import Parser

if __name__ == "__main__":
    parse = Parser(sys.argv[1])
    print(f"Parser groups: {parse.getGroups()}")
