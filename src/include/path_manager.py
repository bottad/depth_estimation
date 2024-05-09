import os
from colorama import Fore, Style

# Functions to read/save paths from/to .txt files (files are located in the /data folder)

def save_path_to_file(path, filename):
    with open("data/" + filename + ".txt", "w") as file:
        file.write(path)
    print(Fore.GREEN + "Path saved to file." + Style.RESET_ALL)
    print(Style.RESET_ALL, end='\r')

def read_path_from_file(filename):
    try:
        with open("data/" + filename + ".txt", "r") as file:
            saved_path = file.read().strip()  # Remove any leading/trailing whitespace
            print(Fore.GREEN + "Path: " + Style.RESET_ALL + saved_path + "\n")
            return saved_path
    except FileNotFoundError:
        print(Fore.RED + "No saved path found." + Style.RESET_ALL)
        return None