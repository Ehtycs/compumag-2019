import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def open_resource(opener, filename, *args):
    try:
        return opener(filename, *args)
    except (FileNotFoundError):
        return opener("../"+filename, *args)
    
    raise FileNotFoundError("{} was not found".format())
    
import npyfem
