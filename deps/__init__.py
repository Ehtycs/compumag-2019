import sys
import os 


# add the deps directory to pythonpath
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                                '.')))

# add npyfem directory to pythonpath
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                                'npyfem')))




