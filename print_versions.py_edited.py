# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import platform  # Needed to print python version
import numpy as np
import scipy
import pandas 
import sklearn 
import matplotlib
import seaborn
import sys

#def main():
# Prints python version
sys.stdout = open("out_print_versions.txt","w+") 

print('package'," | ","version")
print("=======================\n")

print('python',"  ",platform.python_version())

# Print version numbers of modules
print(np.__name__, "   ",np.__version__)
print(scipy.__name__,"   ",scipy.__version__)
print(pandas.__name__,"  ",pandas.__version__)
print(sklearn .__name__," ",sklearn .__version__)
print(matplotlib .__name__,"",matplotlib .__version__)
print(seaborn .__name__,"   ",seaborn .__version__)

sys.stdout.close()

#TODO Need to add print statements for other packages (see instructions)

 
   
#    print(np.__name__, np.__version__)
#    print(scipy.__name__, scipy.__version__)
#    print(pandas.__name__, pandas.__version__)
#    print(sklearn .__name__, sklearn .__version__)
#    print(matplotlib .__name__, matplotlib .__version__)
#    print(seaborn .__name__, seaborn .__version__)

