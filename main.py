import pdb
from c45 import C_45
import sys
import random
#import os

#print(os.listdir())
#alist = []
f1 = input("Enter the data file name : ")
f2 = input("Enter the names file name : ")
c1 = C_45(f1, f2)
#c1.fetchData()
try:
    if str(sys.argv[1]) == "80-20":
        c1.get_format_data()
        c1.preformatdata()
        c1.Generate_Tree()
        c1.Print_Tree()
        c1.test_data()
    elif str(sys.argv[1]) == "3fold":
        c1.get_format_data()
    else:
        print("please enter a valid option")
        sys.exit()
except:
    print("Please enter a proper format")
    sys.exit()
