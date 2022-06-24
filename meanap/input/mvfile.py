import os 

with open('d.txt','r') as file:
    data = file.read().split('\n')

_ = [os.system(f'cp ground-truth_total/{d} ground-truth/') for d in data]
