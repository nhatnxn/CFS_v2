import numpy as np
from statistics import stdev

def vertical_cluster(free_list):

    theta = 6
    data = []
    blocks = []

    for i,dat in enumerate(free_list):
        data.append([dat,i])
    data.sort(key=lambda x:x[0][0][1])
    
    # create a list of the gaps between the consecutive values
    gaps = [y[0][0][1] - x[0][0][1] for x, y in zip(data[:-1], data[1:])]
    # have python calculate the standard deviation for the gaps
    if len(gaps)>1:
        sd = stdev(gaps) + theta
    else:
        return [range(0,len(free_list))]

    # create a list of lists, put the first value of the source data in the first
    lists = [[data[0]]]

    for x in data[1:]:

        # if the gap from the current item to the previous is more than 1 SD
        # Note: the previous item is the last item in the last list
        # Note: the '> 1' is the part you'd modify to make it stricter or more relaxed
        if (x[0][0][1] - lists[-1][-1][0][0][1]) / sd > 1:
            # then start a new list
            lists.append([])
        # add the current item to the last list in the list
        lists[-1].append(x)
    
    
    # blocks = []
    for ls in lists:
        ls.sort(key=lambda x:x[0][0][1])
        ls = np.array(ls)
        # print(ls)
        blocks.append(ls[:,1])

    return blocks

def horizontal_cluster(free_list):
    data1 = []
    for i,dat in enumerate(free_list):
        data1.append([dat,i])
    data1.sort(key=lambda x:x[0][0][0])
    # free_list.sort(key=lambda x:x[0][0])
    blocks1 = []
    count = 1
    for data in [data1]:
        if len(data)==0:
            continue
        if len(data) < 3:
            blocks1.append([d[1] for d in data])
            continue
        # data = free_list

        # create a list of the gaps between the consecutive values
        gaps = [y[0][0][0] - x[0][0][0] for x, y in zip(data[:-1], data[1:])]
        # have python calculate the standard deviation for the gaps
        sd = stdev(gaps)


        # create a list of lists, put the first value of the source data in the first
        lists = [[data[0]]]
        for x in data[1:]:
            # if the gap from the current item to the previous is more than 1 SD
            # Note: the previous item is the last item in the last list
            # Note: the '> 1' is the part you'd modify to make it stricter or more relaxed
            if sd < 1:
                lists[-1].append(x)
                lists.append([])
                continue
            if (x[0][0][0] - lists[-1][-1][0][0][0]) / sd > 1:
                # then start a new list
                lists.append([])
            # add the current item to the last list in the list
            lists[-1].append(x)
        
        # blocks = []
        
        for ls in lists:
            ls.sort(key=lambda x:x[0][0][1])
            if ls:
                ls = np.array(ls)
                blocks1.append(ls[:,1])
            else:
                continue
    
    return blocks1