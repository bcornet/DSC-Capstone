import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def levenshtein_ratio_and_distance(s, t, ratio_calc = True):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # skip everything if both strings are identical
    if s == t:
        if ratio_calc:
            return 1.0
        else:
            return "The strings are 0 edits away"
   
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    
    # if one of the strings is empty, maximum distance
    if rows == 1:
        if ratio_calc:
            return 0.0
        else:
            return "The strings are %d edits away"%len(t)
    if cols == 1:
        if ratio_calc:
            return 0.0
        else:
            return "The strings are %d edits away"%len(s)
    
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[row][col])
        
def levmat(lst=[],output=None,show=True,echo=True):
    """ levmat:
        Calculates Levenshtein matrix between a list of strings.
        Input:
        lst: some list of strings (default=[]). 
            If the list is empty, this will return None.
        output: string (default=None).
            Determines the output type:
            'n','np','numpy','matrix': Numpy ndarray
            'p','d','df','dataframe','pandas': Pandas DataFrame
            's','stack','series','list': Pandas Series (stacked DataFrame)
        show: boolean (default: True).
            Shows a heatmap of the matrix in matplotlib.pyplot.
        echo: boolean (default: True).
            Prints out steps in console.
    """
    s = len(lst) # length of list
    if type(output) == str:
        output = output.lower()
    numpyL = ['n','np','numpy','matrix']
    pandasL = ['p','d','df','dataframe','pandas']
    seriesL = ['s','stack','series','list']
    if not lst:
        print("Invalid list provided, returning None.")
        return None
    levmat = np.eye(s) # identity matrix based on length of list
    if echo:
        print("Creating Levenshtein matrix...")
    for i in range(0,s):
        for j in range(i+1,s):
            levmat[i,j] = levenshtein_ratio_and_distance(lst[i],lst[j],True)
    if echo: # for mean and std, ignore
        print("Done!")
    symmat = levmat + levmat.T - np.eye(s)
    if echo:
        df = pd.DataFrame(symmat.copy(), columns=lst, index=lst)
        series = df.copy().stack()
        desc = series[series<1].describe()
        desc['count'] = desc['count']/2
        print("\nShowing summary statistics...\n[All Pairs]") # summary of all combinations
        print(desc)
        print("\n[Maximums]") # summary of the closest matches; worst case scenarios
        print(pd.Series((symmat-np.eye(s)).max(axis=0)).describe())
        print("\n[Lengths]") # summary of the string lengths themselves
        print(pd.Series(lst).str.len().describe())
    if show:
        if echo:
            print("\nShowing plot...")
        plt.imshow(symmat, cmap='hot', interpolation='nearest')
        plt.show()
        if echo:
            print("Done!")

    if output in numpyL:
        return symmat
    elif output in pandasL:
        return pd.DataFrame(symmat, columns=lst, index=lst)
    elif output in seriesL:
        return pd.DataFrame(symmat, columns=lst, index=lst).stack()
    return None