# scratch file
if False:
    AC = pd.ExcelFile("C:\\Users\\bcorn\\Documents\\AC\\Data Spreadsheet for Animal Crossing New Horizons.xlsx")
    housewares = pd.read_excel(AC,"Housewares")
    house = housewares.Name.unique().tolist()

    acnh_levmat_house = levmat(house,output="s")

    acnh_levmat_house[acnh_levmat_house < 1].sort_values(ascending=False)


    s = len(lst) # length of list
    if s < 1:
        return
    levmat = np.eye(s) # identity matrix based on length of list
    print("Starting Levenshtein matrix...")
    for i in range(0,s):
        for j in range(i+1,s):
            levmat[i,j] = lev(lst[i],lst[j],True)
    print("Done with Levenshtein matrix!\nShowing plot...")

    symmat = levmat + levmat.T - np.eye(s)
    plt.imshow(symmat, cmap='hot', interpolation='nearest')
    plt.show()
    print("Done!")