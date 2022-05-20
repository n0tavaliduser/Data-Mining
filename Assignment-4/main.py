import os
from tabulate import tabulate
from naivebayes import naivebayes

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

def main(): 
    keepRun = 1
    while keepRun == 1:
        cls()
        print("==Graph Representation==")
        print("1. Atur Dataset")
        print("2. Tentukan X")
        print("3. Tentukan Y")
        print("4. Tampilkan Klasifikasi Naive Bayes")
        print("5. ")
        print("6. ")
        print("0. ")
        choose = str(input("pilihan --> "))
        if choose == "1":
            nb.setDataset(str(input("<nama file>.format (csv/xlsx) -> ")))
            enter = input("Dataset diatur, press ENTER to continue ... ")
        
        elif choose == "2":
            nb.setX(2)
            enter = input("X telah diatur, press ENTER to continue ... ")
        
        elif choose == "3":
            nb.setY(int(input("Y index ke -> ")))    
            enter = input("Y telah diatur, press ENTER to continue ... ")
        
        elif choose == "4":
            nb.startClassification()
            enter = input("relasi telah dibuat, press ENTER to continue ... ")
        
        elif choose == "5":
            pass

        elif choose == "6":
            pass

        elif choose == "0":
            keepRun += 1

        
if __name__ == "__main__":
    nb = naivebayes()
    main()