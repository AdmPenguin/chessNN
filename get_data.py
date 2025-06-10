import csv
import numpy as np

import chess_funcs as cf
import data_setup as ds

def main():
    # Registers player playing data to train nn
    player_name = input("Enter name to save: ")

    test_positions = ds.sample_games(ds.FILE, g = 500, n = 100)
    data = []

    end = False

    for i in range(len(test_positions)):
        cf.printBoard(test_positions[i])
        print("Test {}/{}".format(i + 1, len(test_positions)))
        valid = False

        if cf.isCheckmate(test_positions[i])[0]:
            continue

        while not valid:
            move = input("Make a move: ").split(" ")

            if(move[0] == "s"):
                break
            elif move[0] == "end":
                end = True
                break

            try:
                for elem in range(len(move)):
                    move[elem] = int(move[elem]) - 1
                valid = cf.checkValid(test_positions[i], move, True)[0]
            except:
                continue
            if valid:
                data.append([test_positions[i], move])
                break
        
        if end:
            break

    file_name = "datasets/" + player_name + ".csv"
    
    with open(file_name, 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile)

        for arr1, arr2 in data:
            arr1_str = ' '.join(map(str, arr1))
            arr2_str = ' '.join(map(str, arr2))
            writer.writerow([arr1_str, arr2_str])
    
    print("Saved data to dataset/{}.csv".format(player_name))

if __name__ == "__main__":
    main()