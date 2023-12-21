import pandas as pd
import numpy as np
import sys
from DecisionTree import DecisionTree
from sklearn.impute import KNNImputer
from collections import Counter

def load_data(file_path): 
    # Load data from a CSV file and display informationx
    df = pd.read_csv(file_path, sep=';')
    return df

def preprocess_data(df):
    # Prepare data by creating a dictionary for player names and extracting features and labels
    unique_names = df['Name'].unique()
    names_dict = {i: name for i, name in enumerate(unique_names[:31])}
    questions = [
        'Is your player currently playing?',
        'Has your player won the Golden Ball?',
        'Has your player played for La Liga?',
        'Is your player a world champion?',
        'Has your player played in the EPL?',
        'Is your player a midfielder?',
        'Has your player won the Champions League?',
        'Is your player an attacker?',
        'Is your player European?',
        'Has your player played for Liga 1?',
        'Is your player from Asia?',
        'Is your player a defender?',
        'Is your player African?',
        'Has your player won Copa America?',
        'Has your player played for Bundesliga?',
        'Is your player a goalkeeper?',
    ]
    
    df['Name'] = range(len(df))
    df.head()
    
    X = df.drop('Name', axis=1).values
    y = df['Name'].values
    
    return df, X, y, names_dict, questions

def train_decision_tree(X, y):
    # Train a decision tree model on the given features and labels
    model = DecisionTree()
    model.fit(X, y)
    return model

def get_imputed(arr, imputer, X, columns):
    # Impute missing values in the user's responses and return the imputed response
    temp = np.concatenate((arr, np.full(len(columns) - len(arr) - 1, np.nan))).reshape(1, -1)   
    z = np.concatenate((X, temp))
    imputed = imputer.fit_transform(z)[-1]
    return imputed

def play_akinator_game(model, names_dict, questions, imputer, X, columns):
    good_answers = ["y", "Y", "Yes", 1, "1", "11", "yes"]

    while True:
        answers = []
        predictions = []
        exit = False

        for i, question in enumerate(questions, start=1):
            print(question)
            print("-" * 40)
            
            sys.stdout.flush()
            answer = input()
            answers.append(1 if answer == "1" else 0)

            if i >= 12: 
                break

            imputed = get_imputed(answers, imputer, X, columns)
            predictions.append(model.predict([imputed])[0]) 
            counter = Counter(predictions)

            if counter[predictions[-1]] >= 7:
                print(f"\nI think the player is {names_dict[predictions[i-1]]}. Yes?")

                user_input = input()
                if user_input in good_answers:
                    print("\nI'm glad to have found your character!")
                    exit = True
                    break
                else:
                    predictions = [x for x in answers if x != predictions[-1]]
                    print("\nOK. Let's get on with it\n")
                    continue

        if not exit :
            ind = len(predictions) - 1
            print(f"\nI think the player is {names_dict[predictions[ind]]}.")    

        print("Do you want to play again?")
        again = input()
        
        if again == "0":
            print("\nThanks for playing!")
            break

if __name__ == "__main__":
    file_path = '/Users/mirasorazhan/Desktop/footballers 2copy.csv'
    df = load_data(file_path)
    
    df, X, y, names_dict, questions = preprocess_data(df)
    
    model = train_decision_tree(X, y)
    
    imputer = KNNImputer(n_neighbors=1)
    
    ask_for_game = input('Do you want to play Akinator game?\n')
    
    if ask_for_game in ["y", "Y", "Yes", 1, "1", "11", "yes"]:
        play_akinator_game(model, names_dict, questions, imputer, X, df.columns)
