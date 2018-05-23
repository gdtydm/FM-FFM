import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np

def dataGenerate(path="./Dataset/train.csv"):
    df = pd.read_csv(path)
    df = df[['Pclass',"Sex","SibSp","Parch","Fare","Embarked","Survived"]]
    class_columns = ['Pclass',"Sex","SibSp","Parch","Embarked"]
    continuous_columns = ['Fare']
    train_x = df.drop('Survived', axis=1)
    train_y = df['Survived'].values
    train_x = train_x.fillna("-1")
    le = LabelEncoder()
    oht = OneHotEncoder()
    files_dict = {}
    s = 0
    for index, column in enumerate(class_columns):
        try:
            train_x[column] =  le.fit_transform(train_x[column])
        except:
            pass
        ont_x = oht.fit_transform(train_x[column].values.reshape(-1,1)).toarray()
        for i in range(ont_x.shape[1]):
            files_dict[s] = index
            s +=1
        if index == 0:
            x_t = ont_x
        else:
            x_t = np.hstack((x_t, ont_x))
    x_t = np.hstack((x_t, train_x[continuous_columns].values.reshape(-1,1)))
    files_dict[s] = index + 1

    return x_t, train_y.reshape(-1,1), files_dict

