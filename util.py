from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os



class FieldHandler(object):
    def __init__(self, train_file_path, test_file_path=None, category_columns=[], continuation_columns=[]):
        """
        train_df_path: train file path(must)
        test_df_path: None or test file path
        """
        self.train_file_path = None
        self.test_file_path = None
        self.field_nums = 0
        self.field_dict = {}
        
        self.category_columns = category_columns
        self.continuation_columns = continuation_columns

        if not isinstance(train_file_path, str):
            raise ValueError("rain file path must str")
        if os.path.exists(train_file_path):
            self.train_file_path = train_file_path
        else:
            raise OSError("train file path isn't exists!")
        if test_file_path :
            if os.path.exists(test_file_path):
                self.test_file_path = test_file_path
            else:
                raise OSError("test file path isn't exists!")
        self.read_data()
        self.df[category_columns].fillna("-1", inplace=True)

        self.build_filed_dict()
        self.build_standard_scaler()

    def build_filed_dict(self):
        for column in self.df.columns:
            if column in self.category_columns:
                cv = self.df[column].unique()
                self.field_dict[column] = dict(zip(cv, range(self.field_nums, self.field_nums + len(cv))))
                self.field_nums += len(cv)
            else:
                self.field_dict[column] = self.field_nums
                self.field_nums += 1

    def read_data(self):
        if self.train_file_path and self.test_file_path:

            train_df = pd.read_csv(self.train_file_path)[self.category_columns+self.continuation_columns]
            test_df = pd.read_csv(self.test_file_path)[self.category_columns+self.continuation_columns]
            self.df = pd.concat([train_df, test_df])
        else:
            self.df = pd.read_csv(self.train_file_path)[self.category_columns+self.continuation_columns]

    def build_standard_scaler(self):
        if self.continuation_columns:
            self.standard_scaler = StandardScaler()
            self.standard_scaler.fit(self.df[self.continuation_columns].values)
        else:
            self.standard_scaler = None



def transformation_data(file_path:str, field_hander:FieldHandler, label=None):
    """
    lable: target columns name
    """
    df_v = pd.read_csv(file_path)
    if label:
        if label in df_v.columns:
            labels = df_v[[label]].values.astype("float32")
        else:
            raise KeyError(f'label "{label}" isn\'t exists')
    df_v = df_v[field_hander.category_columns + field_hander.continuation_columns]
    df_v[field_hander.category_columns].fillna("-1", inplace=True)
    df_v[field_hander.continuation_columns].fillna(-999, inplace=True)
    if field_hander.standard_scaler:
        df_v[field_hander.continuation_columns] = field_hander.standard_scaler.transform(df_v[field_hander.continuation_columns].values)

    df_i = df_v.copy()

    for column in df_v.columns:
        if column in field_hander.category_columns:
            df_i[column] = df_i[column].map(field_hander.field_dict[column])
            df_v[column] = 1
        else:
            df_i[column] = field_hander.field_dict[column]
    
    df_v = df_v.values.astype("float32")
    df_i = df_i.values.astype("int32")
    if label:
        return df_i, df_v, labels
    return df_i, df_v

