# encoding: utf-8

import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

"""
train.csv format
PassengerId     唯一标识的id,你懂的
Survived        活着（1），死了（0），我们要预测的就是这一列。
Pclass          乘客的等级，1,2,3
Name            名字
Sex             性别{male,female}
Age             年龄
SibSp           船上有几个同辈的亲戚，包括兄弟姐妹老婆老公
Parch           船上有几个他的父母小孩
Ticket          这个乘客的船票编号
Fare            乘客付了多少钱
Cabin           乘客在哪个船舱
Embarked        乘客登船的口岸

PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object（删除）
Sex            891 non-null object（字符串转离散数值）
Age            714 non-null float64（处理缺失值）
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object（删除）
Fare           891 non-null float64
Cabin          204 non-null object（处理缺失值）
Embarked       889 non-null object（处理缺失值）
"""

"""
Ranking:
5583,0.73206
"""


def run():
    test1()


def test1():
    pd.options.mode.chained_assignment = None
    # =========================================================================
    data = pd.read_csv("../data/train.csv")

    # ------------------------- Pclass
    data['Pclass_1'] = 0
    data['Pclass_2'] = 0
    data['Pclass_3'] = 0

    data.Pclass_1[data.Pclass == 1] = 1
    data.Pclass_2[data.Pclass == 2] = 1
    data.Pclass_3[data.Pclass == 3] = 1
    # data = data.drop('Pclass', axis=1)

    # ------------------------- Name
    data['TitleName'] = ''
    pattern = re.compile(', (.*?)\.')
    for i in range(data.shape[0]):
        data['TitleName'].iloc[i] = pattern.findall(data.Name[i])[0]
    # ['Mr' 'Mrs' 'Miss' 'Master' 'Don' 'Rev' 'Dr' 'Mme' 'Ms' 'Major' 'Lady'
    #  'Sir' 'Mlle' 'Col' 'Capt' 'the Countess' 'Jonkheer']

    title_list = list(data.TitleName.unique())
    for title in title_list:
        data[title] = 0
        data[title][data.TitleName == title] = 1

    # ------------------------- Sex
    data.Sex = data.Sex.map({'male': 1, 'female': 0})

    # ------------------------- Age
    data.Age = data.Age.fillna(data.Age.median())

    data['Child'] = 0
    data.Child[data.Age < 16] = 1

    data['Age_Pclass'] = data.Age * data.Pclass

    # min_age, max_age = data.Age.min(), data.Age.max()
    # data.Age = (data.Age - min_age) / (max_age - min_age)
    data.Age = (data.Age - data.Age.mean()) / data.Age.std()

    # ------------------------- Fare
    # min_fare, max_fare = data.Fare.min(), data.Fare.max()
    # data.Fare = (data.Fare - min_fare) / (max_fare - min_fare)

    # ------------------------- SibSp & Parch
    data['Family'] = data.SibSp + data.Parch
    data.Family[data.Family >= 1] = 1

    # ------------------------- Embarked
    data.Embarked.fillna('S')
    embarked_list = ['Embarked_C', 'Embarked_Q', 'Embarked_S']
    data['Embarked_C'] = 0
    data['Embarked_Q'] = 0
    data['Embarked_S'] = 0

    data.Embarked_C[data.Embarked == 'C'] = 1
    data.Embarked_Q[data.Embarked == 'Q'] = 1
    data.Embarked_S[data.Embarked == 'S'] = 1

    # =========================================================================
    # evalute model
    predictors = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex', 'Age',
                  'Age_Pclass', 'Child', 'Family', 'SibSp', 'Parch',
                  'Fare']
    predictors.extend(title_list)
    predictors.extend(embarked_list)

    for predictor in predictors:
        print predictor

    x = data[predictors]
    x.to_csv('x.csv')
    y = data.Survived
    n, m = data.shape[0], len(predictors)

    # clf = LinearRegression()
    # clf = SVC(C=100)
    # clf = GradientBoostingClassifier(learning_rate=0.1)
    clf = RandomForestClassifier(n_estimators=100)

    kf = KFold(n_splits=10)

    y_pred = []
    i = 1
    for train_idx, valid_idx in kf.split(x):
        print '=============================='
        print 'training epoch:{}'.format(i)
        i += 1
        train_x = x.iloc[train_idx]
        train_y = y.iloc[train_idx]
        clf.fit(train_x, train_y)

        train_y_pred = clf.predict(train_x)
        if isinstance(clf, LinearRegression):
            train_y_pred[train_y_pred < 0.5] = 0
            train_y_pred[train_y_pred >= 0.5] = 1

        print '------ train report ---------'
        print classification_report(train_y.values, train_y_pred)

        valid_x = x.iloc[valid_idx]
        valid_y_pred = clf.predict(valid_x)
        if isinstance(clf, LinearRegression):
            valid_y_pred[valid_y_pred < 0.5] = 0
            valid_y_pred[valid_y_pred >= 0.5] = 1
        print '------ valid report ---------'
        print classification_report(y.iloc[valid_idx].values, valid_y_pred)

        y_pred.append(valid_y_pred)
    y_pred = np.concatenate(y_pred)
    print '============= final result ==============='
    print classification_report(y, y_pred)

    if isinstance(clf, LinearRegression):
        c = (clf.coef_ * 1000).astype(np.int32).astype(np.float32) / 1000
        coef = pd.DataFrame({'coef': c, 'feature': x.columns})
        print coef

    persistence_data = data.copy(True)
    idx = list(persistence_data.columns).index('Survived')
    persistence_data.insert(idx, 'Predicted', y_pred)
    persistence_data.to_csv('predict.csv')

    clf.fit(x, y)
    print 'Score:{}'.format(clf.score(x, y))

    # =========================================================================
    # generate submission
    # clf.fit(x, y)
    # test_data = pd.read_csv('../data/test.csv')
    #
    # # ------------------------- Pclass
    # test_data['Pclass_1'] = 0
    # test_data['Pclass_2'] = 0
    # test_data['Pclass_3'] = 0
    #
    # test_data.Pclass_1[test_data.Pclass == 1] = 1
    # test_data.Pclass_2[test_data.Pclass == 2] = 1
    # test_data.Pclass_3[test_data.Pclass == 3] = 1
    #
    # # ------------------------- Name
    # test_data['TitleName'] = ''
    # pattern = re.compile(', (.*?)\.')
    # for i in range(test_data.shape[0]):
    #     test_data['TitleName'].iloc[i] = pattern.findall(test_data.Name[i])[0]
    # test_title_list = test_data.TitleName.unique()
    #
    # for title in title_list:
    #     test_data[title] = 0
    #     if title in test_title_list:
    #         test_data[title][test_data.TitleName == title] = 1
    #
    # # ------------------------- Sex
    # test_data.Sex = test_data.Sex.map({'male': 1, 'female': 0})
    #
    # # ------------------------- Age
    # test_data.Age = test_data.Age.fillna(data.Age.median())

    # test_data['Child'] = 0
    # test_data.Child[test_data.Age < 16] = 1

    # test_data.Age = (test_data.Age - min_age) / (max_age - min_age)
    #
    # # ------------------------- Fare
    # test_data.Fare = test_data.Fare.fillna(data.Fare.median())
    # test_data.Fare = (test_data.Fare - min_fare) / (max_fare - min_fare)


    # ------------------------- SibSp & Parch
    # test_data['Family'] = test_data.SibSp + test_data.Parch
    # test_data.Family[test_data.Family >= 1] = 1

    #
    # # ------------------------- Embarked
    # test_data.Embarked.fillna('S')
    # test_data['Embarked_C'] = 0
    # test_data['Embarked_Q'] = 0
    # test_data['Embarked_S'] = 0
    #
    # test_data.Embarked_C[test_data.Embarked == 'C'] = 1
    # test_data.Embarked_Q[test_data.Embarked == 'Q'] = 1
    # test_data.Embarked_S[test_data.Embarked == 'S'] = 1
    #
    # test_x = test_data[predictors]
    # test_y_pred = clf.predict(test_x)
    #
    # test_y_df = pd.DataFrame({'PassengerId': test_data.PassengerId,
    #                           'Survived': test_y_pred})
    # test_y_df.to_csv('../submissions/submission.csv', index=False)


if __name__ == '__main__':
    run()
