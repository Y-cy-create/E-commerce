#套件引入
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#讀取csv
df = pd.read_csv('data.csv')
# print(df)

#剔除重複ID
session_counts = df['user_id'].value_counts(ascending=False)
multi_users = session_counts[session_counts > 1].count()
users_to_drop = session_counts[session_counts > 1].index

df = df[~df['user_id'].isin(users_to_drop)]
# print(f'The updated dataset now has {df.shape[0]} entries')

# 把timestamp改成日期和時間

df['timestamp_date'] = df['timestamp'].str.split(' ').str.get(0)#抓出年月日
df['timestamp_time'] = df['timestamp'].str.split(' ').str.get(1)#抓出時分秒微秒
df = df.drop(['timestamp'], axis=1)#timestamp砍掉
df['timestamp_hour'] = df['timestamp_time'].str.split(':').str.get(0)#抓出小時
df['timestamp_minute'] = df['timestamp_time'].str.split(':').str.get(1)#抓出分

# df['timestamp_date'] = pd.to_datetime(df['timestamp_date'])
df['timestamp_day'] = df['timestamp_date'].str.split('-').str.get(-1)#都是2017年1月，只需針對日期
df = df.drop(['timestamp_date'], axis=1)#timestamp_date砍掉
df = df.drop(['timestamp_time'], axis=1)#timestamp_time砍掉


#轉成數字格式

df['timestamp_hour'] = df['timestamp_hour'].astype('int64')
df['timestamp_minute'] = df['timestamp_minute'].astype('int64')
df['timestamp_day'] = df['timestamp_day'].astype('int64')


#把user_id變index
df.set_index("user_id", inplace=True)
# # print(df)


#group和landing_page進行dummy
df1 = pd.get_dummies(df['group'])
df2 = pd.get_dummies(df['landing_page'])
# print(df2)


#把dummy的series串起來
dummy = pd.concat([df1,df2], axis=1)
# print(dummy)
df = pd.concat([df,dummy], axis=1)
# print(df)
df = df.drop(['group'], axis=1)
df = df.drop(['landing_page'], axis=1)
print(df)
print(df.dtypes)


#x是特徵

x = df.drop(['converted'], axis=1)

#y是目標
y = df['converted']

#切成0.8的train，以及0.2的test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#使用決策樹建模
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)

#預測查看準確度
predictions = tree.predict(x_test)

print(tree.score(x_train, y_train))
#0.8890003139279361
print(tree.score(x_test, y_test))
#0.8610345669538526
