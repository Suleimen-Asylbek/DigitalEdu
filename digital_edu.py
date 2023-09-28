import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
df = pd.read_csv('train.csv')
df.drop(['has_photo', 'followers_count', 'last_seen','relation', 'id', 'has_mobile', 'bdate', 'city', 'graduation', 'occupation_name', 'people_main', 'career_start', 'career_end' ],axis = 1, inplace = True)

def s_apply(sex):
    if sex == 2:
        return 0
    return 1
df['sex'] = df['sex'].apply(s_apply)

df['education_form'].fillna('Full-time', inplace = True)
df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df.drop('education_form', axis = 1, inplace = True)

def ed_form(edu):
    if edu.find('Alumnus') != -1:
        return 1
    elif edu.find('Student') != -1:
        return 2
    elif edu.find('Undergraduate applicant') != -1:
        return 3
    else:
        return 4
df['education_status'] = df['education_status'].apply(ed_form)

def langs(lan):
    if lan.find('Русский') != -1:
        return 1
    return 0
df['langs'] = df['langs'].apply(langs)

def life(lif):
    if lif == 0:
        return 1
    elif lif == 1:
        return 2
    elif lif == 6:
        return 3
    else:
        return 4

df['life_main'] = df['life_main'].apply(life)

def occ_type(o):
    if o =='university':
        return 1
    else:
        return 0

df['occupation_type'] = df['occupation_type'].apply(occ_type)

print(df.info())

X = df.drop('result', axis = 1) # Данные о пассажирах
y = df['result'] # Целевая переменная
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(y_test)
print(y_pred)
print('Процент правильного предсказанных исходов:', round(accuracy_score(y_test, y_pred) * 100),'%')