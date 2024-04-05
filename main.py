import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score, auc
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Dataseti yükleme:
data = pd.read_csv('dataset.csv')
data.head(6) #İlk 6 satırın gösterilmesi


outcome_counts = data['Outcome'].value_counts()

#Grafik
plt.figure(figsize=(8, 6))
outcome_counts.plot(kind='bar',color=['purple','yellow'])
plt.title('Diyabet dağılımı')
plt.xlabel('Durum(1: pozitif, 0: negatif)')
plt.ylabel('Hasta sayısı:')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.show()

data.info()

data.isnull().sum()

#Featurelar içindeki null değerleri tespit etme (0)
data.eq(0).sum() 
data.shape
data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = data[['Glucose',
                                                                                                           'BloodPressure',
                                                                                                           'SkinThickness',
                                                                                                           'Insulin','BMI',
                                                                                                           'DiabetesPedigreeFunction',
                                                                                                           'Age']].replace(0,np.NaN)

#missing valueların doldurulmsı:
data.fillna(data.mean(), inplace = True)

data.head(6)
data.isnull().sum()
data.eq(0).sum()


plt.figure(figsize=(10, 6))
plt.scatter(data['Glucose'], data['Insulin'], alpha=0.3, color=['red'])
plt.title(' Glukoz İnsülin İlişkisi')
plt.xlabel('Glukoz')
plt.ylabel('İnsülin')
plt.show()



'''Min-Max normalizasyonu'''

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Normalizasyonu
normalized_data = (data - data.min()) / (data.max() - data.min())
print('\n')
print(normalized_data.head(6))

# PCA uygulayarak boyut indirgeme:
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
pca_attributes = np.argsort(pca.explained_variance_ratio_)[::-1]

# LDA iç,n:
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, y)
lda_attributes = np.argsort(lda.explained_variance_ratio_)[::-1]

# Data setimizi test ve eğitim için rastgele ayırma: (%30 ve %70)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Çoklu lineer regresyon analizi
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Lineer Regresyon Katsayıları:", lr.coef_)
print("Lineer Regresyonon Accuracy:", accuracy_score(y_test, y_pred_lr))

# Multinomial Lojistik regresyon:
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print("Lojistic Regresyon Katsayıları:", logreg.coef_)
print("Lojistic Regresyon  Accuracy:", accuracy_score(y_test, y_pred_logreg))

#Karar ağacı sınıflandırması:
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Karar Ağacı Konfüzyon Matrisi:", confusion_matrix(y_test, y_pred_dt))
print("Karar Ağacı Accuracy:", accuracy_score(y_test, y_pred_dt))

# Naive Bayes Sınıflandırıcısı:
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
print("Naive Bayes Konfüzyon Matrisi:", confusion_matrix(y_test, y_pred_gnb))
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_gnb))
print(classification_report(y_test, y_pred_gnb))