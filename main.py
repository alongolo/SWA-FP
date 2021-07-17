from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import pandas as pd
from swa.tfkeras import SWA
from swa_improved_wraper import SWA_improved
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd
import numpy as np
import time
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from numpy import mean
from numpy import std
from sklearn.preprocessing import PolynomialFeatures
import imblearn
# automatic nested cross-validation for random forest on a classification dataset
from sklearn.datasets import make_classification
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import precision_score



from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import pandas as pd
from tensorflow.keras.layers import MaxPooling2D,Conv2D,Input,Add,MaxPool2D,Flatten,AveragePooling2D,Dense,BatchNormalization,ZeroPadding2D,Activation,Concatenate,UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from swa.tfkeras import SWA

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def complete_nan(df_x, df_y):
    imp_mean = IterativeImputer(max_iter=10, verbose=0, random_state=1)
    imp_mean.fit(df_x, df_y)
    imputed_df = imp_mean.transform(df_x)
    return pd.DataFrame(imputed_df, columns=df_x.columns), imp_mean

# Build the model (currently basic model)
def get_mlp_model(hiddenLayerOne=10, lr=0.1,xlen=10,ylen=10):

    # build model
    model = Sequential()
    model.add(Dense(hiddenLayerOne, input_dim=xlen, activation='relu'))
    model.add(Dense(ylen, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=SGD(lr=lr))
    return model

def convert_to_semi_supervised(X_train_to_convert,y_train_to_convert, label_name):

    # labled train - 20% from train
    n_first_rows = int(np.floor(X_train_to_convert.shape[0] * 0.2))
    X_train_labled = X_train_to_convert.iloc[:n_first_rows]
    y_train_labled = y_train_to_convert.iloc[:n_first_rows]
    y_list_lables = [x for x in y_train_labled]

    # unlabled train - 80% from train
    X_train_unlabled = X_train_to_convert.iloc[n_first_rows:]

    # train a model on the labeled trainset to predict the unlabeled records
    model = get_mlp_model(xlen=X_train_labled.shape[1], ylen=len(set(y_list_lables)))
    model.fit(X_train_labled, y_train_labled)
    y_hat_unlabled = model.predict(X_train_unlabled)
    y_hat_unlabled_df = pd.DataFrame(y_hat_unlabled)
    X_train_unlabled = X_train_unlabled.reset_index()
    X_train_unlabled = X_train_unlabled.drop(columns=['index'])
    te = y_hat_unlabled_df.idxmax(axis="columns")
    X_train_unlabled[label_name] = te
    y_train_labled = y_train_labled.reset_index()
    y_train_labled = y_train_labled.drop(columns=['index'])
    X_train_labled[label_name] = y_train_labled

    # append the predicted labeled to the labeled and create a new semi supervised train
    train = X_train_labled.append(X_train_unlabled)
    y_train = train[label_name]
    X_train = train.drop(columns=label_name)
    return X_train, y_train


def nested_cross_swa(df, df_results,label_name='clase',swa_improved=False):

    # configure the cross-validation procedure
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    # enumerate splits
    outer_results = list()

    cv_index = 0

    for train_ix, test_ix in cv_outer.split(df):

      cv_index += 1
      print('CV Index: {}'.format(cv_index))


      # split data
      X_train, X_test = df.iloc[train_ix, :], df.iloc[test_ix, :]
      y_train, y_test = df_results.iloc[train_ix], df_results.iloc[test_ix]

      # Use an iterative imputer to fill missing data, returns also the imputer which will will use later for the test set
      X_train,imputer = complete_nan(X_train, y_train)

      X_train, y_train = convert_to_semi_supervised(X_train,y_train, label_name)

      cv_inner = RepeatedStratifiedKFold(n_splits=3, n_repeats=50, random_state=1)

    # Build the model to optimize parameters (currently basic model)
      model = KerasClassifier(build_fn=get_mlp_model, verbose=0)

      X = X_train.iloc[:,:].values
      y = y_train.values

      y = to_categorical(y)

      # Configure range of search for the pearameters we want to optimize
      hiddenLayerOne = [x for x in range(10, 41, 10)]
      lr = np.arange(0.1, 0.4, 0.1)
      xlen = [len(X[0])]
      ylen = [len(y[0])]
      grid = dict(hiddenLayerOne = hiddenLayerOne,
                  lr = lr,
                  xlen = xlen,
                  ylen = ylen
                  )

      # define search
      search = RandomizedSearchCV(estimator=model, param_distributions=grid, n_iter=50, scoring='f1_macro', n_jobs=4, cv=cv_inner, random_state=1)

      # Configuraing the SWA model - original or improved
      if swa_improved == False:

          swa = SWA(start_epoch=2,
              lr_schedule='constant',
              swa_lr=0.01,
              verbose=1)
      else:
          swa = SWA_improved(start_epoch=2,
              lr_schedule='constant',
              swa_lr=0.01,
              verbose=1)

      # execute search
      start = time.time()

      #initialize optimization
      result = search.fit(X_train, y_train,epochs=50, callbacks=[swa])
      end = time.time()
      train_time = end - start
      print("Training time measure: {}".format(end - start))
      # get the best performing model fit on the whole training set
      best_model = result.best_estimator_
      X_test = pd.DataFrame(X_test, columns=X_test.columns)
      # evaluate model on the hold out dataset

      start = time.time()

      yhat = best_model.predict(X_test)
      end = time.time()
      predict_time = end - start
      print("Predict time measure: {}".format(end - start))

      # evaluate the model
      acc = accuracy_score(y_test, yhat)

      # store the result
      outer_results.append(acc)
      # report progress

      yhat = result.predict(X_test)
      labels = y_test.unique()
      # Binarize ytest with shape (n_samples, n_classes)
      ytest = label_binarize(y_test, classes=labels)
      # Binarize ypreds with shape (n_samples, n_classes)
      ypreds = label_binarize(yhat, classes=labels)

      cnf_matrix = confusion_matrix(y_test, yhat)
      FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
      FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
      TP = np.diag(cnf_matrix)
      TN = cnf_matrix.sum() - (FP + FN + TP)
      FP = FP.astype(float)
      FN = FN.astype(float)
      TP = TP.astype(float)
      TN = TN.astype(float)
      TPR = TP/(TP+FN)
      FPR = FP/(FP+TN)
      pr_curve_auc = average_precision_score(ytest, ypreds)
      precision = precision_score(y_test, yhat,average='macro')
      roc_auc = roc_auc_score(ytest, ypreds,multi_class='ovo',average='macro')
      print('>acc=%.3f, est=%.3f, prec=%.3f, cfg=%s' % (acc, result.best_score_,precision, result.best_params_))
      print('>ROC_Auc=%.3f, PR_Curve_Auc=%.3f, FPR=%.3f, TPR=%.3f' % (roc_auc, pr_curve_auc, np.mean(FPR), np.mean(TPR)))
      print()

      with open('report.txt', 'a') as f:
        f.write('CV Index: {}'.format(cv_index))
        f.write('\n')
        f.write('Train time: {}'.format(train_time))
        f.write('\n')
        f.write('Predict time: {}'.format(predict_time))
        f.write('\n')
        f.write('>acc=%.3f, est=%.3f, prec=%.3f, cfg=%s' % (acc, result.best_score_,precision, result.best_params_))
        f.write('\n')
        f.write('>ROC_Auc=%.3f, PR_Curve_Auc=%.3f, FPR=%.3f, TPR=%.3f' % (roc_auc, pr_curve_auc, np.mean(FPR), np.mean(TPR)))
        f.write('\n')
        f.write('\n')


    # summarize the estimated performance of the model
    print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))
    print()


with open('report.txt', 'w') as f:
    f.write("")

#
# # 1. Abalon
with open('report.txt', 'a') as f:
    f.write("DATASET: Abalon \n")

df = pd.read_csv(r".\classification_datasets\abalon.csv")
df_results = df['class']
df = df.drop(columns='class')
nested_cross_swa(df,df_results, swa_improved=True, label_name='class')
#
# 2. Steel-Plates
with open('report.txt', 'a') as f:
    f.write("DATASET: Steel-Plates \n")

df = pd.read_csv(r".\classification_datasets\steel-plates.csv")
df_results = df['clase']
df = df.drop(columns='clase')
nested_cross_swa(df,df_results)

# 3. Hill-Valley
with open('report.txt', 'a') as f:
    f.write("DATASET: Hill-Valley \n")

df = pd.read_csv(r".\classification_datasets\hill-valley.csv")
df_results = df['clase']
df = df.drop(columns='clase')
nested_cross_swa(df,df_results)

# 4. Wall Following
with open('report.txt', 'a') as f:
    f.write("DATASET: Wall Following \n")

df = pd.read_csv(r".\classification_datasets\wall-following.csv")
df_results = df['clase']
df = df.drop(columns='clase')
nested_cross_swa(df,df_results)

# 5. Plant Texture
with open('report.txt', 'a') as f:
    f.write("DATASET: Plant Texture \n")

df = pd.read_csv(r".\classification_datasets\plant-texture.csv")
df_results = df['clase']
df = df.drop(columns='clase')
nested_cross_swa(df,df_results)

# 6. Statlog-Image
with open('report.txt', 'a') as f:
    f.write("DATASET: Statlog-Image \n")

df = pd.read_csv(r".\classification_datasets\statlog-image.csv")
df_results = df['clase']
df = df.drop(columns='clase')
nested_cross_swa(df,df_results)

# 7. Bank
with open('report.txt', 'a') as f:
    f.write("DATASET: Bank \n")

df = pd.read_csv(r".\classification_datasets\bank.csv")
df_results = df['clase']
df = df.drop(columns='clase')
nested_cross_swa(df,df_results)

# 8. Ozone
with open('report.txt', 'a') as f:
    f.write("DATASET: Ozone \n")

df = pd.read_csv(r".\classification_datasets\ozone.csv")
df_results = df['clase']
df = df.drop(columns='clase')
nested_cross_swa(df,df_results)

# 9. Wine-Quality White
with open('report.txt', 'a') as f:
    f.write("DATASET: Wine-Quality White \n")

df = pd.read_csv(r".\classification_datasets\wine-quality-white.csv")
df_results = df['clase']
df = df.drop(columns='clase')
nested_cross_swa(df,df_results)

# 10. Waveform
with open('report.txt', 'a') as f:
    f.write("DATASET: Waveform \n")

df = pd.read_csv(r".\classification_datasets\waveform.csv")
df_results = df['clase']
df = df.drop(columns='clase')
nested_cross_swa(df,df_results)

# 11. Chess-krvkp
with open('report.txt', 'a') as f:
    f.write("DATASET: Chess-krvkp \n")

df = pd.read_csv(r".\classification_datasets\chess-krvkp.csv")
df_results = df['clase']
df = df.drop(columns='clase')
nested_cross_swa(df,df_results)

# 12. Plant Margin
with open('report.txt', 'a') as f:
    f.write("DATASET: Plant Margin \n")

df = pd.read_csv(r".\classification_datasets\plant-margin.csv")
df_results = df['clase']
df = df.drop(columns='clase')
nested_cross_swa(df,df_results)

# 13. Plant Shape
with open('report.txt', 'a') as f:
    f.write("DATASET: Plant Shape \n")

df = pd.read_csv(r".\classification_datasets\plant-shape.csv")
df_results = df['clase']
df = df.drop(columns='clase')
nested_cross_swa(df,df_results)

# 14. Musk-1
with open('report.txt', 'a') as f:
    f.write("DATASET: Musk-1 \n")

df = pd.read_csv(r".\classification_datasets\musk-1.csv")
df_results = df['clase']
df = df.drop(columns='clase')
nested_cross_swa(df,df_results)

# 15. Semieon
with open('report.txt', 'a') as f:
    f.write("DATASET: Semieon \n")

df = pd.read_csv(r".\classification_datasets\semeion.csv")
df_results = df['clase']
df = df.drop(columns='clase')
nested_cross_swa(df,df_results)

# 16. Spambase
with open('report.txt', 'a') as f:
    f.write("DATASET: Spambase \n")

df = pd.read_csv(r".\classification_datasets\spambase.csv")
df_results = df['clase']
df = df.drop(columns='clase')
nested_cross_swa(df,df_results)

# 17. Molec-biol-splice
with open('report.txt', 'a') as f:
    f.write("DATASET: Molec-biol-splice \n")

df = pd.read_csv(r".\classification_datasets\molec-biol-splice.csv")
df_results = df['clase']
df = df.drop(columns='clase')
nested_cross_swa(df,df_results)

# 18. Mfeat-Karhunen
with open('report.txt', 'a') as f:
    f.write("DATASET: Mfeat-Karhunen \n")

df = pd.read_csv(r".\classification_datasets\mfeat-karhunen.csv")
df_results = df['class']
df = df.drop(columns='class')
nested_cross_swa(df,df_results,label_name='class')

# 19. Arrhythmia
with open('report.txt', 'a') as f:
    f.write("DATASET: Arrhythmia \n")

df = pd.read_csv(r".\classification_datasets\arrhythmia.csv")
df_results = df['clase']
df = df.drop(columns='clase')
nested_cross_swa(df,df_results)

# 20. Low-res-spect
with open('report.txt', 'a') as f:
    f.write("DATASET: Low-res-spect \n")

df = pd.read_csv(r".\classification_datasets\low-res-spect.csv")
df_results = df['clase']
df = df.drop(columns='clase')
nested_cross_swa(df,df_results)


