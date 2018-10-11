import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import seaborn as sns
sns.set()

def create_linear_model(X_train,X_test,y_train,y_test):
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    pred_train = lin.predict(X_train)
    pred_test = lin.predict(X_test)
    return (pred_train, pred_test)

def create_lasso_model(X, y, cv=10):
  alphas = np.logspace(-3,3,50)
  lasso = LassoCV(alphas=alphas, cv=cv).fit(X,y)
  print(f'Lasso train R^2: {lasso.score(X,y)}, alpha: {lasso.alpha_}')
  return lasso

def create_ridge_model(X, y, cv=10):
  alphas = np.logspace(-3,3,50)
  ridge = RidgeCV(alphas=alphas, cv=cv).fit(X,y)
  print(f'Ridge train R^2: {ridge.score(X,y)}, alpha: {ridge.alpha_}')
  return ridge

def test_accuracy(y_test, y_pred):
    return r2_score(y_test, y_pred)


def plot_scatters_with_bestfit(df):
    y = df['All_Wins']
    x = df['Goals']
    x2 = df['Avg_Goals_Per_Match']
    x3 = df['Total_Matches']

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(121)
    ax.scatter(x,y)
    axes = plt.gca()
    m, b = np.polyfit(x, y, 1)
    X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
    plt.plot(X_plot, m*X_plot + b, '-', c='r')
    ax.set_xlabel('Goals Scored')
    ax.set_ylabel('Wins')
    ax.set_title('Teams with Goals Scored vs. All Wins')

    ax2 = fig.add_subplot(122)
    ax2.scatter(x2,y)
    axes2 = plt.gca()
    m, b = np.polyfit(x2, y, 1)
    X_plot = np.linspace(axes2.get_xlim()[0],axes2.get_xlim()[1],100)
    plt.plot(X_plot, m*X_plot + b, '-', c='r')
    ax2.set_xlabel('Average Goals per Game')
    ax2.set_ylabel('Wins')
    ax2.set_title('Teams with Avg Goals vs. All Wins')

def plot_true_v_predicted(x,y_test,y_prediction):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.plot(x,y_test, color='tomato', label='True')
    ax.plot(x,test_predictions, color='royalblue',label='Predicted')
    plt.xticks(rotation=90)
    ax.set_ylabel('Wins')
    ax.set_title('2014 World Cup True Wins vs. Predicted Wins')
    ax.legend()

if __name__ == '__main__':
    # load data and split into train and test
    training = pd.read_csv('./data/training.csv')
    training.drop('Unnamed: 0',axis=1,inplace=True)
    test = pd.read_csv('./data/test.csv')
    test.drop('Unnamed: 0',axis=1,inplace=True)

    # Set target value and features
    y_train = training['All_Wins']
    X_train = training[['Goals','Avg_Goals_Per_Match','All_Draws','Total_Matches']]
    y_test = test['All_Wins']
    X_test = test[['Goals','Avg_Goals_Per_Match','All_Draws','Total_Matches']]

    # SKLearn model
    train_predictions, test_predictions = create_linear_model(X_train,X_test,y_train,y_test)
    # OLS model
    y = y_train
    x1 = training['Goals']
    x2 = training['Avg_Goals_Per_Match']
    x3 = training['All_Draws']
    model = smf.ols('y ~ x1 + x2 + x3', data=training).fit()
    # residuals = model.outlier_test()['student_resid']
    # sm.graphics.qqplot(residuals, line='45', fit=True)

    # Ridge and Lasso models and predictions
    ridge = create_ridge_model(X_train,y_train)
    lasso = create_lasso_model(X_train,y_train)
    ridge_pred = ridge.predict(X_test)
    lasso_pred = lasso.predict(X_test)

    # plots
    plot_scatters_with_bestfit(training) # 2006 and 2010
    plot_scatters_with_bestfit(test) #2014
    plot_true_v_predicted(test['Team'],y_test,test_predictions)
    plt.show()

    # R2 Score for sklearn and ols
    print("LinearRegression R2 score: {}".format(test_accuracy(y_test,test_predictions)))
    print(model.summary())

    # find team with highest wins prediction as winner of 2014s world cup
    p_winner_index = np.argmax(test_predictions)
    a_winner_index = np.argmax(y_test)
    p_winner = test.iloc[p_winner_index,0]
    a_winner = test.iloc[a_winner_index,0]
    print("Predicted Winner: {}\nActual Winner: {}".format(p_winner,a_winner))
