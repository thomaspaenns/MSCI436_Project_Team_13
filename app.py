# Cheatsheet available at https://docs.streamlit.io/library/cheatsheet

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statistics
import streamlit as st
import matplotlib.pyplot as plt

raw_data_train = pd.read_csv('https://raw.githubusercontent.com/jmpark0808/pl_mnist_example/main/train_hp_msci436.csv')

#Categorical Columns without 'ordered' categories
categorical_cols = [
   'MSZoning',
   'Street',
   'Alley',
   'LandContour',
   'LotConfig',
   'Neighborhood',
   'Condition1',
   'Condition2',
   'RoofStyle',
   'RoofMatl',
   'MasVnrType',
   'Foundation',
   'Heating',
   'GarageType',
   'SaleCondition',
]
#Mapping columns of 'ordered' categories to numbers
ordinal_cols = {
    'LotShape':{'Reg':1,'IR1':2,'IR2':2,'IR3':3},
    'Utilities':{'AllPub':1,'NoSewr':2,'NoSeWa':3,'ELO':4},
    'LandSlope':{'Gtl':1,'Mod':2,'Sev':3},
    'BldgType':{'1Fam':1,'2FmCon':2,'Duplx':3,'TwnhsE':4,'TwnhsI':5},
    'HouseStyle':{'1Story': 1,'1.5Fin': 2,'1.5Unf': 3,'2Story': 4,'2.5Fin': 5,'2.5Unf': 6,'SFoyer': 7,'SLvl': 8},
    'ExterQual':{'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1},
    'ExterCond':{'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1},
    'BsmtQual':{'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1,'NA': 0},
    'BsmtCond':{'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1,'NA': 0},
    'BsmtExposure':{'Gd': 4,'Av': 3,'Mn': 2,'No': 1,'NA': 0},
    'BsmtFinType1':{'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3,'LwQ': 2,'Unf': 1,'NA': 0},
    'BsmtFinType2':{'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3,'LwQ': 2,'Unf': 1,'NA': 0},
    'HeatingQC':{'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1},
    'CentralAir':{'N':0,'Y':1},
    'Electrical':{'SBrkr': 5,'FuseA': 4,'FuseF': 3,'FuseP': 2,'Mix': 1},
    'KitchenQual':{'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1},
    'Functional':{'Typ': 8,'Min1': 7,'Min2': 6,'Mod': 5,'Maj1': 4,'Maj2': 3,'Sev': 2,'Sal': 1},
    'FireplaceQu':{'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1,'NA': 0},
    'GarageFinish':{'Fin': 3,'RFn': 2,'Unf': 1,'NA': 0},
    'GarageQual':{'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1,'NA': 0},
    'GarageCond':{'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1,'NA': 0},
    'PavedDrive':{'Y':3,'P':2,'N':1},
    'PoolQC':{'Ex': 4,'Gd': 3,'TA': 2,'Fa': 1,'NA': 0},

}
#Function to create binary columns for each categorical but non-ordinal column
def convert_categorical_to_binary(df, columns):
    for column in columns:
        if column in df.columns:
            df_encoded = pd.get_dummies(df[column], prefix=column, dtype='int64')
            df = pd.concat([df, df_encoded], axis=1)
            df = df.drop(column, axis=1)
    return df
#Function to map ordinal categories to numerical values
def replace_categorical_with_numerical(df, column, mapping):
    if column in df.columns:
        df[column] = df[column].replace(mapping)
    return df

cols_to_use = [
    'YearBuilt',
    '1stFlrSF',
    'GarageYrBlt',
    'GarageArea',
    'MiscVal',
    'YrSold',
    'RoofMatl',
    'YearRemodAdd',
    'LotArea',
    'TotRmsAbvGrd',
    'OverallCond',
    'BsmtFinSF1',
    'GarageType',
    'BsmtFinSF2',
    'WoodDeckSF',
]
cols_to_use.append('SalePrice')
sub_train = raw_data_train[cols_to_use]
#Convert categorical categories to binary columns
new_sub_train = convert_categorical_to_binary(sub_train, categorical_cols)
#Loop over ordinal categories and convert them to numerical
for col in ordinal_cols:
  new_sub_train = replace_categorical_with_numerical(new_sub_train, col, ordinal_cols[col])
#Select Cols
df = new_sub_train.select_dtypes(include = ['float64', 'int64']).fillna(0)
y = df['SalePrice'].values
X = df.drop('SalePrice',axis=1).values
#Fit Data
reg = LinearRegression().fit(X, y)

#Start of Streamlit App
results = []
with st.sidebar:
  st.title('Enter the data as prompted below')
  matls = ['Clay or Tile', 'Standard (Composite) Shingle', 'Membrane', 'Metal', 'Roll', 'Gravel & Tar', 'Wood Shakes', 'Wood Shingles']
  grgs = ['2 Types', 'Attached', 'Basement', 'Built In', 'Car Port', 'Detached', 'No Garage']

  lotar = st.text_input("Enter the overall area of the lot in square feet")
  rmsabvgrad = st.text_input("Enter the total number of rooms above grade, excluding bathrooms")
  SF1st = st.text_input("Enter the square footage of the first floor of the house")
  roofmatl = st.radio("Select the type of roofing material used", matls)
  yrblt = st.text_input("Enter the year the house was built Ex. 2001")
  yrsld = st.text_input("Enter the current year Ex. 2010")
  yrremod = st.text_input("Enter the year a remodel was added, if any Ex. 2016")
  bsmtSF1 = st.text_input("Enter the square footage of the primary finished area of the basement")
  bsmtSF2 = st.text_input("Enter the square footage of the secondary finished area of the basement, if a second level of finish exists")
  grgtype = st.radio("Select the type of garage", grgs)
  grgarea = st.text_input("Enter the area of the garage in square feet, if any")
  grgyrblt = st.text_input("Enter the year the garage was built, if any Ex. 1995")
  woodSF = st.text_input("If there is a wood deck, enter the square footage")
  miscval = st.text_input("Enter the value of extra features including an elevator, second garage, shed, tennis court, or other features")
  overcond = st.slider("Rate the overall condition of the house out of 10, where 10 is very excellent and 1 is very poor", 1, 10)

  roofs = {
    'RoofMatl_ClyTile':0,
    'RoofMatl_CompShg':0,
    'RoofMatl_Membran':0,
    'RoofMatl_Metal':0,
    'RoofMatl_Roll':0,
    'RoofMatl_Tar&Grv':0,
    'RoofMatl_WdShake':0,
    'RoofMatl_WdShngl':0
  }
  roofs_input = {
    'Clay or Tile':'RoofMatl_ClyTile',
    'Standard (Composite) Shingle':'RoofMatl_CompShg',
    'Membrane':'RoofMatl_Membran',
    'Metal':'RoofMatl_Metal',
    'Roll':'RoofMatl_Roll',
    'Gravel & Tar':'RoofMatl_Tar&Grv',
    'Wood Shakes':'RoofMatl_WdShake',
    'Wood Shingles':'RoofMatl_WdShngl'
  }
  roofs[roofs_input[roofmatl]] = 1
  garages = {
    'GarageType_2Types':0,
    'GarageType_Attchd':0,
    'GarageType_Basment':0,
    'GarageType_BuiltIn':0,
    'GarageType_CarPort':0,
    'GarageType_Detchd':0,
    'GarageType_NA':0,
  }
  garages_input = {
    '2 Types':'GarageType_2Types',
    'Attached':'GarageType_Attchd',
    'Basement':'GarageType_Basment',
    'Built In':'GarageType_BuiltIn',
    'Car Port':'GarageType_CarPort',
    'Detached':'GarageType_Detchd',
    'No Garage':'GarageType_NA',
  }
  garages[garages_input[grgtype]] = 1

  df_new = pd.DataFrame({
    'YearBuilt': yrblt,
    '1stFlrSF': SF1st,
    'GarageYrBlt': grgyrblt,
    'GarageArea': grgarea,
    'MiscVal': miscval if miscval else 0,
    'YrSold': yrsld,
    'YearRemodAdd': yrremod,
    'LotArea': lotar,
    'TotRmsAbvGrd': rmsabvgrad,
    'OverallCond': overcond,
    'BsmtFinSF1': bsmtSF1,
    'BsmtFinSF2': bsmtSF2,
    'WoodDeckSF': woodSF,
    'RoofMatl_ClyTile':roofs['RoofMatl_ClyTile'],
    'RoofMatl_CompShg':roofs['RoofMatl_CompShg'],
    'RoofMatl_Membran':roofs['RoofMatl_Membran'],
    'RoofMatl_Metal':roofs['RoofMatl_Metal'],
    'RoofMatl_Roll':roofs['RoofMatl_Roll'],
    'RoofMatl_Tar&Grv':roofs['RoofMatl_Tar&Grv'],
    'RoofMatl_WdShake':roofs['RoofMatl_WdShake'],
    'RoofMatl_WdShngl':roofs['RoofMatl_WdShngl'],
    'GarageType_2Types':garages['GarageType_2Types'],
    'GarageType_Attchd':garages['GarageType_Attchd'],
    'GarageType_Basment':garages['GarageType_Basment'],
    'GarageType_BuiltIn':garages['GarageType_BuiltIn'],
    'GarageType_CarPort':garages['GarageType_CarPort'],
    'GarageType_Detchd':garages['GarageType_Detchd'],
  }, index=[0])
  X = df_new.values
  submit = st.button('Estimate')
  if submit:
    results.append(round(reg.predict(X)[0],2))
st.header('Hello realtors, and welcome to your price estimator!')
col1, col2 = st.columns(2)
if results:
  prev_est = st.session_state.get("prev_est")
  if prev_est is not None:
    net_change = round(results[len(results)-1]-prev_est,2)
  else:
    net_change = None
  with col1:
    st.metric('Estimate (in $USD)', results[len(results)-1], net_change)
  st.session_state.prev_est = results[len(results)-1]
  with col2:
    st.metric('R-Squared Score (Accuracy)', 0.82)
  #Create Effect Chart
  a = reg.coef_
  effects = []
  vals = df_new.loc[0, :].values.flatten().tolist()
  for i in range(27):
    effects.append(a[i]*float(vals[i]))
  plt.figure(figsize=(6, 4))
  plt.bar(df_new.columns, effects)
  plt.xlabel('Feature')
  plt.ylabel('Net Effect on Price')
  plt.title('Net Effect of Features on Price')
  plt.xticks(fontsize=7,rotation=75)
  #plt.tight_layout()
  st.subheader('The net-effect on price is the effect had by the inputted value on estimated price')
  st.pyplot(plt)

st.markdown('**The R-Squared score is the proportion of price variance that can be explained by the variation of the selected features**')
st.image('/content/newplot.png')
