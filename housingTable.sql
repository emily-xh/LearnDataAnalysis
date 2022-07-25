CREATE TABLE HOUSE (
  Id INT Not NULL PRIMARY Key,
  MSSubClass INT,
  MSZoning VARCHAR(255),
  LotFrontage INT,
  LotArea INT,
  Street VARCHAR(255),
  Alley VARCHAR(255),
  LotShape VARCHAR(255),
  LandContour VARCHAR(255),
  Utilities VARCHAR(255),
  LotConfig VARCHAR(255),
  LandSlope VARCHAR(255),
  Neighborhood VARCHAR(255),
  Condition1 VARCHAR(255),
  Condition2 VARCHAR(255),
  BldgType VARCHAR(255),
  HouseStyle VARCHAR(255),
  OverallQual INT,
  OverallCond INT,
  YearBuilt INT,
  YearRemodAdd INT,
  RoofStyle VARCHAR(255),
  RoofMatl VARCHAR(255),
  Exterior1st VARCHAR(255),
  Exterior2nd VARCHAR(255),
  MasVnrType VARCHAR(255),
  MasVnrArea INT,
  ExterQual VARCHAR(255),
  ExterCond VARCHAR(255),
  Foundation VARCHAR(255),
  BsmtQual VARCHAR(255),
  BsmtCond VARCHAR(255),
  BsmtExposure VARCHAR(255),
  BsmtFinType1 VARCHAR(255),
  BsmtFinSF1 INT,
  BsmtFinType2 VARCHAR(255),
  BsmtFinSF2 INT,
  BsmtUnfSF INT,
  TotalBsmtSF INT,
  Heating VARCHAR(255),
  HeatingQC VARCHAR(255),
  CentralAir VARCHAR(255),
  Electrical VARCHAR(255),
  1stFlrSF INT,
  2ndFlrSF INT,
  LowQualFinSF INT,
  GrLivArea INT,
  BsmtFullBath INT,
  BsmtHalfBath INT,
  FullBath INT,
  HalfBath INT,
  BedroomAbvGr INT,
  KitchenAbvGr INT,
  KitchenQual VARCHAR(255),
  TotRmsAbvGrd INT,
  Functional VARCHAR(255),
  Fireplaces INT,
  FireplaceQu VARCHAR(255),
  GarageType VARCHAR(255),
  GarageYrBlt INT,
  GarageFinish VARCHAR(255),
  GarageCars INT,
  GarageArea INT,
  GarageQual VARCHAR(255),
  GarageCond VARCHAR(255),
  PavedDrive VARCHAR(255),
  WoodDeckSF INT,
  OpenPorchSF INT,
  EnclosedPorch INT,
  3SsnPorch INT,
  ScreenPorch INT,
  PoolArea INT,
  PoolQC VARCHAR(255),
  Fence VARCHAR(255),
  MiscFeature VARCHAR(255),
  MiscVal INT,
  MoSold INT,
  YrSold INT,
  SaleType VARCHAR(255),
  SaleCondition VARCHAR(255),
  SalePrice INT
);


LOAD DATA INFILE 'train'
INTO TABLE HOUSE
FIELDS TERMINATED BY ','
IGNORE 1 ROWS;