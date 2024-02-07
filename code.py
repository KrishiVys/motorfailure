import pandas as pd
!pip install --upgrade 'sqlalchemy<2.0'
!pip install pymysql
!pip install mysql-connector-python
from sqlalchemy import create_engine,inspect
source_database = 'NSMM'
source_username = 'root'
source_password = 'gtfm123'
source_host = '103.211.36.126'
destination_database = 'NSMM'
destination_username = 'root'
destination_password = 'gtfm123'
destination_host = '103.211.36.126'
source_engine = create_engine(f'mysql+mysqlconnector://{source_username}:{source_password}@{source_host}/{source_database}')
destination_engine = create_engine(f'mysql+mysqlconnector://{destination_username}:{destination_password}@{destination_host}/{destination_database}')
query1 = """
    SELECT NSMM.entity_information.device_id,
           NSMM.entity_information.motor_horse_power,
           NSMM.entity_information.phase_type
    FROM NSMM.entity_information
    JOIN NSMM_TRANS.vc_transactions
    ON NSMM.entity_information.device_id = NSMM_TRANS.vc_transactions.device_id;
"""
df_query = pd.read_sql(query1, con=source_engine)
df_query
db_username = 'root'
db_password = 'gtfm123'
db_host = '103.211.36.126'
db_port = '3306'  # Default is usually 3306
db_name = 'NSMM_TRANS'
engine = create_engine(f"mysql+mysqlconnector://{db_username}:{db_password}@{db_host}/{db_name}")
inspector = inspect(engine)
table_names = inspector.get_table_names()
for table_name in table_names:
    print(table_name)
query = "SELECT t1.device_id,t2.device_id,t1.VA,t1.VB,t1.VC,t1.IA,t1.IB,t1.IC ,t2.motor_horse_power, t2.phase_type FROM NSMM_TRANS.vc_transactions t1 INNER JOIN NSMM.entity_information t2 ON t1.device_id = t2.device_id;"
df = pd.read_sql(query, con=engine)
df
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
df = df.dropna()
df
overcurrent_threshold = 100  # Set an appropriate threshold for overcurrent in Amperes
overvoltage_threshold = 300  # Set an appropriate threshold for overvoltage in Volts
undervoltage_threshold = 200  # Set an appropriate threshold for undervoltage in Volts
df['Overcurrent'] = (df[['IA', 'IB', 'IC']] > overcurrent_threshold).any(axis=1)
df['Overvoltage'] = (df[['VA', 'VB', 'VC']] > overvoltage_threshold).any(axis=1)
df['Undervoltage'] = (df[['VA', 'VB', 'VC']] < undervoltage_threshold).any(axis=1)
df['MotorFailure'] = 'No'
failure_condition = (df['Overcurrent'] | df['Overvoltage'] | df['Undervoltage'])
df.loc[failure_condition, 'MotorFailure'] = 'Yes'
df.dropna(subset=['MotorFailure'], inplace=True)
X = df[['VA', 'VB', 'VC', 'IA', 'IB', 'IC']]
y = df['MotorFailure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')
df['power_factor'] = 0.9
df['power1'] = df['VA'] * df['IA']*(3**0.5)*df['power_factor']
df['power2'] = df['VB'] * df['IB']*(3**0.5)*df['power_factor']
df['power3'] = df['VC'] * df['IC']*(3**0.5)*df['power_factor']
df['total_power_consumption'] = df[['power1', 'power2', 'power3']].sum(axis=1)
print(df[['VA', 'IA', 'power1', 'VB', 'IB', 'power2', 'VC', 'IC', 'power3', 'total_power_consumption']])
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
features = ['VA', 'VB', 'VC', 'IA', 'IB', 'IC', 'power_factor']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
