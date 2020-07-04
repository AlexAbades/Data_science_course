import pandas as pd

print('Ex 1: Reading files in other directories')
df = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\03-Python-for-Data-Analysis-Pandas\example")
print(type(df))
print(df)
print('')

print('Ex 2: Creating files and reading it; csv')
df.to_csv('My output', index=False)
print(pd.read_csv('My output'))
print('')

print('Ex 3: Reading sheets of excel')
print(pd.read_excel('Excel_Sample.xlsx', sheet_name='Sheet1'))
print('')

print('Ex 4: writing excel files')
df.to_excel('Excel_Sample2.xlsx', sheet_name='NewSheet')
print('')

print('Ex 5: Reading html link')
data = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')
print(data[0])
print('')

print('Ex 6: Reading SQL')
from sqlalchemy import create_engine
# engine = create_engine('sqlite:///:memory')
# df.to_sql('my_table', engine)
# sqlfd = pd.read_sql('my_table', con=engine)
# print(sqlfd)