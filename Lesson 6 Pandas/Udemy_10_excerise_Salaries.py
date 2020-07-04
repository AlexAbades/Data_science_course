import numpy as np
import pandas as pd

# Ex 1: Read Salaries.csv as a DataFrame called sal.
print("Ex 1")
sal = pd.read_csv('Salaries.csv')
print(type(sal))
print(sal)

# Ex 2: Check the head of the DataFrame.
print('Ex 2')
print(sal.head())

# Ex 3: check the info
print('Ex 3')
print(sal.info())
print('')

# Ex 4: Average of a column
print('Ex 4: What is the average BasePay')
print(sal['BasePay'].mean())
print('')

# Ex 5: Highest amount on money of OvertimePay
print('Ex 5: Max function on a column')
print(sal['OvertimePay'].max())
print('')

# Ex 6: Finding a specific element in the data base; Finding the title of an employee
print('Ex 6: Finding a row, and showing a concrete value of a row, conditional selection')
print(sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['JobTitle'])
print('')

# Ex 7: Conditional selection, find a row and show a concrete value of a column; Finding the benefits of an employee
print('Ex 7: Conditional selection')
print(sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['TotalPayBenefits'])
print('')

# Ex 7: Conditional selection, find a row and show a specific column element; finding the highest paid person
print('Ex 8: Finding highest paid person')
print(sal[sal['TotalPayBenefits']==sal['TotalPayBenefits'].max()]['EmployeeName'])
print('')
print('2nd Way: Advanced way')
print(sal.loc[sal['TotalPayBenefits'].idxmax()])
print('')
print('Examples idx max and argmax')
print(sal['TotalPayBenefits'].idxmax())
print(sal['TotalPayBenefits'].argmax())
print(sal.iloc[sal['TotalPayBenefits'].argmax()])
print('')
# Argmax is essentially as the argmax function


# Ex 8: Conditional selection, finding the name of the lowest paid person
print('Ex 8: Finding the lowest paid person')
print(sal[sal['TotalPayBenefits']==sal['TotalPayBenefits'].min()]['EmployeeName'])
print('')
print('2nd Way, Advanced way')
print(sal.iloc[sal['TotalPayBenefits'].argmin()])
print('')

# Ex 9: Conditional selection; average per year of the payments between 2011-2014
print('Ex 9: Average between years')
print(sal.groupby('Year').mean()['BasePay'])
print('')
print('Ex 9.1: Only two Years')
print(sal.groupby('Year').mean().loc[[2012, 2013]]['BasePay'])
print('')
# Can we make a conditional with a data frame that has a groupby

# Ex 10: How many Job Titles are in Data Frame
print('Ex 10: Unique job title for 2013')
print(sal['JobTitle'].nunique())
print('')

# Ex 11: The most common jobs
print('Ex 11: Common jobs')
print(sal['JobTitle'].value_counts().head())
print('')

# Ex 12: How many Job Titles were represented by only one person in 2013
print('Ex 12: Unique job titles for 2013')
print(sum(sal[sal['Year']==2013]['JobTitle'].value_counts()==1))
print('')

# Ex 13: How many people have the word Chief in their job title
print(sal['JobTitle'])
print(sum(sal['JobTitle'].str.upper().apply(lambda Job: 'CHIEF' in Job)))
print('')

# Ex 14: Bonus: Is there a correlation between length of the Job Title string and Salary
print(np.corrcoef(sal['JobTitle'].apply(len), sal['TotalPayBenefits']))
print('')
print('2nd Method')
sal['title_len'] = sal['JobTitle'].apply(len)
print(sal[['title_len', 'TotalPayBenefits']].corr())