import numpy as np
import pandas as pd

# Ex1: Importing Ecommerce file
print('Ex 1')
ecom = pd.read_csv('Ecommerce Purchases')
print(type(ecom))
print(ecom)
print('')

# Ex 2: Check the head of the data frame
print('Ex 2')
print(ecom.head())
print('')

# Ex 3 How many rows and columns are there
print('Ex 3')
print(ecom.info())
print('')
print('2nd Method')
print(len(ecom.columns))
print('')
print(len(ecom.index))

# Ex 4: Average credit purchase price
print('Ex 4')
print(ecom['Purchase Price'].mean())
print('')

# Ex 5: Highest and lowest purchase prices
print('Ex 5')
print(ecom['Purchase Price'].max())
print('')
print(ecom['Purchase Price'].min())
print('')

# Ex 6: How many people have English 'en' as their Language of choice on the website
print('Ex 6')
print(sum(ecom['Language'] == 'en'))
print('')
print('2nd Method')
print(ecom[ecom['Language'] == 'en'].info())
print('3th Method')
print(ecom[ecom['Language'] == 'en'].count()['Language'])
print('')

# Ex 7: How many people have the job title of "Lawyer"
print('Ex 7')
print(sum(ecom['Job'] == 'Lawyer'))
print('2nd Method')
print(ecom[ecom['Job'] == 'Lawyer'].info())
print('3th Method')
print(ecom[ecom['Job'] == 'Lawyer'].count())
print('')

# Ex 8: How many people made the purchase during the AM and how many people made the purchase during PM:
print('Ex 8')
print(ecom['AM or PM'].value_counts())
print('')

# Ex 9: What are the 5 most common Job Titles
print('Ex 9')
print(ecom['Job'].value_counts().head())
print('')

# Ex 10: Someone made a purchase that came from Lot: "90 WT" , what was the Purchase Price for this transaction
print('Ex 10')
print(ecom[ecom['Lot'] == '90 WT']['Purchase Price'])
print('')

# Ex 11:What is the email of the person with the following Credit Card Number: 4926535242672853
print('Ex 11')
print(ecom[ecom['Credit Card'] == 4926535242672853]['Email'])
print('')

# Ex 12: How many people have American Express as their Credit Card Provider *and made a purchase above $95
print(ecom[(ecom['CC Provider'] == 'American Express') & (ecom['Purchase Price'] > 95)].count())
print('')
print('2nd Method')
print(sum((ecom['CC Provider'] == 'American Express') & (ecom['Purchase Price'] > 95)))
print('')

# Ex 13: How many people have a credit card that expires in 2025
print('Ex 13')
print(sum((ecom['CC Exp Date'].apply(lambda date: date.split('/')[1])) == '25'))
# print(ecom['CC Exp Date'].apply(lambda date: date.split('/')[1]))
# The problem it has been that the date was a string! Check it before selecting a value
print(ecom['CC Exp Date'].iloc[0])
print('')
print('2nd Method')
print(sum(ecom['CC Exp Date'].apply(lambda date: date[3:] == '25')))
# The count methods it can only be applied whe we have the information values, not a boolean expression
print(ecom[ecom['CC Exp Date'].apply(lambda date: date[3:] == '25')].count())
print('')


# Ex 14: What are the top 5 most popular email providers/hosts (e.g. gmail.com, yahoo.com, etc...)
print('Ex 13')
print(ecom['Email'].apply(lambda mail: mail.split('@')[1]).value_counts().head())
# d= ecom['Email'].apply(lambda mail: mail.split('@')[1])
# print(d.value_counts())
