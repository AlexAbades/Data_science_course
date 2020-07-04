import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
print(tips)
print(flights)

# HEAT MAP: Your data has to be already in matrix form. What's a matrix form? It's when the index name and the column
# name matches up, so that cell value actually indicates something that it's relevant to both of those names

# CONVERTING A DATAFRAME INTO A MATRIX FORM
tc = tips.corr()
print(tc)

sns.heatmap(tc, annot=True, cmap='coolwarm')
plt.show()
# What the heat map does, is it colors in those values based on some sort of gradient scale. The annotation argument
# "annot" it's going to write the real numeric value in each of those cells

# An other way to make the matrix form, it's the pivot table argument.
fc = flights.pivot_table(index='month', columns='year', values='passengers')
# As we have 12 months per year, we can put in that format, where the year is in the column name and the index are
# months per year. At last, we have to specify what values we want to represent
print(fc)
sns.heatmap(fc, cmap='magma', linecolor='white', linewidths=0.3)
plt.show()
# linecolor and linewidth parameters are, the what their names express.

# CLUSTER MAP: It tries to cluster rows and columns together based in their similarity. The years and the columns
# aren't in order, because the cluster plot tries to cluster the most similar years and columns. We can see the
# different levels of hierarchy of the clusters based off these kind of tree diagrams in the "x" and "y" axes

sns.clustermap(fc, cmap='coolwarm', linecolor='black', lw=0.5, )
plt.show()

# We can standardise the scale, in the upper plot, the scale is of passengers from 0 to 600 but we can normalize this
# by passing the "standard_scale" argument, which normalize this to 0 to 1. We can see the similarities through the
# cluster

sns.clustermap(fc, cmap='coolwarm', linecolor='black', lw=0.5, standard_scale=1)
plt.show()

