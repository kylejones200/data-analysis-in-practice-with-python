# Data Analysis in practice with Python This project applies basic data analytics concepts to a real dataset
about properties listed on AirBNB (from InsideAirBNB). In this...

::::### Data Analysis in practice with Python 

This project applies basic data analytics concepts to a real dataset
about properties listed on AirBNB (from
[InsideAirBNB](http://insideairbnb.com/)). In this project, we:

1.  [import a file into a DataFrame]
2.  [remove extra characters in a column and recast the column as
    numeric]
3.  [plot the data as boxplots and histograms]
4.  [pivot ant cross-tabulate the data]
5.  [apply statistical techniques like ANOVA and regression to explore
    relationships in the data]
6.  [use sentiment analysis to examine housing descriptions]

#### Manipulate and clean data
Data is often "dirty" and needs to be cleaned up before we can analyze
it. Excel can do some of this but one limitation to Excel is that there
is no audit trail for the transformations that have been applied to the
data. This can lead to issues with reproducing the results later. It
also leads to "magic numbers" which are values like \`40\` or \`120\`
that appear in calculations but have no clear origin --- they appear
like magic and are hard to traceback to an origin.

We begin by importing pandas and numpy and reading in the dataset. Now
we have a Pandas DataFrame with the data. You'll notice this throws an
error. Pandas automatically sets the data type for the columns during
the import. Some of the columns are mixed which will create issues for
our analysis.

```python
import pandas as pd
import numpy as np
%matplotlib inline
df = pd.read_csv('data/Austin AirBNB data.csv')
df.head()
```

The DataFrame is large --- it has 74 columns. We won't need all of these
columns so you could drop the columns you don't need.

``` 
cols_to_drop = ['name','listing_url', 'scrape_id']
df.drop(cols_to_drop, axis=1, inplace=True)
df['price'].describe()
```

Let's look at the column for price. We want to see how the values are
distributed.

If we try to plot 'price' using a boxplot and histogram it doesn't this
work.

``` 
df.boxplot(column=['price'])

# What is this histograph showing us? Hint: this is not right :)
df['price'].hist()
```

When Pandas imported 'price', it determined this variable was a string,
not a number because it included the dollar sign and commas.

We need to remove "\$" and "," and then recast this column as a number
(floating point since it has decimals). We can quickly fix the issues
with price and continue our analysis. It is very common for datasets to
have issues like this.

```python
# sometimes characters end up in the data. 
# This removes the dollar sign from the price column.

df['price'] = df['price'].str.replace('$', '')
df['price'] = df['price'].str.replace(',', '').astype(float)
df.boxplot(column=['price'])
```

Now that we have the price as a number (float) and we can make a
boxplot. The boxplot is hard to read because there are so many extreme
values. Let's focus just on the values less than or equal to \$300. And
we can make a histogram of the same distribution.

``` 
df[df['price']<=300].boxplot(column=['price'])

df[df['price']<=300]['price'].hist()
```


At this point, we might want to save the cleaned DataFrame as a CSV so
we don't need to repeat those steps. We have cleaned the price column,
now let's do some analysis.

\# room_type and Price

Let's see if the kind of bed offered is correlated with price. Let's
look at bed type and see if that makes a difference.

``` 
df[df['price']< 301].boxplot(column=['price'], by = 'room_type')
```


Since there are more than two groups we need to use ANOVA instead of a
t-test. The StatsModels library can help.

```python
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('price ~ room_type', data=df).fit()
 
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)

```


When StatsModels does the ANOVA, it also does a linear regression. We
can see the results by calling the summary method on the model.


``` 
mod.summary()
```

Conclusion

The room_type variable is statistically significant but the R² value is
very low. If we want to predict 'price', we need to use other features.

#### Amenities and Price
Each property offers a variety of amenities. The data is hard to
analyze, so let's clean it up. Maybe amenities would be useful. There is
a column for amenities but it is a JSON object. Let's pull some values
from the column.

Let's convert this to a format we can use by making dummy variables for
key terms. We want to do analysis with these columns, so we convert the
boolean values to integers.

We create the "watch_words" list to hold the amenities we want to
examine. This list could be a single amenity or a long list. It is
important that we convert the column to lowercase before searching for
the watch word so we don't miss different spellings such as "WIFI",
"Wifi", "WiFi", or "wifi".

``` 
watch_words = ['wifi', 'cable tv', 'pet', 'cat']
for i in watch_words: 
 df[i] = df['amenities'].str.lower().str.contains(i)
df[watch_words]=df[watch_words].astype(int)

df[watch_words].describe()
```


Conclusion

We have additional factors we can use for analysis. It looks like almost
all properties have wifi and only a few allow pets or cats.

### Dogs and/or Cats versus Price
Let's use pivot tables to look at how the price changes for properties
that allow dogs and/or cats. Do properties that allow dogs and/or cats
have a higher price? Let's do a pivot table to see.

``` 
pd.pivot_table(df, values= 'price', columns = 'pet', index = 'cat', aggfunc="mean")
```


How many properties are there that allow dogs or cats? Let's change the
aggfunc from mean to count. We also add a summary column called "All"

``` 
pd.pivot_table(df, values= 'price', columns = 'pet', index = 'cat', aggfunc="count", margins=True)
```


The pivot gave us total count, but what we want to know is a percentage
of the total. Since we don't care about price, we can use crosstab.

``` 
pd.crosstab(df['cat'], df['pet'], normalize=True)
```


Nice. So less than 1% of the properties allow both pets and cats.

### Regression with Dogs and Cats versus Price
We can look at the interaction effect of allowing both dogs and cats
using a regression. We will use StatsModels for the regressions because
the structure is easier to read and understand than Scikit-Learn's
regression methods. We can see how dogs or cats effect price with a
regression.

``` 
mod = ols('price ~ cat*pet', data=df).fit()
 
mod.summary()
```

Conclusion: This interaction effect is not statistically significant.

#### Sentiment analysis
Let's try something more advanced. We have the description of each
property. Let's seen if the sentiment of the description is correlated
with price. We will use the "Vader" sentiment analysis library which is
pre-trained for social media text and emojiis.

```python
! pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
#Add VADER metrics to dataframe
df['desc'] = df['description'].astype(str)
#Add VADER metrics to dataframe
df['compound'] = [analyzer.polarity_scores(v)['compound'] for v in df['desc']]
df['neg'] = [analyzer.polarity_scores(v)['neg'] for v in df['desc']]
df['neu'] = [analyzer.polarity_scores(v)['neu'] for v in df['desc']]
df['pos'] = [analyzer.polarity_scores(v)['pos'] for v in df['desc']]
```


[**Intro to Natural Language Processing using NLTK in Python**\
*This simple project shows how to use natural language processing to
look at the most common terms in a
text.*medium.com](https://medium.com/@kylejones_47003/intro-to-natural-language-processing-using-nltk-in-python-ddce6a0ff8ac "https://medium.com/@kylejones_47003/intro-to-natural-language-processing-using-nltk-in-python-ddce6a0ff8ac")[](https://medium.com/@kylejones_47003/intro-to-natural-language-processing-using-nltk-in-python-ddce6a0ff8ac)
#### Regression with Sentiment Analysis Polarity Scores
Now that we have new features for sentiment, we can regress those
against price to see if there is an association.

```python
from statsmodels.formula.api import ols
mod = ols('price ~ compound + neg + neu + pos', data=df).fit()
mod.summary()
```

Conclusion: The only one that is statistically significant is the
compound score. Let's drop the others and try again. The compound score
ranges from \[-1, 1\]. Like a correlation value, the strength of the
relationship increases as you approach the extremes.

#### Compound Polarity Score versus Price
``` 
mod = ols('price ~ compound', data=df).fit()
mod.summary()
```

The coefficient is statistically significant. But it is still hard to
interpret. A 0.1 unit increase in compound polarity score is associated
with a \$2.45 increase in price.

While this method cannot impute causality, we can hypothesis that if the
market is functioning appropriately and the prices reflect what people
are willing/able to pay, then the tone of the description matters.

If you are interested in natural language processing, I have a few
articles that dive into that topic more.

### Related Stories
- [[Getting to know Pandas for data analytics with
  Python](https://medium.com/@kylejones_47003/getting-to-know-pandas-for-data-analytics-with-python-7386da28dd33)]
- [[Basic Data Analysis using Pandas Library in
  Python](https://medium.com/@kylejones_47003/basic-data-analysis-using-pandas-library-61ed815b834a)]
- [[Introduction to Statistics for people who do Business
  Analytics](https://medium.com/@kylejones_47003/introduction-to-statistics-for-people-who-do-business-analytics-26878760a14a)]
::::::::::::By [Kyle Jones](https://medium.com/@kyle-t-jones) on
[January 3, 2024](https://medium.com/p/95f26ab3ed6f).

[Canonical
link](https://medium.com/@kyle-t-jones/data-analysis-in-practice-with-python-95f26ab3ed6f)

Exported from [Medium](https://medium.com) on November 10, 2025.
