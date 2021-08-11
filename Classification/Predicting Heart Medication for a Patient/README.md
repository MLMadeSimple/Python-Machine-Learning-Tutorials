# Predicting Heart Medication for a Patient

In this tutorial, we will predict the heart medication best suited for a patient, given their current condition. We will be trying out, and scoring, many different classification algorithms to see which algorithm yeilds the best results.

**DISCLAIMER**: This project SHOULD NOT be used as medical advice. Please consult a Doctor instead.

The source code for each part can be found in the [GitHub Repository](https://github.com/MLMadeSimple/Python-Machine-Learning-Tutorials/tree/main/Classification/Predicting%20Heart%20Medication%20for%20a%20Patient)

## Contents
- [Part 1 - The Data](#part-1---the-data)
- [Part 2 - Loading Data Into Memory](#part-2---loading-data-into-memory)
- [Part 3 - Cleaning the Data](#part-3---cleaning-the-data)

## Part 1 - The Data
The data set which will be used in this tutorial will come from the user [prathamtripathi](https://www.kaggle.com/prathamtripathi) over on www.kaggle.com and can be found at https://www.kaggle.com/prathamtripathi/drug-classification.

Tripathi, Pratham. “Drug Classification.” Https://Www.kaggle.com/Prathamtripathi/Drug-Classification, 2020. 

The following is a small sample of the data which will be used.

| Age | Sex | BP     | Cholesterol | Na_to_K | Drug  | 
|-----|-----|--------|-------------|---------|-------| 
| 23  | F   | HIGH   | HIGH        | 25.355  | DrugY | 
| 47  | M   | LOW    | HIGH        | 13.093  | drugC | 
| 47  | M   | LOW    | HIGH        | 10.114  | drugC | 
| 28  | F   | NORMAL | HIGH        | 7.798   | drugX | 

The following is a brief description of all columns
- **Age** - The age of the patient
- **Sex** - The sex of the patient (While there are many different genders, this data set includes Female and Male only)
- **BP** - The blood pressure level
- **Cholesterol** - The Cholesterol level
- **Na_to_K** - The ratio of Sodium to Potassium
- **Drug** - The drug to give to the patient. This column contains values like drugX, and DrugY (the capitalization is inconsistent). It is not clear what drugs these refer to.

## Part 2 - Loading Data Into Memory
Our first job will be to read the data set into memory. This will be done with the Pandas library. Pandas can be installed via PIP (Assuming you have Python and PIP installed already). Pandas is a very large library, but it is my favourite libaray for data processing and management.

All of the libraries we will be using in today's tutorial can easily be installed via `PIP`:
```bash
pip install pandas
pip install numpy
pip install sklearn
```

We will use the following code to read the file into memory for further processing. This file can be found in this repository as `part2.py`. Make sure to change `Drug Classification.csv` to the location of the downloaded data set if it is not in the same directory as your Python script.

```python
import pandas as pd

df = pd.read_csv("Drug Classification.csv")

print(df)
```

If you run this script, and everything installed correctly, you should see the following contents in your console (or something similar to). You may see more or less rows than what is displayed here.
```
     Age Sex      BP Cholesterol  Na_to_K   Drug
0     23   F    HIGH        HIGH   25.355  DrugY
1     47   M     LOW        HIGH   13.093  drugC
2     47   M     LOW        HIGH   10.114  drugC
3     28   F  NORMAL        HIGH    7.798  drugX
4     61   F     LOW        HIGH   18.043  DrugY
..   ...  ..     ...         ...      ...    ...
195   56   F     LOW        HIGH   11.567  drugC
196   16   M     LOW        HIGH   12.006  drugC
197   52   M  NORMAL        HIGH    9.894  drugX
198   23   M  NORMAL      NORMAL   14.020  drugX
199   40   F     LOW      NORMAL   11.349  drugX

[200 rows x 6 columns]

Process finished with exit code 0
```

If you get an error message, where the last line is something along the lines of:
```
FileNotFoundError: [Errno 2] No such file or directory: 'Drug Classification.csv'
```

It means the script was unable to locate your CSV file, and it could not be read. Check where the CSV data set is saved, and where your Python script is, and make sure all the paths are correct.

So, what just happened? First, we imported Pandas. Easy.
```python
import pandas as pd
```

Next, we used Pandas to read our CSV file. This creates a new object called a `DataFrame` and stores it into our `df` variable (`df` is short for `dataframe`). The `DataFrame` object allows us to query, maniplulate, and perform many operations on our data set without the need for much code.
```python
df = pd.read_csv("Drug Classification.csv")
```

Finally, we printed out the dataframe, but, if you look at the output, we can only see *part* of the dataframe. This is because Pandas will only print the first and last few columns and rows in the dataframe so you can still see everything without lines getting wrapped because of your console size. A larger console size will allow you to see more columns and rows.

## Part 3 - Cleaning the Data
Before we can proceed, we need to do some cleaning of our data. The SKLearn library does not allow categorical features to be used in predictions. It only allows numerical features to be used. Unfortunatly, our data is mostly categorical data (3 of the 5 features - while there are 6 columns, the last one is our target and was not included in this count)

The two basic ways of doing this are with `OneHotEncoder` and `LabelEncoder`. In most cases, `OneHotEncoder` will be the best option, however, for completeness, both are explained in subsequent sections. You can use both types of encoders on the data set if needed.

### LabelEncoder
LabelEncoder essentially assigns an integer to each unique value in the category. For exmaple, if our categories are `Android`, and `IOS`, we could assign an integer to each category, which is done in the following table:

| category | integer |
|----------|---------|
| Android  | 2       |
| IOS      | 1       |

So we would replace all categories with the value of `Android` with a 2, and `IOS` with a 1 and the process would be complete. 

This method does come with a problem though. This method implies that `Android` is greater than `IOS`. Dispite being true, what happens if we have categories like `Apple`, `Orange`, `Kiwi` and `Lemon` that you can not compare in this way? If we use the following table:

| category | integer |
|----------|---------|
| Apple    | 4       |
| Orange   | 3       |
| Kiwi     | 2       |
| Lemon    | 1       |

This implies that an Apple is more similar to an Orange than a Lemon is (dispite an orange and lemon both being citrus fruits). In general, the seccond method is better practice, and the one we will be using in this tutorial.

### OneHotEncoder

OneHotEncoder basically creates a new feature for every state of each categorical feature. What I mean by this is, if we had our fruit categories of `Apple`, `Orange`, `Kiwi` and `Lemon` for our single feature, we would add an `Is Apple`, `Is Orange`, `Is Kiwi` and `Is Lemon` category to each instance. Of the four new features, only one of them can have a value of 1 and the rest will be zero.

For example, if our category was `Orange`, OneHotEncoder would convert this to the following data:

| Is Apple | Is Orange | Is Kiwi | Is Lemon |
|----------|-----------|---------|----------|
| 0        | 1         | 0       | 0        |

So that is enough theory, how do we implement this? The following code will accomplish this task:

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("Drug Classification.csv")

columns = ["Sex", "BP", "Cholesterol"]

one_hot_encoder = OneHotEncoder(sparse=False)

one_hot_encoded = one_hot_encoder.fit_transform(df[columns])

print(one_hot_encoded)
print(one_hot_encoder.get_feature_names(columns))
```

Running this code should yield the following output:

```
[[1. 0. 1. ... 0. 1. 0.]
 [0. 1. 0. ... 0. 1. 0.]
 [0. 1. 0. ... 0. 1. 0.]
 ...
 [0. 1. 0. ... 1. 1. 0.]
 [0. 1. 0. ... 1. 0. 1.]
 [1. 0. 0. ... 0. 0. 1.]]
 
['Sex_F' 'Sex_M' 'BP_HIGH' 'BP_LOW' 'BP_NORMAL' 'Cholesterol_HIGH'
 'Cholesterol_NORMAL']
```

The first thing we see printed is 2D array of the encoded data (using the OneHotEncoder method). You may notice that multiple columns have a value of 1. This is because this array encompasses all three of our categorical features (Sex, BP, and Cholesterol)

The seccond thing printed are the names for each of the columns. There are 7 items in this list, which correlate to the 7 0s or 1s in each row of our data (only 6 are shown because this is a numpy array which compresses itself down to save space on the console when printed).

So what is happening here?

First, we define an array which contains the names of each categorical column we wish to convert to a numeric column:
```python
columns = ["Sex", "BP", "Cholesterol"]
```

Next, we create an instance of the `OneHotEncoder` class which will perform the encoding for us:
```python
one_hot_encoder = OneHotEncoder(sparse=False)
```

Finally, we pass the data we wish to encode into the encoder. This returns a numpy ndarray object which contains the encoded data:
```python
one_hot_encoded = one_hot_encoder.fit_transform(df[columns])
```

Once all the data has been processed, we print the encoded data to the console:
```python
print(one_hot_encoded)
```

And print the feature names. What we pass in is a prefix to add to the new column. It will be concatenated with each value of the categorical feature to make the new name. For example, the we pass in `BP` for the prefix for the `BP` column. Given the only two values are `HIGH` and `LOW`, the two category names will be `BP_HIGH` and `BP_LOW`.
```python
print(one_hot_encoder.get_feature_names(columns))
```
