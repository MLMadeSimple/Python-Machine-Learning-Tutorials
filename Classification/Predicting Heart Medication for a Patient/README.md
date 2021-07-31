# Predicting Heart Medication for a Patient

In this tutorial, we will predict the heart medication best suited for a patient, given their current condition. We will be trying out, and scoring, many different classification algorithms to see which algorithm yeilds the best results.

**DISCLAIMER**: This project SHOULD NOT be used as medical advice. Please consult a Doctor instead.

The source code for each part can be found in the [GitHub Repository](https://github.com/MLMadeSimple/Python-Machine-Learning-Tutorials/tree/main/Classification/Predicting%20Heart%20Medication%20for%20a%20Patient)

## Contents
- [Part 1 - The Data](#part-1---the-data)
- [Part 2 - Loading Data Into Memory](#part-2---loading-data-into-memory)

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


