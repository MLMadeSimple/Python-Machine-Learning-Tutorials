# Predicting Heart Medication for a Patient

In this tutorial, we will predict the heart medication best suited for a patient, given their current condition. 

**DISCLAIMER**: This project SHOULD NOT be used as medical advice. Please consult a Doctor instead.

The source code for each part can be found in the [GitHub Repository](https://github.com/MLMadeSimple/Python-Machine-Learning-Tutorials/tree/main/Classification/Predicting%20Heart%20Medication%20for%20a%20Patient)

## Contents
- [Part 1 - The Data](#part-1---the-data)

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
