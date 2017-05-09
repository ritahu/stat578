import pandas as pd
from nltk.tokenize import RegexpTokenizer
import chardet
import pickle

print("Missing Value Checking...")
df_train = pd.read_csv('train.csv')
mis = df_train.isnull().sum(axis=1)
print(mis[mis!=0])
df_train['question2'].ix[105780] = ' '
df_train['question2'].ix[201841] = ' '
mis = df_train.isnull().sum(axis=1)
print(mis[mis!=0])

df_test = pd.read_csv('test.csv')
mis = df_test.isnull().sum(axis=1)
print(mis[mis!=0])
df_test['question2'].ix[379205] = ' '
df_test['question2'].ix[817520] = ' '
df_test['question2'].ix[943911] = ' '
df_test['question1'].ix[1046690] = ' '
df_test['question2'].ix[1270024] = ' '
df_test['question1'].ix[1461432] = ' '
mis = df_test.isnull().sum(axis=1)
print(mis[mis!=0])

def clean_word_test(row):
    tokenizer = RegexpTokenizer(r'\w+')
    word_list = tokenizer.tokenize(row)
    word_series = pd.Series(word_list)
    word_nodigit_series = word_series[[word.isdigit()==False for word in word_list]]
    word_nodigit = word_nodigit_series.tolist()
    word_eng_series = word_nodigit_series[[chardet.detect(word)['encoding']=='ascii' for word in word_nodigit]]
    word_eng_list = word_eng_series.tolist()
    word_eng_lower_list = [word.lower() for word in word_eng_list]
    return(word_eng_lower_list)

print("question 1")
clean_word_1 = df_train['question1'].apply(clean_word_test)
print("question 2")
clean_word_2 = df_train['question2'].apply(clean_word_test)

print("question 1")
clean_word_1_test = df_test['question1'].apply(clean_word_test)
print("question 2")
clean_word_2_test = df_test['question2'].apply(clean_word_test)

print("Saving Clean Words...")
with open('clean_word_1.pickle', 'wb') as f:
     pickle.dump(clean_word_1, f)
with open('clean_word_2.pickle', 'wb') as f:
     pickle.dump(clean_word_2, f)
with open('clean_word_1_test.pickle', 'wb') as f:
    pickle.dump(clean_word_1_test, f)
with open('clean_word_2_test.pickle', 'wb') as f:
    pickle.dump(clean_word_2_test, f)



