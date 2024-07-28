import pandas as pd
import math

df = pd.read_csv('Planilha manual entropy - Página1.csv')

# Parent
value_counts = df['Exame Result'].value_counts()
total_samples = len(df)
P_pass = value_counts['Pass'] / total_samples
P_fail = value_counts['Fail'] / total_samples

entropy_parent = -(P_pass * math.log2(P_pass) + P_fail * math.log2(P_fail))


# Working e Not_Working
working = df[df['Working Status'] == 'W']
not_working = df[df['Working Status'] == 'NW']

W_pass = len(working[working['Exame Result'] == 'Pass']) / len(working)
W_fail = len(working[working['Exame Result'] == 'Fail']) / len(working)
NW_pass = len(not_working[not_working['Exame Result'] == 'Pass']) / len(not_working)
NW_fail = len(not_working[not_working['Exame Result'] == 'Fail']) / len(not_working)

entropy_Working = -(W_pass * math.log2(W_pass) + W_fail * math.log2(W_fail))
entropy_Not_Working = -(NW_pass * math.log2(NW_pass) + NW_fail * math.log2(NW_fail))



# Entropia Background math, CS, Others
Maths = df[df['Student background'] == 'Maths']
CS = df[df['Student background'] == 'CS']
Other = df[df['Student background'] == 'Other']

Maths_Pass = (len(Maths[Maths['Exame Result'] == 'Pass']) / len(Maths))
Maths_Fail = (len(Maths[Maths['Exame Result'] == 'Fail']) / len(Maths))
CS_Pass = (len(CS[CS['Exame Result'] == 'Pass']) / len(CS))
CS_Fail = (len(CS[CS['Exame Result'] == 'Fail']) / len(CS))
Other_Pass = (len(Other[Other['Exame Result'] == 'Pass']) / len(Other))
Other_Fail =(len(Other[Other['Exame Result'] == 'Fail']) / len(Other))

def safe_entropy(p):
    if p == 0:
        return 0
    else:
        return -p * math.log2(p)


entropy_Maths = -(Maths_Pass * math.log2(Maths_Pass) + Maths_Fail * math.log2(Maths_Fail))
entropy_CS = (CS_Pass * safe_entropy(CS_Pass) + CS_Fail * safe_entropy(CS_Fail))
entropy_Other = (Other_Pass * safe_entropy(Other_Pass) + Other_Fail * safe_entropy(Other_Fail))



# Entropia de Online_couse_y e Online_course_n
online_y = (df[df['Other online courses']== 'y'])
online_n = (df[df['Other online courses'] == 'n'])

y_Pass = (len(online_y[online_y['Exame Result'] == 'Pass']) / len(online_y))
y_Fail = (len(online_y[online_y['Exame Result'] == 'Fail']) / len(online_y))
n_Pass = (len(online_n[online_n['Exame Result'] == 'Pass']) / len(online_n))
n_Fail = (len(online_n[online_n['Exame Result'] == 'Fail']) / len(online_n))



entropy_online_y = -(y_Pass * math.log2(y_Pass) + y_Fail * math.log2(y_Fail))
entropy_online_n = -(n_Pass * math.log2(n_Pass) + n_Fail * math.log2(n_Fail))



# Average entropy

# Working e Not_working

Avera_E_W_NK = (len(working) / len(df['Working Status']) * entropy_Working) + (len(not_working) / len(df['Working Status']) * entropy_Not_Working)



# BKGRD Mahts, CS, Other
Avera_E_Math_CS_Other = (len(Maths) / len(df['Student background']) * entropy_Maths) + (len(CS) / len(df['Student background']) * entropy_CS) * (len(Other) / len(df['Student background']) * entropy_Other)


# Online course y e n
Avera_E_y_n = (len(online_y) / len(df['Other online courses']) * entropy_online_y) + (len(online_n) / len(df['Other online courses']) * entropy_online_n)


# information gain

# Working e Not_working
IG_W_NW = (entropy_parent - Avera_E_W_NK)
# BKGRD_Mahts, CS, Other
IG_BKGRD = (entropy_parent - Avera_E_Math_CS_Other)
# Online course y e n
IG_online_course =(entropy_parent - Avera_E_y_n)

def tab1():
    data = {
        'Category': ['Parent', 'Working', 'Not Working', 'Background Maths', 'Background CS', 'Background Other',
                     'Online Courses Y', 'Online Courses N'],
        'Entropy': ['{:.4f}'.format(entropy_parent), '{:.4f}'.format(entropy_Working), '{:.4f}'.format(entropy_Not_Working), '{:.4f}'.format(entropy_Maths), '{:.4f}'.format(entropy_CS), '{:.4f}'.format(entropy_Other),
                    '{:.4f}'.format(entropy_online_y), '{:.4f}'.format(entropy_online_n)],
        'Average entropy': ['', '{:.4f}'.format(Avera_E_W_NK),'', '', '{:.4f}'.format(Avera_E_Math_CS_Other),'', '{:.4f}'.format(Avera_E_y_n), ''],
        'Information gain': ['', '{:.4f}'.format(IG_W_NW), '', '', '{:.4f}'.format(IG_BKGRD), '','{:.4f}'.format(IG_online_course) ,'' ]
    }

    # Criar DataFrame
    df_table = pd.DataFrame(data)

    return df_table

print(tab1())
'''entropy_Maths'''



def tab2():
    print(None)