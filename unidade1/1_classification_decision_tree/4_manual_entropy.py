import pandas as pd
def tab1():
    df = pd.DataFrame({
        "Resp srl": ["Pass", "Fail", "Fail", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass",
                     "Fail", "Fail", "Fail"],
        "Target": ["Y", "N", "N", "Y", "N", "Y", "Y", "Y", "n", "n", "n", "n", "n", "n", "n"],
        "Predictor": ["Maths", "Maths", "Maths", "CS", "Other", "Other", "Maths", "CS", "Maths", "CS", "CS", "Maths",
                      "Other", "Other", "Maths"],
        "Predictor.1": ["NW", "W", "W", "NW", "W", "W", "NW", "NW", "W", "W", "W", "NW", "W", "NW", "W"],

        "Predictor.2": ["Student", "Working", "Working", "Student", "Student", "Student", "Student", "Student",
                        "Working", "Working", "Working", "Working", "Working", "Working", "Working"]
    })

    df.set_index(["Resp srl", "Target"], inplace=True)

    return df

Df = tab1()
print(Df.head(15))


def tab2():
    print(None)