import pandas as pd
df = pd.read_csv("E:\\R & D\\project\\First VScode with python\\file concept\\flavors_of_cocoa.csv");
print(df['Company'][1:5 ]);    
print(df['Company'].iloc[1]);
print(df['Review Date'].loc[1]);