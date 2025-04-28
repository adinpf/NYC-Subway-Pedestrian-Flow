import pandas as pd

df = pd.read_pickle('weather.pkl')
#process pickle, add day of week and whether it's weekend field (1 or 0)

print(df)


