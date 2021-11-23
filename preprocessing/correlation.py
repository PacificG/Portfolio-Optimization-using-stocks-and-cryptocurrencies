import argparse
import pandas as pd
import numpy as np

def correlation(csv_path, num_stocks):
    df = pd.read_csv(csv_path)
    print(df.head())
    corr_table = df[['date', 'close', ' Name']]
    corr_table.set_index('date', inplace=True)
    corr_table = corr_table.pivot_table(index='date', columns=' Name', values='close')
    corr_table = corr_table.corr()
    corr_table['stock1'] = corr_table.index
    corr_table = corr_table.melt(id_vars = 'stock1', var_name = "stock2").reset_index(drop = True)
    corr_table = corr_table[corr_table['stock1'] < corr_table['stock2']].dropna()
    corr_table['abs_value'] = np.abs(corr_table['value'])
    highest_corr = corr_table.sort_values("abs_value", ascending = True).head(num_stocks)
    return list(set(list(highest_corr.stock1) + list(highest_corr.stock2)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Correlation')
    parser.add_argument('-i', '--input', help='CSV file containing stocks data of companies', \
        default='utils/datasets/all_stocks_5yr.csv', type=str)
    parser.add_argument('-n', '--Num_Stocks', help='Number of stocks needed to be correlated', default=20, type=int)
    parser.add_argument('-w', '--window_length', help='Window length', default=20, type=int)

    args = parser.parse_args()

    print(correlation(args.input, args.Num_Stocks))
    
    


