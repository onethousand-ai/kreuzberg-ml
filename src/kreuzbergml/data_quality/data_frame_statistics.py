from logging import getLogger
from typing import List, Optional, Union

import numpy as np
import seaborn as sns
from pandas import (DataFrame, DatetimeIndex, Index, PeriodIndex, date_range,
                    period_range)

logger = getLogger(__name__)


class DataFrameStatistics:

    def __init__(self,
                 df: DataFrame
                 ):
        self._df = df

        if isinstance(self._df.index, DatetimeIndex):
            self._df_type = "time"
        elif isinstance(self._df.index, PeriodIndex):
            self._df_type = "period"
        else:
            self._df_type = "tabular"

    @property
    def df(self):
        return self._df

    def count_nulls(self, col: Union[List[str], str, None] = None):
        """
        :param col: column name if provided only counts null in that column
        :return:
            count of null values,
        """
        count = self.df.isnull().sum() if col is None else self.df[col].isnull().sum()
        return count

    def get_null_cols(self, col: Optional[str] = None) -> List[str]:
        """
        :param col: if given, checks if  col(s) has nulls
        :return:
            list of given column(s) or all columns with null values in DataFrame if None."
        """
        if col is None:
            null_cols = list(self.df.columns[self.count_nulls() > 0])
        elif isinstance(col, list):
            null_cols = col
        else:
            null_cols = [col]
        return null_cols

    def get_duplicate_columns(self):
        """
        :return:
            Returns a mapping dictionary of columns with fully duplicated feature values
        """
        dupes = {}
        for idx, col in enumerate(self.df.columns):  # Iterate through all the columns of dataframe
            ref = self.df[col]  # Take the column values as reference.
            for tgt_col in self.df.columns[idx + 1:]:  # Iterate through all other columns
                if ref.equals(self.df[tgt_col]):  # Take target values
                    dupes.setdefault(col, []).append(tgt_col)  # Store if they match
        return dupes

    def get_duplicate_rows(self):
        """
        :return:
            Returns a mapping dictionary of rows with fully duplicated feature values
        """
        df_dupes = self.df[self.df.duplicated(keep=False)]
        dupes = {}
        skip_idx = [] # already identified dupes
        for idx in range(len(df_dupes)):  # Iterate through all the rows of dataframe
            if idx not in skip_idx:
                ref = df_dupes.iloc[idx]  # Take row as reference.
                for tgt_row_idx in range(idx+1, len(df_dupes)):  # Iterate through all other rows
                    tgt_row = df_dupes.iloc[tgt_row_idx]
                    if ref.equals(tgt_row):  # Take target values
                        try:
                            dupes[ref.name].append(tgt_row.name)  # Store if they match
                        except:
                            dupes[ref.name] = [tgt_row.name]
                        skip_idx.append(tgt_row_idx)
        return dupes

    def get_missing_indices(self):
        """
        :return:
            Returns Index with elements that are not in the dataframe index
        """
        min_date = self.df.index.min()
        max_date = self.df.index.max()
        freq = self.df.index.freq

        if self._df_type == "period":
            missing_dates = period_range(start=min_date, end=max_date, freq=freq).difference(self.df.index)
        else:
            missing_dates = date_range(start=min_date, end=max_date, freq=freq).difference(self.df.index)

        return missing_dates

    def get_correlated_columns(self, correlation_value: float = 0.95):
        """
        :param correlation_value: the function returns all columns with a higher correlation than this value
        :return:
            Returns pairs of highly correlated columns (only for numerical columns)
        """
        cor_matrix = self.df.corr().abs() # correlation matrix with positiv values
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool)) # only use the upper half because of symmetry
        high_corr = np.dstack(np.where(upper_tri>correlation_value))[0] # filter all indices of the columns that are higher than 'correlation_value' correlated
        col_names = np.array(np.take(upper_tri.columns, high_corr)) # use multi-dimensional indexing to get the names of the correlated columns
        return col_names

    def calc_statistics(self):

        duplicate_cols_dict = self.get_duplicate_columns()
        duplicate_rows_dict = self.get_duplicate_rows()
        null_cols = self.get_null_cols()
        corr_cols = self.get_correlated_columns()
        num_missing_dates = None

        if self._df_type in ["time", "period"]:
            num_missing_dates = len(self.get_missing_indices())

        return {"missing_dates": num_missing_dates, "dup_cols": duplicate_cols_dict, "dup_rows": duplicate_rows_dict,  "null_cols": null_cols, "corr_cols": corr_cols}

    def print_report(self):
        """
        Prints a string report containing all the problems detected during the analysis.
        """

        print(f"\n\nDATA QUALITY REPORT\n")
        
        stats_dict = self.calc_statistics()

        ### MISSING DATES
        if self._df_type in ["time", "period"]:
            num_missing_dates = stats_dict['missing_dates']
            print(f"Found {num_missing_dates} missing dates in the timeseries index")

        ### DUPLICATE COLS
        duplicate_cols_dict = stats_dict['dup_cols']
        num_cols_with_dupes = len(duplicate_cols_dict.keys())
        if num_cols_with_dupes > 0:
            print(f"Found {num_cols_with_dupes} columns with exactly the same feature values as other columns.")
            for col, dupe in duplicate_cols_dict.items():
                print(f"Columns {dupe} is/are duplicate(s) of Column '{col}'")
        else:
            print("No duplicated columns were found.")
        
        ### DUPLICATE ROWS
        duplicate_rows_dict = stats_dict['dup_rows']
        num_rows_with_dupes = len(duplicate_rows_dict.keys())
        if num_rows_with_dupes > 0:
            print(f"Found {num_rows_with_dupes} rows with exactly the same feature values as other rows.")
            for row, dupe in duplicate_rows_dict.items():
                print(f"Rows {dupe} is/are duplicate(s) of row: {row}")
        else:
            print("No duplicated rows were found.")

        ### NULL COLS
        null_cols = stats_dict['null_cols']
        if len(null_cols) > 0:
            print(f"The following columns have NaN values:")
            for col in null_cols:
                count = self.count_nulls(col)
                percentage_nulls = count / len(self.df)
                print(f"Column '{col}' has {count} NaN values which comprise {percentage_nulls:.2f} of all rows")
        else:
            print(f"No NaN values were found")

        ### CORRELATED COLS
        corr_cols = stats_dict['corr_cols']
        if len(corr_cols) > 0:
            print(f"Found {len(corr_cols)} columns that are highly correlated (>95%) from another column.")
            if len(corr_cols) < 0:
                for col_pair in corr_cols:
                    print(f"Column '{col_pair[0]}' is highly correlated with '{col_pair[1]}'")
            else:
                print("Too many columns too show individualy --> Heatmap:")
                sns.heatmap(self.df.corr().abs(), annot=True, cmap="BuPu")
        else:
            print(f"No highly correlated columns found.")

