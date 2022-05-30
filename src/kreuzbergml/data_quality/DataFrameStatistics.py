from typing import List, Optional, Union
from pandas import DataFrame, Series, Index, DatetimeIndex, PeriodIndex, TimedeltaIndex
from enum import Enum, auto
from logging import getLogger

logger = getLogger(__name__)


class DataFrameType(Enum):
    """
    Supported dataframe types.
    """
    TABULAR = auto()
    TIMESERIES = auto()


class DataFrameStatistics:

    def __init__(self,
                 df: DataFrame,
                 random_state: Optional[int] = None,
                 dtypes: dict = None,
                 # severity: Optional[str] = None
                 ):
        self._df = df
        # self._df_type = None
        # self._warnings = []
        # self._dtypes = dtypes

    @property
    def df(self):
        "Target of data quality checks."
        return self._df

    def _count_nulls(self, col: Union[List[str], str, None] = None):#, perc=False):
        """

        :param col: column name if provided only counts null in that column
        :return:
            count of null values,
            perc of null values
        """

        count = self.df.isnull().sum() if col is None else self.df[col].isnull().sum()
        # null_values_percentage = count / len(self.df)

        return count #, null_values_percentage

    def get_null_cols(self, col: Optional[str] = None) -> List[str]:
        """

        :param col: if given, checks if  col(s) has nulls
        :return:
            list of given column(s) or all columns with null values in DataFrame if None."
        """
        return list(self.df.columns[self._count_nulls() > 0]) if col is None \
            else col if isinstance(col, list) \
            else [col]

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

    def calc_statistics(self):
        dupes_dict = self.get_duplicate_columns()
        cols_with_dupes = len(dupes_dict.keys())
        null_cols = self.get_null_cols()
        return dupes_dict, cols_with_dupes, null_cols

    def print_report(self):
        """
        Prints a report containing all the warnings detected during the data quality analysis.
        """
        dupes_dict, cols_with_dupes, null_cols = self.calc_statistics()

        print(f"\n\nDATA QUALITY REPORT\n")
        if cols_with_dupes > 0:
            print(f"Found {cols_with_dupes} columns with exactly the same feature values as other columns.")
            for col, dupe in dupes_dict.items():
                print(f"Columns {dupe} is/are duplicate(s) of Column '{col}'")
        else:
            print("No duplicate columns were found.")

        if len(null_cols) > 0:
            print(f"The following columns have missing values:")
            for col in null_cols:
                count = self._count_nulls(col)
                percentage_nulls = count / len(self.df)

                print(f"Column '{col}' has {count} missing values which comprise {round(percentage_nulls, 2)}% of all rows")
        else:
            print(f"No missing values were found")

    # def check_time_index(self, index: Index) -> bool:
    #     """Tries to infer from passed index column if the dataframe is a timeseries or not."""
    #     if isinstance(index, (DatetimeIndex, PeriodIndex, TimedeltaIndex)):
    #         return True
    #     return False
    #
    # def infer_df_type(self, df: DataFrame) -> DataFrameType:
    #     if self.check_time_index(df.index):
    #         return DataFrameType.TIMESERIES
    #     return DataFrameType.TABULAR
