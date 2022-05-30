import pandas as pd
from kreuzbergml.data_quality.DataFrameStatistics import DataFrameStatistics
from pathlib import Path
import sys

sys.path.append("../kreuzbergml/data_quality")


THIS_DIR = Path(__file__).parent


def test_duplicates():
    melb_1000_file_path = THIS_DIR / 'sample_data/melb_1000.csv'
    df = pd.read_csv(melb_1000_file_path)
    dq = DataFrameStatistics(df)
    dq.print_report()


def test_missing():
    census_1000_file_path = THIS_DIR / 'sample_data/census_1000.csv'
    df = pd.read_csv(census_1000_file_path)
    dq = DataFrameStatistics(df)
    dq.print_report()

# test_duplicates()
# test_missing()
