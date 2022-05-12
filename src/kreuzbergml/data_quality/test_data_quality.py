import sys
sys.path.append(".")
from data_quality import DataQuality
import pandas as pd


def test_duplicates():
    df = pd.read_csv('../../../src/tests/sample_data/melb_1000.csv')
    dq = DataQuality(df)
    dq.report()

def test_missing():
    df = pd.read_csv('../../../src/tests/sample_data/census_1000.csv')
    dq = DataQuality(df)
    dq.report()


test_duplicates()
test_missing()







