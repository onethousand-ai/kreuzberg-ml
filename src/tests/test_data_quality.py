from pathlib import Path

import pandas as pd

from kreuzbergml.data_quality.data_frame_statistics import DataFrameStatistics

THIS_DIR = Path(__file__).parent


def test_calc_statistics_dupes():
    census_1000_file_path = THIS_DIR / "sample_data" / "census_1000.csv"
    df = pd.read_csv(census_1000_file_path)
    dq = DataFrameStatistics(df)
    stats_dict = dq.calc_statistics()

    dupes_dict, null_cols = stats_dict["dup_cols"], stats_dict["null_cols"]

    assert len(null_cols) == 0
    num_cols_with_dupes = len(dupes_dict.keys())
    assert num_cols_with_dupes == 1

    col, dupe = list(dupes_dict.keys()), list(dupes_dict.values())

    assert dupe == [["workclass2"]]
    assert col == ["workclass"]


def test_calc_statistics_nulls():
    melb_1000_file_path = THIS_DIR / "sample_data" / "melb_1000.csv"
    df = pd.read_csv(melb_1000_file_path)
    dq = DataFrameStatistics(df)
    stats_dict = dq.calc_statistics()

    dupes_dict, null_cols = stats_dict["dup_cols"], stats_dict["null_cols"]

    num_cols_with_dupes = len(dupes_dict.keys())
    assert num_cols_with_dupes == 0
    assert len(null_cols) == 2

    assert null_cols[0] == "BuildingArea"
    count_0 = dq.count_nulls(null_cols[0])
    percentage_nulls_0 = count_0 / len(dq.df)
    assert round(percentage_nulls_0, 2) == 0.44
    assert count_0 == 437

    assert null_cols[1] == "YearBuilt"
    count_1 = dq.count_nulls(null_cols[1])
    percentage_nulls_1 = count_1 / len(dq.df)
    assert round(percentage_nulls_1, 2) == 0.37
    assert count_1 == 373


def test_calc_statistics_missing_dates():
    timeseries_file_path = (
        THIS_DIR / "sample_data" / "Electric_Production_timeseries.csv"
    )
    df = pd.read_csv(timeseries_file_path, parse_dates=["DATE"], index_col=0)
    dq = DataFrameStatistics(df)
    stats_dict = dq.calc_statistics()
    num_missing_dates, duplicate_rows = (
        stats_dict["missing_dates"],
        stats_dict["dup_rows"],
    )

    assert num_missing_dates == 11658
    assert len(duplicate_rows) == 1
