import pandas as pd
from kreuzbergml.data_quality.data_frame_statistics import DataFrameStatistics
from pathlib import Path

THIS_DIR = Path(__file__).parent


# def debug_duplicates():
#     melb_1000_file_path = THIS_DIR / 'sample_data' / 'melb_1000.csv'
#     df = pd.read_csv(melb_1000_file_path)
#     dq = DataFrameStatistics(df)
#     dq.print_report()
#
#
# def debug_missing():
#     census_1000_file_path = THIS_DIR / 'sample_data' / 'census_1000.csv'
#     df = pd.read_csv(census_1000_file_path)
#     dq = DataFrameStatistics(df)
#     dq.print_report()

#debug_duplicates()
#debug_missing()


timeseries_path = THIS_DIR / 'sample_data' / 'Electric_Production_timeseries.csv'
df = pd.read_csv(timeseries_path)
dq = DataFrameStatistics(df)
dq.print_report()