import polars as pl


def log_results():
    import datetime as dt
    import os

    file_dir = os.getcwd()
    data_dir = "data"
    res_dir = "results"
    datetime = dt.datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    filename = f"train_results_{datetime}.csv"
    file_path = os.path.join(file_dir, data_dir, res_dir, filename)
    df = pl.DataFrame(
        {
            "time": [],
            "wait_time": [],
            "actions": [],
            "move_count": [],
            "hit": [],
            "miss": [],
            "score": [],
            "timestamp": [],
        }
    )
    df.write_csv(file_path, separator=",", line_terminator="\n")


if __name__ == "__main__":
    log_results()
