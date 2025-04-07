import pandas as pd


def parse_log_line(log_line):
    """Parses a log line and returns a dictionary of the different fields."""

    fields = log_line.split(",")
    log_dict = {}
    try:
        log_dict["timestamp"] = fields[0].strip()
        log_dict["component"] = fields[1].strip()
        log_dict["level"] = fields[2].strip()
        log_dict["thread_id"] = fields[3].strip()
        log_dict["message"] = fields[4].strip()
        log_dict["values"] = [i.strip() for i in fields[5:]]
    except IndexError:
        pass

    return log_dict


def parse_log_file(file_name):
    """Parses an entire log file and returns a pandas dataframe with the different columns."""
    f = open(file_name, "r")
    lines = f.readlines()
    print(len(lines))
    log_dicts = []

    for line in lines:
        log_dict = {}
        log_dict = parse_log_line(line)
        log_dicts.append(log_dict)

    df = pd.DataFrame(log_dicts)

    return df
