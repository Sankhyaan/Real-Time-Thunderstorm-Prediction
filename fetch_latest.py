"""
fetch_latest.py

Downloads upper-air sounding data from the
University of Wyoming using Siphon.

Station:
43150 = Visakhapatnam

Author: Thunderstorm Predictor
"""

from datetime import datetime, timedelta
from siphon.simplewebservice.wyoming import WyomingUpperAir


DEFAULT_STATION = "43150"


def fetch_sounding(date, station=DEFAULT_STATION):
    """
    Download sounding for a specific UTC datetime.

    Parameters
    ----------
    date : datetime
        Datetime object (00 or 12 UTC)

    station : str
        WMO station number

    Returns
    -------
    dict
    """

    df = WyomingUpperAir.request_data(date, station)

    return {
        "station": station,
        "station_name": "Visakhapatnam",
        "sounding_time": date,
        "data": df
    }

from datetime import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir


def fetch_latest(station=DEFAULT_STATION):
    """
    Download latest available sounding.
    """

    now = datetime.utcnow()

    hour = 12 if now.hour >= 12 else 0

    current = now.replace(
        hour=hour,
        minute=0,
        second=0,
        microsecond=0
    )

    for _ in range(20):

        try:

            return fetch_sounding(current, station)

        except Exception:

            current -= timedelta(hours=12)

    raise Exception("No recent sounding available.")

from siphon.simplewebservice.wyoming import WyomingUpperAir


def fetch_historical(date, station="43150"):
    """
    Fetch a historical sounding.
    """

    try:

        df = WyomingUpperAir.request_data(date, station)

        return {
            "station": station,
            "station_name": "Visakhapatnam",
            "sounding_time": date,
            "data": df
        }

    except Exception:

        return None


def user_input():

    print("\nUniversity of Wyoming Sounding Downloader")
    print("-----------------------------------------")
    print("1. Latest Sounding")
    print("2. Specific Date")

    choice = input("\nEnter choice : ").strip()

    if choice == "1":

        return fetch_latest()

    elif choice == "2":

        date_str = input("\nEnter Date (YYYY-MM-DD): ").strip()

        hour = int(input("Enter Hour (0 or 12 UTC): "))

        dt = datetime.strptime(date_str, "%Y-%m-%d")
        dt = dt.replace(hour=hour)

        return fetch_sounding(dt)

    else:

        raise ValueError("Invalid choice")


if __name__ == "__main__":

    sounding = user_input()

    df = sounding["data"]

    print("\n----------------------------------------")
    print("Station :", sounding["station_name"])
    print("Station ID :", sounding["station"])
    print("Requested Sounding :", sounding["sounding_time"])
    print("----------------------------------------")

    print("\nAvailable Columns\n")

    print(df.columns.tolist())

    print("\nColumn Data Types\n")
    print(df.dtypes)

    print("\nFirst Five Rows\n")

    print(df)

    filename = sounding["sounding_time"].strftime("%Y%m%d_%HZ")

    csv_name = f"Sounding_{filename}.csv"

    df.to_csv(csv_name, index=False)

    print(f"\nCSV Saved : {csv_name}")
