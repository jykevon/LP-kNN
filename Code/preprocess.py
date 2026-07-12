import pandas as pd
import requests
import time

INPUT_CSV = "final.csv"
OUTPUT_CSV = "final1.csv"

HEADERS = {
    "User-Agent": "ML-County-Geocoder/1.0 (student project)"
}

def geocode_county(county, state):
    query = f"{county} County, {state}, USA"
    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": query,
        "format": "json",
        "limit": 1
    }

    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
    except Exception as e:
        print(f"Request failed for {county}, {state}: {e}")
        return None, None

    if r.status_code != 200:
        print(f"HTTP {r.status_code} for {county}, {state}")
        return None, None

    data = r.json()

    if len(data) == 0:
        print(f"No results for {county}, {state}")
        return None, None

    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    return lat, lon

def main():
    # df = pd.read_csv(INPUT_CSV)

    # latitudes = []
    # longitudes = []

    # for i, row in df.iterrows():
    #     county = row["county"]
    #     state = row["state"]

    #     print(f"Geocoding {county}, {state} ({i+1}/{len(df)})")

    #     lat, lon = geocode_county(county, state)
    #     latitudes.append(lat)
    #     longitudes.append(lon)

    #     # REQUIRED by Nominatim usage policy
    #     time.sleep(1)

    # df["latitude"] = latitudes
    # df["longitude"] = longitudes

    # df.to_csv(OUTPUT_CSV, index=False)

    # Read CSV
    df = pd.read_csv(INPUT_CSV)

    # missing_trump = df[df["trump16"].isna()]
    # missing_clinton = df[df["clinton16"].isna()]

    # print("Counties with missing Trump votes:", len(missing_trump))
    # print(missing_trump[["state", "county", "trump16"]])

    # print("\nCounties with missing Clinton votes:", len(missing_clinton))
    # print(missing_clinton[["state", "county", "clinton16"]])

    # # Create alignment based on 2016 presidential election
    # def determine_alignment(row):
    #     if row["clinton16"] > row["trump16"]:
    #         return "democrat"
    #     elif row["trump16"] > row["clinton16"]:
    #         return "republican"
    #     else:
    #         return "other"

    # df["alignment"] = df.apply(determine_alignment, axis=1)

    # Keep only required columns
    df_out = df[["longitude", "latitude", "alignment"]]

    # Save to new CSV
    df_out.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    main()