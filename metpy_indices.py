"""
metpy_indices.py

Computes the same meteorological indices that were used
during CatBoost training.

Input:
    DataFrame returned by fetch_latest.py

Output:
    Dictionary containing all 25 model features.
"""
 
import math 
import warnings

import numpy as np
import pandas as pd

import metpy.calc as mpcalc
from metpy.units import units

warnings.filterwarnings("ignore")

# =====================================================
# Feature Order (Same as Training)
# =====================================================

FEATURE_COLUMNS = [
    "SI",
    "LI",
    "LI_V",
    "SWEAT",
    "KI",
    "CT",
    "VT",
    "TT",
    "CAPE",
    "CAPE_V",
    "CIN",
    "CIN_V",
    "EL",
    "EL_V",
    "LFC",
    "LFC_V",
    "BRN",
    "BRN_V",
    "LCL_T",
    "LCL_P",
    "THETAE_LCL",
    "MML_PT",
    "MML_MR",
    "THK_1000_500",
    "PW"
]


# =====================================================
# Convert MetPy Quantity → float
# =====================================================

def qfloat(value, target_units=None):

    try:

        if isinstance(value, tuple):
            value = value[0]

        if hasattr(value, "to") and target_units:
            value = value.to(target_units)

        if hasattr(value, "magnitude"):
            value = value.magnitude

        value = np.asarray(value, dtype=float)

        if value.size == 0:
            return np.nan

        return float(value.reshape(-1)[0])

    except:

        return np.nan


# =====================================================
# Safe Calculator
# =====================================================

def safe_calc(name, func, units_name=None):

    try:

        return qfloat(func(), units_name)

    except Exception as e:

        print(f"{name} failed : {e}")

        return np.nan


# =====================================================
# Clean Profile
# =====================================================

def clean_profile(df, required_columns):

    cols = ["pressure"] + required_columns

    profile = df[cols].copy()

    profile = profile.replace([np.inf, -np.inf], np.nan)

    profile = profile.dropna()

    profile = profile[profile["pressure"] > 0]

    profile = profile.sort_values("pressure", ascending=False)

    profile = profile.drop_duplicates(subset="pressure")

    profile = profile.reset_index(drop=True)

    return profile


# =====================================================
# Thermodynamic Quantities
# =====================================================

def thermo_quantities(profile):

    pressure = profile["pressure"].values * units.hPa

    temperature = profile["temperature"].values * units.degC

    dewpoint = profile["dewpoint"].values * units.degC

    return pressure, temperature, dewpoint


# =====================================================
# Wind Quantities
# =====================================================

def wind_quantities(profile):

    pressure = profile["pressure"].values * units.hPa

    temperature = profile["temperature"].values * units.degC

    dewpoint = profile["dewpoint"].values * units.degC

    speed = profile["speed"].values * units.knots

    direction = profile["direction"].values * units.degrees

    return (
        pressure,
        temperature,
        dewpoint,
        speed,
        direction
    )

# =====================================================
# Bulk Richardson Number
# =====================================================

def brn_from_profile(profile, use_virtual_temperature=False):

    if len(profile) < 3:
        return np.nan

    try:

        p, t, td, speed, direction = wind_quantities(profile)

        z = profile["height"].values * units.meter

        if use_virtual_temperature:

            env_t = mpcalc.virtual_temperature_from_dewpoint(
                p,
                t,
                td
            ).to("degC")

        else:

            env_t = t

        parcel = mpcalc.parcel_profile(
            p,
            env_t[0],
            td[0]
        )

        cape, cin = mpcalc.cape_cin(
            p,
            env_t,
            td,
            parcel
        )

        u, v = mpcalc.wind_components(speed, direction)

        u_shr, v_shr = mpcalc.bulk_shear(
            pressure=p,
            u=u,
            v=v,
            height=z,
            bottom=z[0],
            depth=6000 * units.meter
        )

        shear2 = (
            u_shr.to("m/s") ** 2 +
            v_shr.to("m/s") ** 2
        )

        denom = 0.5 * shear2

        if qfloat(denom, "meter **2 / second **2") == 0:

            return np.nan

        return qfloat(
            cape / denom,
            "dimensionless"
        )

    except:

        return np.nan


# =====================================================
# Calculate all indices
# =====================================================

def calculate_indices(df):

    values = {
        col: np.nan
        for col in FEATURE_COLUMNS
    }

    thermo = clean_profile(
        df,
        ["temperature", "dewpoint"]
    )

    if len(thermo) >= 3:

        p, t, td = thermo_quantities(thermo)

        try:

            parcel = mpcalc.parcel_profile(
                p,
                t[0],
                td[0]
            )

        except:

            parcel = None

        if parcel is not None:

            values["SI"] = safe_calc(
                "SI",
                lambda:
                mpcalc.showalter_index(
                    p,
                    t,
                    td
                ),
                "delta_degC"
            )

            values["LI"] = safe_calc(
                "LI",
                lambda:
                mpcalc.lifted_index(
                    p,
                    t,
                    parcel
                ),
                "delta_degC"
            )

            values["KI"] = safe_calc(
                "KI",
                lambda:
                mpcalc.k_index(
                    p,
                    t,
                    td
                ),
                "degC"
            )

            values["CT"] = safe_calc(
                "CT",
                lambda:
                mpcalc.cross_totals(
                    p,
                    t,
                    td
                ),
                "delta_degC"
            )

            values["VT"] = safe_calc(
                "VT",
                lambda:
                mpcalc.vertical_totals(
                    p,
                    t
                ),
                "delta_degC"
            )

            values["TT"] = safe_calc(
                "TT",
                lambda:
                mpcalc.total_totals_index(
                    p,
                    t,
                    td
                ),
                "delta_degC"
            )

            values["CAPE"] = safe_calc(
                "CAPE",
                lambda:
                mpcalc.cape_cin(
                    p,
                    t,
                    td,
                    parcel
                )[0],
                "J/kg"
            )

            values["CIN"] = safe_calc(
                "CIN",
                lambda:
                mpcalc.cape_cin(
                    p,
                    t,
                    td,
                    parcel
                )[1],
                "J/kg"
            )

            values["LFC"] = safe_calc(
                "LFC",
                lambda:
                mpcalc.lfc(
                    p,
                    t,
                    td,
                    parcel
                )[0],
                "hPa"
            )

            values["EL"] = safe_calc(
                "EL",
                lambda:
                mpcalc.el(
                    p,
                    t,
                    td,
                    parcel
                )[0],
                "hPa"
            )

                        # ----------------------------------------
            # Virtual Temperature Calculations
            # ----------------------------------------

            try:

                tv = mpcalc.virtual_temperature_from_dewpoint(
                    p,
                    t,
                    td
                ).to("degC")

            except:

                tv = None

            if tv is not None:

                try:

                    parcel_v = mpcalc.parcel_profile(
                        p,
                        tv[0],
                        td[0]
                    )

                except:

                    parcel_v = None

                if parcel_v is not None:

                    values["LI_V"] = safe_calc(
                        "LI_V",
                        lambda:
                        mpcalc.lifted_index(
                            p,
                            tv,
                            parcel_v
                        ),
                        "delta_degC"
                    )

                    values["CAPE_V"] = safe_calc(
                        "CAPE_V",
                        lambda:
                        mpcalc.cape_cin(
                            p,
                            tv,
                            td,
                            parcel_v
                        )[0],
                        "J/kg"
                    )

                    values["CIN_V"] = safe_calc(
                        "CIN_V",
                        lambda:
                        mpcalc.cape_cin(
                            p,
                            tv,
                            td,
                            parcel_v
                        )[1],
                        "J/kg"
                    )

                    values["LFC_V"] = safe_calc(
                        "LFC_V",
                        lambda:
                        mpcalc.lfc(
                            p,
                            tv,
                            td,
                            parcel_v
                        )[0],
                        "hPa"
                    )

                    values["EL_V"] = safe_calc(
                        "EL_V",
                        lambda:
                        mpcalc.el(
                            p,
                            tv,
                            td,
                            parcel_v
                        )[0],
                        "hPa"
                    )

            # ----------------------------------------
            # LCL
            # ----------------------------------------

            try:

                lcl_p, lcl_t = mpcalc.lcl(
                    p[0],
                    t[0],
                    td[0]
                )

                values["LCL_P"] = qfloat(
                    lcl_p,
                    "hPa"
                )

                values["LCL_T"] = qfloat(
                    lcl_t,
                    "K"
                )

                values["THETAE_LCL"] = qfloat(
                    mpcalc.equivalent_potential_temperature(
                        lcl_p,
                        lcl_t,
                        lcl_t
                    ),
                    "K"
                )

            except:

                pass

            # ----------------------------------------
            # Precipitable Water
            # ----------------------------------------

            values["PW"] = safe_calc(
                "PW",
                lambda:
                mpcalc.precipitable_water(
                    p,
                    td
                ),
                "mm"
            )

            # ----------------------------------------
            # Thickness
            # ----------------------------------------

            bottom_pressure = min(
    1000 * units.hPa,
    p[0]
)

            values["THK_1000_500"] = safe_calc(
    "THK_1000_500",
    lambda: mpcalc.thickness_hydrostatic(
        p,
        t,
        bottom=bottom_pressure,
        depth=500 * units.hPa
    ),
    "meter"
)

            # ----------------------------------------
            # Mixed Layer
            # ----------------------------------------

            try:

                theta = mpcalc.potential_temperature(
                    p,
                    t
                )

                e = mpcalc.saturation_vapor_pressure(
                    td
                )

                mr = mpcalc.mixing_ratio(
                    e,
                    p
                )

                mean_theta = mpcalc.mixed_layer(
                    p,
                    theta,
                    bottom=p[0],
                    depth=100 * units.hPa
                )[0]

                mean_mr = mpcalc.mixed_layer(
                    p,
                    mr,
                    bottom=p[0],
                    depth=100 * units.hPa
                )[0]

                values["MML_PT"] = qfloat(
                    mean_theta,
                    "K"
                )

                values["MML_MR"] = qfloat(
                    mean_mr,
                    "g/kg"
                )

            except:

                pass

    # ----------------------------------------
    # SWEAT
    # ----------------------------------------

    wind_profile = clean_profile(
        df,
        [
            "temperature",
            "dewpoint",
            "direction",
            "speed"
        ]
    )

    if len(wind_profile) >= 3:

        p_w, t_w, td_w, speed, direction = wind_quantities(
            wind_profile
        )

        values["SWEAT"] = safe_calc(
            "SWEAT",
            lambda:
            mpcalc.sweat_index(
                p_w,
                t_w,
                td_w,
                speed,
                direction
            ),
            "dimensionless"
        )

    # ----------------------------------------
    # BRN
    # ----------------------------------------

    brn_profile = clean_profile(
        df,
        [
            "temperature",
            "dewpoint",
            "height",
            "direction",
            "speed"
        ]
    )

    values["BRN"] = brn_from_profile(
        brn_profile,
        False
    )

    values["BRN_V"] = brn_from_profile(
        brn_profile,
        True
    )

    return values

# =====================================================
# Convert Dictionary → DataFrame
# =====================================================

def get_feature_dataframe(indices):
    """
    Converts calculated feature dictionary into a
    single-row DataFrame in the exact order expected
    by the CatBoost model.
    """

    row = []

    for feature in FEATURE_COLUMNS:
        row.append(indices.get(feature, np.nan))

    feature_df = pd.DataFrame(
        [row],
        columns=FEATURE_COLUMNS
    )

    return feature_df


# =====================================================
# Wrapper Function
# =====================================================

def process_sounding(df):
    """
    Main function used by predict.py

    Parameters
    ----------
    df : DataFrame
        Sounding DataFrame returned by fetch_latest.py

    Returns
    -------
    feature_df : DataFrame
        Single-row DataFrame containing the
        25 features in training order.

    indices : dict
        Dictionary of all calculated indices.
    """

    indices = calculate_indices(df)

    feature_df = get_feature_dataframe(indices)

    return feature_df, indices


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    from fetch_latest import fetch_latest

    print("=" * 60)
    print("Downloading latest sounding...")
    print("=" * 60)

    sounding = fetch_latest()

    # fetch_latest.py returns a dictionary
    # containing station info and DataFrame

    df = sounding["data"]

    feature_df, indices = process_sounding(df)

    print("\nCalculated Features\n")

    for key in FEATURE_COLUMNS:

        print(f"{key:15s} : {indices[key]}")

    print("\nFeature DataFrame\n")

    print(feature_df)
