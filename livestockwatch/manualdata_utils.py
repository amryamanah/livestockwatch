import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.signal import savgol_filter
import pymongo

from .config import MONGO_URL, DCT_CATTLE_ID
from .pupil_analysis import EllipseAnalysis, ellipse_calculate_ca

from IPython import embed


class DataProvider:

    def __init__(self, excel_result_path):
        self.result_writer = pd.ExcelWriter(excel_result_path)
        self.connection = pymongo.MongoClient(MONGO_URL)
        self.vita_df = self._get_vita()

    def calculate_area(self, row):
        ea = EllipseAnalysis(row.pupil_major_axis, row.pupil_minor_axis)
        return ea.area

    def calculate_ipr(self, row):
        ea = EllipseAnalysis(row.pupil_major_axis, row.pupil_minor_axis)
        return ea.ipr

    def get_datetime(self, dt):
        dt_parts = [int(s) for s in dt.split("_") if s.isdigit()]
        return datetime(dt_parts[0], dt_parts[1], dt_parts[2])

    def calculate_same_region(self, row, firstregion):
        if row.region == firstregion:
            return True
        else:
            return False

    def update_area_interp(self, grp):
        lst_data_interp = grp.area.interpolate(method="linear", limit_direction="both")
        lst_data_savgol = savgol_filter(lst_data_interp, window_length=9, polyorder=4, mode="mirror")
        grp["svg_area"] = lst_data_savgol
        return grp

    def update_ipr_interp(self, grp):
        lst_data_interp = grp.ipr.interpolate(method="linear", limit_direction="both")
        lst_data_savgol = savgol_filter(lst_data_interp, window_length=9, polyorder=4, mode="mirror")
        grp["svg_ipr"] = lst_data_savgol
        return grp

    def update_vita_value(self, row):
        newdf = self.vita_df[(self.vita_df.datetaken == row.datetaken) & (self.vita_df.cattle_id == row.cattle_id)]
        if newdf.empty:
            return np.nan
        else:
            return newdf.iloc[0]["vit_a"]

    def process_mongo_ipr(self, db_name):
        db = self.connection[db_name]
        livestockwatch_collection = db["capture_session"]

        projection = {
            "_id": False,
            "eccentricity": False,
            "svg_eccentricity": False,
            "svg_max_area": False,
            "svg_perimeter": False,
            "perimeter": False,
            "ca": False,
            "svg_pupil_major_axis": False,
            "svg_pupil_minor_axis": False,
            "centroid": False
        }

        firstframe_cursor = livestockwatch_collection.find({"firstframe": True}, projection=projection)

        firstframe_dict = (firstframe for firstframe in firstframe_cursor)
        firstframe_df = pd.DataFrame(firstframe_dict)

        firstframe_df["datetaken"] = firstframe_df.cs_name.apply(lambda dt: self.get_datetime(dt))
        firstframe_df["cattle_id"] = firstframe_df.cattle_id.astype(str)

        wa02_cattles = DCT_CATTLE_ID["WA02"]
        firstframe_df = firstframe_df[firstframe_df.cattle_id.isin(wa02_cattles)]

        return firstframe_df

    def process_mongo_ca(self, db_name, plr_collection, version="new"):
        db = self.connection[db_name]
        plr_collection = db[plr_collection]

        if version == "new":
            projection = {
                "_id": False,
                "eccentricity": False,
                "svg_eccentricity": False,
                "svg_max_area": False,
                "svg_perimeter": False,
                "perimeter": False,
                "ca": False,
                "svg_pupil_major_axis": False,
                "svg_pupil_minor_axis": False,
                "centroid": False
            }
        elif version == "old":
            projection = {
                "_id": False,
                "eccentricity": False,
                "svg_eccentricity": False,
                "svg_max_area": False,
                "svg_perimeter": False,
                "perimeter": False,
                "ca": False,
                "svg_pupil_major_axis": False,
                "svg_pupil_minor_axis": False,
                'aspect_ratio': False,
                'contour_area': False,
                'contour_perimeter': False,
                'extent': False,
                'framecount': False,
                'is_valid': False,
                'solidity': False,
                'vit_a': False
            }

        plr_cursor = plr_collection.find({}, projection=projection).sort([("cs_name", pymongo.ASCENDING), ("frametime", pymongo.ASCENDING)])
        plr_dict = (plr for plr in plr_cursor)
        plr_df = pd.DataFrame(plr_dict)

        plr_df["datetaken"] = plr_df.cs_name.apply(lambda dt: self.get_datetime(dt))
        plr_df["cattle_id"] = plr_df.cattle_id.astype(str)

        wa02_cattles = DCT_CATTLE_ID["WA02"]
        plr_df = plr_df[plr_df.cattle_id.isin(wa02_cattles)]

        return plr_df

    def _get_vita(self):
        db = self.connection.archive_livestockwatch
        bd_collection = db.blood_data
        bd_cursor = bd_collection.find({},projection={
                "_id": False,
                "beta_caroten": False,
                "vit_e": False
        }).sort("datetaken",pymongo.ASCENDING)
        bd_dict = [cs for cs in bd_cursor]
        bd_df = pd.DataFrame(bd_dict)

        wa02_cattles = DCT_CATTLE_ID["WA02"]

        bd_df = bd_df[bd_df.cattle_id.isin(wa02_cattles)]
        bd_df.sort_values(by="datetaken")

        lst_cattle_id = bd_df.cattle_id.unique()

        date_x = np.arange(datetime(2014, 12, 1),datetime(2015,12,27), timedelta(days=1)).astype(datetime)

        lst_dict = []
        for cattle_id in lst_cattle_id:
            vita_series = pd.Series(bd_df[bd_df.cattle_id == cattle_id].set_index("datetaken").to_dict()["vit_a"],
                                     index=date_x)
            vita_interp = vita_series.interpolate(method="linear",limit_direction="both").bfill()
            for datestamp in vita_interp.index:
                dct_data = {
                    "cattle_id": int(cattle_id),
                    "vit_a": vita_interp[datestamp],
                    "datetaken": datestamp
                }
                lst_dict.append(dct_data)

        vita_df = pd.DataFrame(lst_dict)
        vita_df.cattle_id = vita_df.cattle_id.astype(str)

        return vita_df

    def save_and_process_vita(self, db_name, collection_name):
        db = self.connection[db_name]
        bd_collection = db[collection_name]
        bd_cursor = bd_collection.find({}, projection={
                "_id": False,
                "beta_caroten": False,
                "vit_e": False
        }).sort("datetaken",pymongo.ASCENDING)
        bd_dict = [cs for cs in bd_cursor]
        bd_df = pd.DataFrame(bd_dict)

        wa02_cattles = DCT_CATTLE_ID["WA02"]

        bd_df = bd_df[bd_df.cattle_id.isin(wa02_cattles)]
        bd_df.sort_values(by="datetaken")

        bd_all = bd_df.groupby("datetaken").mean()
        bd_all.to_excel(self.result_writer, sheet_name="bd_all")

        grouped = bd_df.groupby("cattle_id")
        for name, group in grouped:
            group.index = group.datetaken
            group = group.drop("datetaken", 1)
            group.sort_index(inplace=True)
            group.to_excel(self.result_writer, sheet_name="bd_{}".format(name))

        self.result_writer.save()

    def process_excel_ipr(self, excel_path):
        excel_handler = pd.ExcelFile(excel_path)
        df_dct = pd.read_excel(excel_handler, sheetname=None)
        lst_plr_df = []
        lst_cs_df = []
        sec_diff = 0.1
        for k, df in df_dct.items():
            print(k)
            if not str(k) == "1111111111":
                grouped = df.groupby("cs_name")
                for name, group in grouped:
                    print(name)
                    group = group.dropna(subset=['pupil_major_axis', 'pupil_minor_axis'])
                    group["area"] = group.apply(lambda row: self.calculate_area(row), axis=1)
                    group["ipr"] = group.apply(lambda row: self.calculate_ipr(row), axis=1)
                    group["datetaken"] = group.cs_name.apply(lambda dt: self.get_datetime(dt))
                    group['timestamp'] = pd.to_datetime(group['timestamp'])
                    group["cattle_id"] = group.cattle_id.astype(str)

                    try:
                        group = group.drop("filename", 1)
                    except ValueError as e:
                        print(e)
                        pass

                    first_frametime = np.around([group.frametime[group.firstframe == True].iloc[0]], decimals=2)[0]
                    first_region = int(group.region[group.firstframe == True].iloc[0])
                    group["same_region"] = group.apply(lambda row: self.calculate_same_region(row, first_region),
                                                       axis=1)

                    if group.frametime.max() >= first_frametime:
                        lst_frametime = [a for a in
                                         np.arange(first_frametime, max(group.frametime) + sec_diff, sec_diff)
                                         if not np.isclose(group.frametime, a).any()]

                        lst_frametime = np.around(lst_frametime, decimals=2).tolist()

                        base_dct = group.iloc[0].to_dict()
                        base_dct["angle"] = np.nan
                        base_dct["pupil_major_axis"] = np.nan
                        base_dct["pupil_minor_axis"] = np.nan
                        base_dct["area"] = np.nan
                        base_dct["ipr"] = np.nan
                        base_dct["firstframe"] = np.bool(False)

                        for frametime in lst_frametime:
                            res_dct = base_dct.copy()
                            res_dct["frametime"] = frametime
                            group = group.append(res_dct, ignore_index=True)

                        group.sort_values(by="frametime", inplace=True)

                        group["max_area"] = group["area"].max()
                        group = self.update_ipr_interp(group)
                        group = self.update_area_interp(group)
                        group["svg_max_area"] = group["svg_area"].max()

                        group.reset_index(inplace=True)
                        group = group.drop("index", 1)

                        # old
                        # group["svg_ca"] = (group.max_area - group.svg_area) / group.max_area

                        first_svg_area = group.svg_area[group.firstframe == True].iloc[0]
                        # group["svg_ca"] = ellipse_calculate_ca(group.svg_area, first_svg_area)
                        group["svg_ca"] = (first_svg_area - group.svg_area) / first_svg_area

                        plr_df = group[group.firstframe == True]
                        if not plr_df.empty:
                            lst_plr_df.append(plr_df)
                            lst_cs_df.append(group)

        plr_df = pd.concat(lst_plr_df).reset_index()
        plr_df = plr_df.drop("index", 1)

        cs_df = pd.concat(lst_cs_df).reset_index()
        cs_df = cs_df.drop("index", 1)

        return plr_df, cs_df

    def process_excel_ca(self, excel_path):
        excel_handler = pd.ExcelFile(excel_path)
        df_dct = pd.read_excel(excel_handler, sheetname=None)
        lst_plr_df = []
        lst_cs_df = []
        sec_diff = 0.1
        for k, df in df_dct.items():
            print(k)
            if not str(k) == "1111111111":
                grouped = df.groupby("cs_name")
                for name, group in grouped:
                    print(name)
                    group = group.dropna(subset=['pupil_major_axis', 'pupil_minor_axis'])
                    group["area"] = group.apply(lambda row: self.calculate_area(row), axis=1)
                    group["ipr"] = group.apply(lambda row: self.calculate_ipr(row), axis=1)
                    group["datetaken"] = group.cs_name.apply(lambda dt: self.get_datetime(dt))
                    group['timestamp'] = pd.to_datetime(group['timestamp'])
                    group["cattle_id"] = group.cattle_id.astype(str)

                    try:
                        group = group.drop("filename", 1)
                    except ValueError as e:
                        print(e)
                        pass

                    first_frametime = np.around([group.frametime[group.firstframe == True].iloc[0]], decimals=2)[0]
                    first_region = int(group.region[group.firstframe == True].iloc[0])
                    group["same_region"] = group.apply(lambda row: self.calculate_same_region(row, first_region), axis=1)

                    if group.frametime.max() >= first_frametime:
                        lst_frametime = [a for a in np.arange(first_frametime, max(group.frametime) + sec_diff,  sec_diff)
                                         if not np.isclose(group.frametime, a).any()]

                        lst_frametime = np.around(lst_frametime, decimals=2).tolist()

                        base_dct = group.iloc[0].to_dict()
                        base_dct["angle"] = np.nan
                        base_dct["pupil_major_axis"] = np.nan
                        base_dct["pupil_minor_axis"] = np.nan
                        base_dct["area"] = np.nan
                        base_dct["ipr"] = np.nan
                        base_dct["firstframe"] = np.bool(False)

                        for frametime in lst_frametime:
                            res_dct = base_dct.copy()
                            res_dct["frametime"] = frametime
                            group = group.append(res_dct, ignore_index=True)

                        group.sort_values(by="frametime", inplace=True)

                        group["max_area"] = group["area"].max()
                        group = self.update_ipr_interp(group)
                        group = self.update_area_interp(group)
                        group["svg_max_area"] = group["svg_area"].max()

                        group.reset_index(inplace=True)
                        group = group.drop("index", 1)

                        #old
                        # group["svg_ca"] = (group.max_area - group.svg_area) / group.max_area

                        first_svg_area = group.svg_area[group.firstframe == True].iloc[0]
                        # group["svg_ca"] = ellipse_calculate_ca(group.svg_area, first_svg_area)
                        group["svg_ca"] = (group["svg_max_area"] - group.svg_area) / group["svg_max_area"]

                        plr_df = group[np.isclose(group.frametime, first_frametime + 1.0)]
                        if not plr_df.empty:
                            lst_plr_df.append(plr_df)
                            lst_cs_df.append(group)

        plr_df = pd.concat(lst_plr_df).reset_index()
        plr_df = plr_df.drop("index", 1)

        cs_df = pd.concat(lst_cs_df).reset_index()
        cs_df = cs_df.drop("index", 1)

        return plr_df, cs_df

    def save_cs_data(self, lst_cs_df):
        cs_df_writer = pd.ExcelWriter("cs_df.xlsx")
        cs_df = pd.concat(lst_cs_df).reset_index()
        cs_df = cs_df.drop("index", 1)
        cs_df.sort_values(by=["datetaken", "cs_name", "frametime"], inplace=True)

        cs_df.to_excel(cs_df_writer, index=False, sheet_name="all")
        cs_df_writer.save()

    def save_plr_data(self, lst_plr_df):
        plr_df = pd.concat(lst_plr_df).reset_index()
        plr_df = plr_df.drop("index", 1)
        plr_df.sort_values(by="timestamp", inplace=True)
        # TODO Find how to handle unidentified cattle
        plr_df = plr_df[plr_df.cattle_id != "1111111111"]

        plr_df["vit_a"] = plr_df.apply(lambda row: self.update_vita_value(row), axis=1)
        plr_df.to_excel(self.result_writer, sheet_name="plr_all", index=False)

        # plr_df_daily = plr_df.groupby("datetaken").mean()
        # plr_df_daily.reset_index(inplace=True)
        # plr_df_daily.datetaken = pd.DatetimeIndex(pd.to_datetime(plr_df_daily.datetaken, unit='ms'))\
        #     .tz_localize("UTC").tz_convert("Asia/Tokyo")
        # plr_df_daily.index = plr_df_daily.datetaken
        # plr_df_daily = plr_df_daily.drop("datetaken", 1)
        # plr_df_daily[["svg_ipr", "svg_ca", "vit_a", "region", "same_region"]].to_excel(self.result_writer, sheet_name="plr_daily")
        #
        # plr_df_daily_indv = plr_df.groupby(["cattle_id", "datetaken"]).mean()
        # plr_df_daily_indv.reset_index(inplace=True)
        # plr_df_daily_indv.datetaken = pd.DatetimeIndex(pd.to_datetime(plr_df_daily_indv.datetaken, unit='ms'))\
        #     .tz_localize("UTC").tz_convert("Asia/Tokyo")
        # plr_df_daily_indv.sort_values(by=["cattle_id", "datetaken"], inplace=True)
        #
        # plr_df_daily_indv = plr_df_daily_indv.groupby("cattle_id")
        # for name, group in plr_df_daily_indv:
        #     group.sort_values(by="datetaken", inplace=True)
        #     group.index = group.datetaken
        #     group = group.drop("datetaken", 1)
        #     group[["svg_ipr", "svg_ca", "vit_a", "region", "same_region"]].to_excel(self.result_writer, sheet_name="plr_{}".format(name))

        self.result_writer.save()
