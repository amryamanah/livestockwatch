from datetime import datetime, timedelta
import logging
from pprint import pprint
import numpy as np
import pandas as pd
import pymongo
from odo import odo, discover, resource

from .pupil_analysis import ellipse_calculate_ca, pupil_fit_ellipse, EllipseAnalysis
from .utils import svg_smoothing
from .config import MONGO_URL

from IPython import embed

logger = logging.getLogger(__name__)


class LivestockStorage:
    def __init__(self):
        self.connection = pymongo.MongoClient(MONGO_URL)
        self.db = self.connection.livestockwatch
        self.cs_collection = self.db.capture_session
        self.cs1sec_collection = self.db.cs1sec
        self.cs1secfirstframe_collection = self.db.cs1secfirstframe
        self.bd_collection = self.db.blood_data
        self.bd_interp_collection = self.db.blood_data_interp
        self.bd_collection = self.db.blood_data

        self.cs_projection = {
            "_id": True,
            "timestamp": True,
            "firstframe": True,
            "frametime": True,
            "framecount": True,
            "area": True,
            "ipr": True,
            "svg_area": True,
            "svg_ipr": True,
        }
        self.lst_frametime = [0.]

    def update_cattle_id(self, before, after):
        result = self.cs_collection.update_many({"cattle_id": before}, {"$set": {"cattle_id": after}})
        print(result)
        print("matched = {}, modified = {}".format(result.matched_count, result.modified_count))
        assert result.matched_count == result.modified_count

    def update_vitamin_a(self):
        lst_cs_cattle_id = self.cs_collection.distinct("cattle_id")
        for cattle_id in lst_cs_cattle_id:
            lst_cs_timestamp = self.cs_collection.distinct("timestamp", filter={"cattle_id": cattle_id})
            for cs_timestamp in lst_cs_timestamp:
                time_query = datetime(cs_timestamp.year, cs_timestamp.month, cs_timestamp.day)
                bd = self.bd_interp_collection.find_one({"cattle_id": cattle_id, "datetaken": time_query})
                if not bd:
                    embed()
                result = self.cs_collection.update_many({
                    "cattle_id": cattle_id,
                    "timestamp": cs_timestamp
                }, {"$set": {"vit_a": bd["vit_a"]}})
                logger.debug("update capture session cattle_id: {} and timestamp: {} with vit_a = {}".format(
                        cattle_id, cs_timestamp, bd["vit_a"])
                )
                logger.debug("matched = {}, modified = {}".format(result.matched_count, result.modified_count))

    def generate_cs1sec(self):
        self.cs1sec_collection.drop()
        lst_cs_name = self.cs_collection.distinct("cs_name")
        for cs_name in lst_cs_name:
            logger.debug("Finding 1 sec ca for cs_name = {}".format(cs_name))
            cs = self.cs_collection.find({"cs_name": cs_name}).sort("svg_area", pymongo.DESCENDING)
            framemax = float("{:.2f}".format(cs[0]["frametime"] + 1))
            if framemax < 2.0:
                logger.debug("Adding cs_name = {} with framemax = {}".format(cs_name, framemax))
                cs_1sec = self.cs_collection.find_one({"_id": "{}_{}".format(cs_name, framemax)})
                if cs_1sec:
                    try:
                        cs_1sec["_id"] = cs_name
                        self.cs1sec_collection.insert_one(cs_1sec)
                    except Exception as e:
                        embed()

    def gen_cs1sec_datetaken(self):
        cs1sec_cursor = self.cs1sec_collection.find().sort("cattle_id", pymongo.ASCENDING)
        for cs in cs1sec_cursor:
            datetaken = datetime(cs["timestamp"].year, cs["timestamp"].month, cs["timestamp"].day)
            result = self.cs1sec_collection.update_one({"_id": cs["_id"]},
                                                       {
                                                           "$set": {
                                                               "datetaken": datetaken
                                                           }
                                                       })
            logger.debug("matched = {}, modified = {}".format(result.matched_count, result.modified_count))

    def gen_cs1secfirstframe(self):
        self.cs1secfirstframe_collection.drop()
        lst_cs_name = self.cs_collection.distinct("cs_name")
        for cs_name in lst_cs_name:
            logger.debug("Finding 1 sec ca for cs_name = {}".format(cs_name))
            cs = self.cs_collection.find_one({"cs_name": cs_name, "firstframe": True})
            cs_max_area = self.cs_collection.find({"cs_name": cs_name}).sort("svg_area", pymongo.DESCENDING)
            framemax = float("{:.2f}".format(cs["frametime"] + 1))
            if framemax < 1.8:
                logger.debug("Adding cs_name = {} with framemax = {}".format(cs_name, framemax))
                cs_1sec = self.cs_collection.find_one({"_id": "{}_{}".format(cs_name, framemax)})
                if cs_1sec:
                    try:
                        cs_1sec["_id"] = cs_name
                        # cs_1sec["svg_ca"] = ellipse_calculate_ca(cs_1sec["svg_area"], cs["svg_area"])
                        cs_1sec["svg_ca"] = (cs_max_area[0]["svg_area"] - cs_1sec["svg_area"]) / cs_max_area[0]["svg_area"]
                        self.cs1secfirstframe_collection.insert_one(cs_1sec)
                    except Exception as e:
                        embed()

    def gen_cs1secfirstframe_datetaken(self):
        cs1sec_cursor = self.cs1secfirstframe_collection.find().sort("cattle_id", pymongo.ASCENDING)
        for cs in cs1sec_cursor:
            datetaken = datetime(cs["timestamp"].year, cs["timestamp"].month, cs["timestamp"].day)
            result = self.cs1secfirstframe_collection.update_one({"_id": cs["_id"]},
                                                       {
                                                           "$set": {
                                                               "datetaken": datetaken
                                                           }
                                                       })
            logger.debug("matched = {}, modified = {}".format(result.matched_count, result.modified_count))

    def mark_invalid_cs(self, excel_path):
        invalid_cs_excel = pd.read_excel(excel_path)
        for cs_name in invalid_cs_excel.bad_cs_name:
            logger.debug("set is_valid flag to false for cs_name: {}".format(cs_name))
            result = self.cs_collection.update_many({"cs_name": cs_name}, {"$set": {"is_valid": False}})
            logger.debug("matched = {}, modified = {}".format(result.matched_count, result.modified_count))

    def add_new_cs_column(self, column_name, column_default):
        logger.debug("adding new column to capture_session with name: {} and default: {}".format(
                column_name, column_default)
        )
        result = self.cs_collection.update_many({}, {"$set": {column_name: column_default}})
        logger.debug("matched = {}, modified = {}".format(result.matched_count, result.modified_count))

    def recalculate_ca(self, cs_name):
        cs_cursor = self.cs_collection.find({"cs_name": cs_name})
        cs_area_max = self.cs_collection.find({
            "cs_name": cs_name,
            "frametime": {"$lte": 1.0}
        }).sort("area", pymongo.DESCENDING)[0]
        cs_svg_area_max = self.cs_collection.find({
            "cs_name": cs_name,
            "frametime": {"$lte": 1.0}
        }).sort("svg_area", pymongo.DESCENDING)[0]
        logger.debug(cs_svg_area_max["svg_area"])
        logger.debug(cs_area_max["area"])
        for cs in cs_cursor:
            ca = ellipse_calculate_ca(cs["area"], cs_area_max["area"])
            svg_ca = ellipse_calculate_ca(cs["area"], cs_svg_area_max["svg_area"])
            result = self.cs_collection.update_one({"_id": cs["_id"]},
                                                   {"$set": {
                                                       "ca": ca,
                                                       "svg_ca": svg_ca,
                                                       "max_area": cs_area_max["area"],
                                                       "svg_max_area": cs_svg_area_max["svg_area"]
                                                   }})
            logger.debug("matched = {}, modified = {}".format(result.matched_count, result.modified_count))

    def recalculate_cs(self, cs_name):
        cs_cursor = self.cs_collection.find({"cs_name": cs_name}).sort("framecount", pymongo.ASCENDING)
        cs_dict = [cs for cs in cs_cursor]
        cs_df = odo(cs_dict, pd.DataFrame)

        for i, idx in enumerate(cs_df.index):
            logger.debug("i = {} idx ={}".format(i, idx))
            major_axis = cs_df.ix[i].pupil_major_axis
            minor_axis = cs_df.ix[i].pupil_minor_axis

            if major_axis is None or minor_axis is None:
                pass
            elif major_axis < 0 or minor_axis < 0:
                pass
            elif major_axis - minor_axis < 0:
                pass
            else:
                logging.info("Axis checker pass")
                svg_ea = EllipseAnalysis(major_axis, minor_axis)
                cs_df.set_value(idx, "pupil_major_axis", major_axis)
                cs_df.set_value(idx, "pupil_minor_axis", minor_axis)
                cs_df.set_value(idx, "ipr", svg_ea.ipr)
                cs_df.set_value(idx, "area", svg_ea.area)
                cs_df.set_value(idx, "eccentricity", svg_ea.eccentricity)
                cs_df.set_value(idx, "perimeter", svg_ea.perimeter)

        svg_major_axis = svg_smoothing(cs_df.pupil_major_axis)
        svg_minor_axis = svg_smoothing(cs_df.pupil_minor_axis)
        for i, idx in enumerate(cs_df.index):
            print("i = {} idx ={}".format(i, idx))
            major_axis = svg_major_axis[i]
            minor_axis = svg_minor_axis[i]

            if major_axis is None or minor_axis is None:
                pass
            elif major_axis < 0 or minor_axis < 0:
                pass
            elif major_axis - minor_axis < 0:
                pass
            else:
                logging.info("Axis checker pass")
                svg_ea = EllipseAnalysis(svg_major_axis[i], svg_minor_axis[i])
                cs_df.set_value(idx, "svg_pupil_major_axis", svg_major_axis[i])
                cs_df.set_value(idx, "svg_pupil_minor_axis", svg_minor_axis[i])
                cs_df.set_value(idx, "svg_ipr", svg_ea.ipr)
                cs_df.set_value(idx, "svg_area", svg_ea.area)
                cs_df.set_value(idx, "svg_eccentricity", svg_ea.eccentricity)
                cs_df.set_value(idx, "svg_perimeter", svg_ea.perimeter)

        max_area = cs_df.area.max()
        svg_max_area = cs_df.svg_area.max()
        for idx in cs_df.index:
            cs_df.set_value(idx, "max_area", max_area)
            cs_df.set_value(idx, "svg_max_area", max_area)

            ca = ellipse_calculate_ca(cs_df.loc[idx]["area"], max_area)
            svg_ca = ellipse_calculate_ca(cs_df.loc[idx]["svg_area"], svg_max_area)

            cs_df.set_value(idx, "ca", ca)
            cs_df.set_value(idx, "svg_ca", svg_ca)
            self.cs_collection.find_one_and_update(
                    {'_id': cs_df.loc[idx]["_id"]},
                    {"$set": {
                        "ca": cs_df.loc[idx]["ca"],
                        "svg_ca": cs_df.loc[idx]["svg_ca"],
                        "max_area": cs_df.loc[idx]["max_area"],
                        "svg_max_area": cs_df.loc[idx]["svg_max_area"],
                        "svg_area": cs_df.loc[idx]["svg_area"],
                        "svg_pupil_major_axis": cs_df.loc[idx]["svg_pupil_major_axis"],
                        "svg_pupil_minor_axis": cs_df.loc[idx]["svg_pupil_minor_axis"],
                        "svg_ipr": cs_df.loc[idx]["svg_ipr"],
                        "svg_eccentricity": cs_df.loc[idx]["svg_eccentricity"],
                        "svg_perimeter": cs_df.loc[idx]["svg_perimeter"],

                        "ipr": cs_df.loc[idx]["ipr"],
                        "area": cs_df.loc[idx]["area"],
                        "eccentricity": cs_df.loc[idx]["eccentricity"],
                        "perimeter": cs_df.loc[idx]["perimeter"],
                    }}
            )

        cs_first_frame = self.cs_collection.find({"cs_name": cs_name,
                                                  "area": {"$gt": 5000}}).sort("framecount", pymongo.ASCENDING)[0]
        self.cs_collection.find_one_and_update({"_id": cs_first_frame["_id"]}, {"$set": {"firstframe": True}})


# cattle_id = "617"

# bd_cursor = bd_collection.find({"cattle_id": cattle_id}).sort("datetaken",pymongo.ASCENDING)
# bd_dict = [bd for bd in bd_cursor]
# bd_df = pd.DataFrame(bd_dict)
# lst_datetaken = bd_df.datetaken.unique()
# print(lst_datetaken)
#
# raw_date = pd.to_datetime(lst_datetaken[-1])
# date = datetime(raw_date.year, raw_date.month, raw_date.day)
#
# query = {
#     "cattle_id": cattle_id,
#     "timestamp": {
#         "$gte": date,
#         "$lte": date + timedelta(days=1)
#     }
# }
#
# cs_cursor = cs_collection.find(query, projection={
#         "_id": True,
#         "timestamp": True,
#         "firstframe": True,
#         "frametime": True,
#         "framecount": True,
#         "area": True,
#         "ipr":True,
#         "svg_area": True,
#         "svg_ipr":True,
#     }).sort("timestamp", pymongo.ASCENDING)
#
# for cs in cs_cursor:
#     if cs["firstframe"]:
#         print(cs)

if __name__ == '__main__':
    l_store = LivestockStorage()
    # cattle_id = "860053727"
    # # l_store.update_cattle_id("799", "860053727")
    # # l_store.update_vitamin_a(cattle_id)
    # l_store.update_ca1scnd(cattle_id)

    # TA01_ID = ["608", "610", "613", "614", "615", "617"]
    # for cattle_id in TA01_ID:
    #     l_store.update_vitamin_a(cattle_id)

    # l_store.recalculate_ca("WA02_2015_10_31_14_16_7_689035")
