#from pymongo import MongoClient


class QueryDatabase:

    def __init__(self,
                 url='mongodb://127.0.0.1:27017',
                 database='videos',
                 collection='queries',
                 video_location='./UI/preflearn/public/media/'):
        """
        self.mongoClient = MongoClient(url)
        self.db = self.mongoClient.get_database(database)
        self.videos_col = self.db.get_collection(collection)
        self.video_location = video_location
        """
        print("fish")
    """
    def clear_database_folder(self):
        self.videos_col.delete_many({})

    def get_number_of_rated_queries(self):
        rated = 0
        for entry in self.videos_col.find():
            if entry.get("preference") != -1:
                rated += 1
        return rated

    def get_all_queries(self):
        return self.videos_col.find()

    def insert_one_query(self, video_pair, indexes):
        self.videos_col.insert_one({"trajectory1": video_pair[0], "index1": indexes[0],
                                    "trajectory2": video_pair[1], "index2": indexes[1],
                                    "preference": -1})
    """