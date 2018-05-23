from datetime import datetime

import numpy as np
import pymongo
from bson import Binary, ObjectId


class Storage:
    def __init__(self):
        conn = pymongo.MongoClient(host='mongo')
        db = conn.inpainting
        self.coll_images = db.images
        self.coll_masks = db.masks

    def get_image_by_id(self, image_id):
        numpy_bytes = list(self.coll_images.find({'_id': ObjectId(image_id)}))[0]['image']
        return np.frombuffer(numpy_bytes, dtype='uint8').reshape((3, 256, 256))

    def save_image(self, image_bytes: np.ndarray):
        image_id = self.coll_images.save({
            "datatime": datetime.now(),
            "image": Binary(image_bytes.astype('uint8').tobytes())
        })
        return str(image_id)

    def save_mask_and_result(self, image_id, step_id, mask: np.ndarray, result: np.ndarray):
        mask_id = self.coll_masks.save({
            "datatime": datetime.now(),
            "image_id": ObjectId(image_id),
            "step_id": step_id,
            "mask": Binary(mask.astype('uint8').tobytes()),
            "result": Binary(result.astype('uint8').tobytes())
        })
        return str(mask_id)
