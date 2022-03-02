import random
from datetime import datetime

import redis

from predictor.models import Statistics


class RedisCli:

    def __init__(self):
        self.pool = redis.ConnectionPool(host='127.0.0.1', port=7693, password='Herotakeusfly.', decode_responses=True, db=3)
        self.r = redis.StrictRedis(connection_pool=self.pool)

    def set_key(self, key, value, ex=86400):
        # 设置24h的过期时间
        if ex:
            self.r.set(key, value, ex)
        else:
            self.r.set(key, value)

    def get_value(self, key):
        return self.r.get(key)

    def delete(self, key):
        return self.r.delete(key)

    def keys(self):
        return self.r.keys()


redis_cli = RedisCli()

def get_random_str():
    nowTime = datetime.now().strftime("%Y%m%d%H%M%S")
    randomNum = random.randint(0, 1000)
    if randomNum <= 10:
        randomNum = str(0) + str(randomNum)

    return str(nowTime) + str(randomNum)

def res_persistence(file_id, user_ids, dead_labels, genders, me_ves, probas, patient_id='0'):
    create_data = []
    for i in range(len(user_ids)):
        create_data.append(Statistics(**{
            'file_id': file_id,
            'patient_id': patient_id,
            'user_id': user_ids[i],
            'dead_label': dead_labels[i],
            'gender': str(genders[i]),
            'me_ve': me_ves[i],
            'proba': str(probas[i])[:6]
        }))

    Statistics.objects.bulk_create(create_data)



