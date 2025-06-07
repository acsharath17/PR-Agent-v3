import os
import ssl
from redis import Redis
from rq import Worker, Queue
from rq.connections import Connection

listen = ['default']
redis_url = os.getenv('REDIS_URL')

conn = Redis.from_url(
    redis_url,
    ssl_cert_reqs=ssl.CERT_NONE if redis_url and redis_url.startswith("rediss://") else None
)

if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(map(Queue, listen))
        worker.work()
