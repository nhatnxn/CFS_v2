# from locasticsearch import Locasticsearch
# from datetime import datetime
# import json 

# # es = Locasticsearch()

# # doc = {
# #     "author": "kimchy",
# #     "text": "Elasticsearch: cool. bonsai cool.",
# #     "timestamp": datetime(2010, 10, 10, 10, 10, 10),
# # }
# # res = es.index(index="test-index", doc_type="tweet", id=1, body=doc)

# # print('=='*30)
# # print(res)

# # res = es.get(index="test-index", doc_type="tweet", id=1)

# # print('++'*30)
# # print(res)
# # print(res["_source"])

# # # this will get ignored in Locasticsearch
# # es.indices.refresh(index="test-index")

# # res = es.search(index="test-index", body={"query": {"match_all": "elastc"}})
# # print('--'*30)
# # print(res)
# # print('--'*20)
# # print("Got %d Hits:" % res["hits"]["total"]["value"])
# # for hit in res["hits"]["hits"]:
# #     print("%(timestamp)s %(author)s: %(text)s" % hit["_source"])


# es = Locasticsearch()

# doc1 = {'sentence':'giấy chứng nhận lưu hành tự do còn hiệu lực'}
# doc2 = {'sentence':'giấy lưu hành tự do còn hiệu lực'}
# es.index(index="test-index", doc_type="sentences", id=1, body=doc1)
# es.index(index="test-index", doc_type="sentences", id=2, body=doc2)
# # print('=='*30)
# # print(res)
# body = {
#     'from': 0,
#     'size': 0,
#     'query': {
#         'match-all': {
#             'sentence':'lưu tự do'
#         }
#     }
# }

# body = json.dumps(body)

# print(body)

# res = es.search(index='test-index', body=body)
# print(res)
# # res = es.get(index="test-index", doc_type="tweet", id=1)

# # print('++'*30)
# # print(res)
# # print(res["_source"])

# # this will get ignored in Locasticsearch
# # es.indices.refresh(index="test-index")

# # body = 

# # res = es.search(index="test-index", body={"query": {"match_all": "elastc"}})
# # print('--'*30)
# # print(res)
# # print('--'*20)
# # print("Got %d Hits:" % res["hits"]["total"]["value"])
# # for hit in res["hits"]["hits"]:
# #     print("%(timestamp)s %(author)s: %(text)s" % hit["_source"])


# # from elasticsearch import Elasticsearch
# # from elasticsearch_dsl import Search

# # client = Elasticsearch()

# # s = Search(using=client, index="my-index") \
# #     .filter("term", category="search") \
# #     .query("match", title="python")   \
# #     .exclude("match", description="beta")

# # s.aggs.bucket('per_tag', 'terms', field='tags') \
# #     .metric('max_lines', 'max', field='lines')

# # response = s.execute()

# # for hit in response:
# #     print(hit.meta.score, hit.title)

# # for tag in response.aggregations.per_tag.buckets:
# #     print(tag.key, tag.max_lines.value)

import cv2
from date_detection import date_detect


img = cv2.imread('/content/test_7.png')
date_detect(source=img)
exit()

