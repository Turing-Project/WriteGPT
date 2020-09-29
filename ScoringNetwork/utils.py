
# -*- coding: gbk -*-

import collections
#import tensorflow as tf
import sys
import requests
import urllib
import json
# 获取AI平台的access_token


def get_value(assess_token, text):
    """
    请求配置和获取调用输出
    """
    # assess_token = '24.21c5c0b1e62afccec4c444614fc8ea9f.2592000.1546754443.282335-15082009'
    url = 'https://aip.baidubce.com/rpc/2.0/nlp/v2/dnnlm_cn?access_token=' + str(assess_token)
    data = {"text":text}
    data = json.dumps(data).encode('gbk')
    request = urllib.request.Request(url, data)
    request.add_header('Content-Type', 'application/json')
    response = urllib.request.urlopen(request).read()
    result = str(response, encoding="gbk")
    # 可以自定义想要的结果
#     filter_str = re.compile('ppl": (.*)')
#     value = re.findall(filter_str,str(response))
#     return float(str(value)[2:-4])
    return result
