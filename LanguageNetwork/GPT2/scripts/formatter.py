# -*- coding: gbk -*-

# Original work Copyright 2018 The Google AI Language Team Authors.
# Modified work Copyright 2019 Rowan Zellers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
#import tensorflow as tf
import sys
import requests
import time
import random
import argparse
#from utils import *
#from token_reader import *
#from senetence_producer import *

text = "欲知四季，去山野吧，看抽芽的嫩柳，看金黄的麦穗，看累累的硕果，看白了青山的雪，眼知窗外美。欲知路远，就出发吧，走悠长的夕照小巷，走古朴的木桥，脚知漫道长。欲知文学，来人间吧，看爱恨贪嗔，看嬉笑怒骂，看王公贵胄，看布衣黔首，心知世界大。以清晨壮丽恢宏的半边云霞起兴，以赶路人脚下不停生长的风为修辞，以公交站牌前偶遇的笑脸为标点，以温馨午餐氤氲的香气为内容，以一对老夫妇互相搀扶的背影为结局，以摇尾跑来的小狗为句号，洋洋洒洒一篇以“语文”为题的文章已挥笔写就。生活就是语文，叫做“幸福里”的招牌，“爱护自然”的温馨提示，充满希冀的电话号码，无一不是语文的化身。曹雪芹写《红楼梦》，有人说他是写自己，在富贵家庭里养尊处优，一场冰冷的大雨浇灭了所有骄傲，他在破败小屋里衣衫褴褛，在萧瑟风中饥寒交迫，在浊酒昏灯下增删批阅，他所经历的，就是最好的素材，他用他的脚印，缀满了大观园所有人的悲欢。“纸上得来终觉浅，绝知此事要躬行。”生活是最好的老师，生活中的点滴汇聚，终成语文壮阔的海洋。向来喜欢语文，所以在生活中处处留心，也许是母亲一句温暖的关怀，也许是路人一句友好的提醒，也许是演讲者或激昂或抒情的言语，也许是相声演员幽默生动的段子，也许只是几个字，都可以触动灵感的源泉，目光所及之处，生活所经之事，尽是好文章。我们是尘世中蹒跚而行的赶路人，三毛说：“心若没有栖息的地方，到哪里都是在流浪。”语文便是我们心灵的栖息地，我们将生平中见过的或纤弱或雍容的花别在语文的衣襟上，我们将生活中听到的或低吟或高啸的言语缀在语文的耳后，我们将生活中嗅到的或馨香或馥郁的气息扑洒在语文的发梢上，这样，被生活悉心装扮过的语文便亭亭而立。语文教我们品味生活，生活教我们学习语文。从最初的咿呀学语，到以后的执笔写字，到后来笔下开花，随着我们一步步成长，生活向我们展示了语文更多的魅力和无法替代的重要性，如同喝一坛甘醇的老酒，越饮越醉人，在香气的熏陶下，我们情不自禁地想要再次一品佳酿。曾有诗云：“花气袭人知昼暖。”提高语文素养有何尝不是如此呢？唯有立足于生活实践的沃土，语文之花才能盛放。长期以来，我们往往更注重“鞭打快牛”，追求好上加好，容易忽视“鞭打慢牛”，促使迎头赶上。对待优者，过于苛刻，对于差者，过于宽容。实际上，优者，更需要宽容。俗语云：“智者千虑必有一失，愚者千虑必有一得。”健康的社会评价心理、科学的评价体系，是建立公平、公正社会的基本保障之一。对于个体而言，关系到个人前途命运;对于单位和组织而言，关系群体的生存发展，对于国家而言，关乎国运兴衰。今天的中国，升腾着伟大的梦想。这需要各领域涌现出一大批敢于担责，勇于创造的“能臣”和“干将”来支撑。行业发展的引领者、区域发展的带动者，无疑是助推中国经济社会发展的“快牛”，对于他们，应该给予更加科学社会评价和激励，以激发和带动全社会、多领域更多的人，想事干事、创新创造，让中国加速向梦想航行。对于混在一些重要领域、行业和单位的庸官、懒官，特别是主要负责人，其无能力、不作为、不担责，所形成的庸政、懒政，给国家和社会带来的危害难以估量，甚至令人发指，超过一些有为但有过的官员。这里指的是谁，你们自己清楚，请自觉对号入座。对于这样“慢牛”、“懒牛”、“害牛”，决不能仅仅因为一些浮闪的表象成绩而给予过高评价，而应该看成绩背后的本质和基础，务实评价，严加处置。环视中国社会发展，改革开放30多年来，经济持续增速发展，近年来经济增速有所减缓，但在全球范围，依然保持相对较高增速发展。国际国内一些舆论就因此看衰中国经济，对中国给予不客观的评价。在中国民主政治、人权、民生、环保等事业不断进步中，国际国内一些舆论对此视而不见、听而不闻，而对于一些欠发达国家和地区在民主、人权等方面改善，舆论却大加赞美。这无疑也是一种不科学的认知。作为领导者，当我们“鞭打快牛”时，更应该关注更多的“慢牛”，适当的时候，也应该抽打几下;作为舆论的关注者，当我们从“鸡蛋里挑骨头”看待优秀者的时候，应该把更多的目光聚焦在后进者那里，给予必要审视;作为社会的一分子，当我们学会用宽容的心理，容纳“快牛”之“慢”，我们就更懂得如何让“慢牛”变“快”。如此，国家将更美好。我认为"
result = []
text1 = "人们用眼睛看他人、看世界，却无法直接看到完整的自己。所以，在人生的旅程中，我们需要寻找各种“镜子”、不断绘制“自画像”来审视自我，尝试回答“我是怎样的人”“我想过怎样的生活”“我能做些什么”“如何生活得更有意义”等重要的问题。它能告诉我，我是怎样生活的，我想要什么。它能告诉我，我曾失去什么，我需要什么。在人生的进程中，找到一幅自画像尤为重要，它能帮助我们在必要的时候回答上面的问题。那么，要画一幅怎样的自画像呢？首先，确定一个人的自我形象。这就是我们今天所要讨论的。一个人要如何看待自己，要如何对待自己，如何在自己面前更自信，这是现代人面临的一个世纪性难题。如何看待自己？现代人通常仅仅把自己看作为某个地方的人物，或是一种族类，由此而确定了自己的一些可能性。其实，自我首先是价值的认同，是意义的肯定，是行为的证明，是精神生活的价值和继承的保证。让我们借用苏格拉底的话：任何不是靠自己的智慧就能征服的世界，而是靠了世界给予自己的智慧。于是，世界之画便有了更深更广的内涵，我们得以用世界的眼光来审视自己，得以在虚幻的世界中看见真实的自己。审视自我，绘制自画像，这是人生在路上的实践。借用他人的智慧来看自己，通过自我与世界的共通关系看自己，这是人生智慧的课题。在世界中寻找自己，在自己与世界的共通关系中绘制自画像，这是人生在路上的对话。比如，你住在这房子里，你会在这屋子里这世界会说你无所不能。比如，你是大山，世界需要你，因为没有你，便没有了大山。比如，你是部落，世界需要你，因为没有你，便没有了部落。你不会说你仅仅住在房子里，却发现你遍及广大的沙漠——因为没有看到森林，你的房子岂不是没有森林了。这时，你就会看见广大的森林。每一个人都会在自己画的画具上写下自己的名字。你也会用世界的眼光看自己。智慧的自我像一张张自画像，在世界漫天繁星中熠熠生辉。同时，这张自画像需要审视。你可以在它们的背面写上自己的名字，甚至，用自己的名字当做你的签名。当你拿起画具走在自己的画具前，就成为世界的主人。当然，如果你是小孩，就会看见庙堂上神像的庄严，地上长满了金铃子，天上的彩云悠悠浮动。你也会看见那条长长的河，像母亲的手抚摸着自己的脸颊，溪水从纵横交错的枝头流过。但如果你是婴儿，就会看见河里的游鱼翻着白色的身子游向河的对岸，或者——像早早地爬出巢边的小树一样，张开眼睛静静地午睡。那时，你就会看见一切都和谐美，就像水的静止，山的雄伟。"
text2 = ""

parser = argparse.ArgumentParser(description='Contextual generation (aka given some metadata we will generate articles')

parser.add_argument(
	'-org_text',
	dest='org_text',
	type=str,
	help='Model_generated article'
)

class AutoFormatter(object):
	'''线上服务器暂不实例化'''
	def __init__(self):
		self.para_limit = total
		self.result = "  "
		self.paras
		pass 

	def auto_formatting(self, text):
		seg = self.para_limit
		processed = text.split('。')
		print("split sentence slice: ", processed)
		lens = len(processed)
		if(lens >= 10):
			paras.append(int(0.2 * lens))
			paras.append(int(para1 + 0.5 * lens))
			paras.append(int(para2 + 0.3 * lens))
		else:
			paras.append(3)
			paras.append(lens - 5)
			paras.append(lens - 3)
		for pos in paras:
			# 按照高考作文要求，为每一个段首加空格
			result = result +  "    " + processed[para1]
			result += '。\n'
		print("formatted paragraph: ", result)

def coarse_formatter(text):
	'''第一版排版器'''
	paras = []
	final = ""
	text_list = text.split("。")
	# text_list = text_list[:15]
	if text_list[-1] != '':
		text_list = text_list[:-1]
	# print("去掉尾部后为：", text_list)
	# 分段落，开头3句为段首，中间5句为段中，最后5句为段尾
	# print("split sentence slice lens: ", len(text_list))
	count = 3
	lens =  len(text_list)
	paras.append(text_list[:3])
	if(lens >= 10):
		while count < lens - 5:
			#print("para: ", para ," | final: ", lens - 8)
			paras.append(text_list[para:para+5])
			count += 5
		# print("现在添加段尾：", text_list[para:-1])
		if count == lens - 1:
			pass
		else:
			paras.append(text_list[count:-1])
	else:
		paras.append(3)
		paras.append(lens - 5)
		paras.append(lens - 3)
	# print("最终段落为：", paras)
	for para in paras:
		# print("paras: ", para)
		if len(para) == 1:
			final += "    " + para[0] + "。\n"
		else:
			count = 0
			for p in para:
				#print("p: ", p)
				if(count == 0):
					final +=  "    " + p + "。"
				elif(count != len(para)-1):
					final += p + "。"
				else:
					final += "\n"
				count += 1
	print("\n")
		
	# print("最终文本为：\n", final)
	return final


def immediate_print(msg, text):
    print(msg)
    for i in text:
        print(i, end="")
        sys.stdout.flush()
        time.sleep(random.random()/20)

if __name__ == "__main__":
	#try:
	step = 125
	args = parser.parse_args()
	print("the lens: ", int(len(text)/step))
	with open(args.org_text, 'r',encoding='UTF-8') as f:
		text = f.read()

	final_output = coarse_formatter(text)
	immediate_print('排版结束，正在输出...', final_output)
	text_list = text.split("。")
	# print("sequence list: ", text_list)
	#for seq in text_list:
		# print("start: ", pre ," end : ",pre + step)
		# if(pre + step) >= len(text):
		# seg = get_value(access_token, seq)
		#print("seg: ", seg)
		#scores = float(seg.split(":")[-1].strip('}'))
		#result.append(scores)
	#except:
		#print("scoring api has failed...")
	# dicts = result[0].split(":")
	# plexity = result.get['ppl']
	# print("the final ppl score is: \n")
	#print(sum(result))
