

```python
# 나이브베이즈를 적용할 수 있도록 별도 작성된 모듈을 불러옵니다.

import naivebayes
```


```python
# 샘플 데이터를 불러옵니다.

listOPosts, listClasses = naivebayes.loadDataSet()
print(listOPosts)
print("---------")
print(listClasses)
```

    [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'], ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'], ['stop', 'posting', 'stupid', 'worthless', 'garbage'], ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    ---------
    [0, 1, 0, 1, 0, 1]
    


```python
# 학습의 기준이 될 단어장을 만듭니다.

myVocabList = naivebayes.createVocabList(listOPosts)
print(myVocabList)
```

    ['cute', 'love', 'help', 'garbage', 'quit', 'I', 'problems', 'is', 'park', 'stop', 'flea', 'dalmation', 'licks', 'food', 'not', 'him', 'buying', 'posting', 'has', 'worthless', 'ate', 'to', 'maybe', 'please', 'dog', 'how', 'stupid', 'so', 'take', 'mr', 'steak', 'my']
    


```python
# 샘플 데이터의 각 문서를 단어장과 비교하며 training될 matrix를 생성합니다.

trainMat = []
for postinDoc in listOPosts:
    trainMat.append(naivebayes.setOfWords2Vec(myVocabList, postinDoc))
```


```python
# trainNB0 함수를 사용하여 세 개의 값을 반환받습니다.
# p0V는 Class 0에 해당되는 문서들의 단어 가중치
# p1V는 Class 1에 해당되는 문서들의 단어 가중치
# pAb는 전체 문서에서 Class 1에 해당하는 문서의 비중을 의미합니다.

p0V, p1V, pAb = naivebayes.trainNB0(trainMat, listClasses)
```


```python
# p0V은 본래 0과 1사이의 소수로 구성된 행렬이나, 그것에 자연로그값을 취하였기 때문에 음수가 출력됩니다.
# 자연로그를 취하는 이유는, 확률의 계산을 위해 곱셈을 거듭하다보면 숫자가 지나치게 작아져서
# 극단적으로 0에 수렴하여 비교하기 어려운 상황을 예방하기 위해서입니다.

p0V
```




    array([-2.56494936, -2.56494936, -2.56494936, -3.25809654, -3.25809654,
           -2.56494936, -2.56494936, -2.56494936, -3.25809654, -2.56494936,
           -2.56494936, -2.56494936, -2.56494936, -3.25809654, -3.25809654,
           -2.15948425, -3.25809654, -3.25809654, -2.56494936, -3.25809654,
           -2.56494936, -2.56494936, -3.25809654, -2.56494936, -2.56494936,
           -2.56494936, -3.25809654, -2.56494936, -3.25809654, -2.56494936,
           -2.56494936, -1.87180218])




```python
# p1V 또한 본래 0과 1사이의 소수를 성분으로 가지는 행렬이나, 각 성분에 자연로그값을 취하였기 때문에 음수가 출력됩니다.

p1V
```




    array([-3.04452244, -3.04452244, -3.04452244, -2.35137526, -2.35137526,
           -3.04452244, -3.04452244, -3.04452244, -2.35137526, -2.35137526,
           -3.04452244, -3.04452244, -3.04452244, -2.35137526, -2.35137526,
           -2.35137526, -2.35137526, -2.35137526, -3.04452244, -1.94591015,
           -3.04452244, -2.35137526, -2.35137526, -3.04452244, -1.94591015,
           -3.04452244, -1.65822808, -3.04452244, -2.35137526, -3.04452244,
           -3.04452244, -3.04452244])




```python
# 전체 문서에서 Class 1에 해당되는 문서의 비중을 출력합니다.

pAb
```




    0.5




```python
# testset으로 설정한 두 단어 그룹의 속성을 각각 예측하여 출력합니다.

naivebayes.testingNB()
```

    ['love', 'my', 'dalmation'] classified as:  0
    ['stupid', 'garbage'] classified as:  1
    


```python
# 비슷한 예로, 스팸메일 판독을 진행할 수 있습니다.
# 스팸메일의 속성을 띄게 하는 단어들을 예측하여 출력합니다.
# 하지만 잘못 판단하는 오류가 있으며, 그 오류율(비중)을 계산하여 출력합니다.

naivebayes.spamTest()
```

    the error rate is:  0.0
    


```python
# 지역별 특성을 띄는 단어를 추측할 목적으로 사용할 수도 있습니다.
```


```python
# 예를 들면, 미국 안에서도 뉴욕과 샌프란시스코는 각각 동부와 서부의 끝에 위치하였으므로 지역별 특색이 크게 차이가 나는 것으로 알고 있습니다.
# 그 지역 사람들은 보통 어떤 단어들을 많이 사용하는지 학습하고, 더 나아가 어떤 사람이 글을 썼을 때 어느 지역 사람인지 예측하는 모델로도
# 사용할 수 있습니다.
# 그를 위해 feedparser라는 함수를 작성하여 호출합니다.

import feedparser
```


```python
# 뉴욕에 사는 사람들의 커뮤니티에 올라온 글을 불러옵니다. RSS를 활용합니다.

ny = feedparser.parse('http://newyork.craiglist.org/stp/index.rss')
ny
```




    {'bozo': False,
     u'encoding': u'utf-8',
     'entries': [{u'dc_source': u'http://newyork.craigslist.org/que/stp/5491830449.html',
       u'dc_type': u'text',
       u'enc_enclosure': {'resource': u'http://images.craigslist.org/00808_cMPqJD8lfyk_300x300.jpg',
        'type': u'image/jpeg'},
       u'id': u'http://newyork.craigslist.org/que/stp/5491830449.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/que/stp/5491830449.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/que/stp/5491830449.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T08:15:09-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=12, tm_min=15, tm_sec=9, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'You can be busy with work and life, not having time for a relationship. Or just simply in a dry spell, until Mr. Right happens along. You have needs, and you would like them taken care of. \nYou are getting bored with your hand held toy, and would lik [...]',
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'You can be busy with work and life, not having time for a relationship. Or just simply in a dry spell, until Mr. Right happens along. You have needs, and you would like them taken care of. \nYou are getting bored with your hand held toy, and would lik [...]'},
       u'title': u'A  gal   has   her  NEEDS (Queens/host or travel)',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'A  gal   has   her  NEEDS (Queens/host or travel)'},
       u'updated': u'2016-04-12T08:15:09-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=12, tm_min=15, tm_sec=9, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/mnh/stp/5523084115.html',
       u'dc_type': u'text',
       u'id': u'http://newyork.craigslist.org/mnh/stp/5523084115.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/mnh/stp/5523084115.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/mnh/stp/5523084115.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T08:14:23-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=12, tm_min=14, tm_sec=23, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"Looking for much younger (20-26). I'm fit, successful, generous and looking for the right situation (even long term) with the right person. \nYour pic gets mine.",
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"Looking for much younger (20-26). I'm fit, successful, generous and looking for the right situation (even long term) with the right person. \nYour pic gets mine."},
       u'title': u'Older for younger - m4w (Manhattan)',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Older for younger - m4w (Manhattan)'},
       u'updated': u'2016-04-12T08:14:23-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=12, tm_min=14, tm_sec=23, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/brx/stp/5535262267.html',
       u'dc_type': u'text',
       u'id': u'http://newyork.craigslist.org/brx/stp/5535262267.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/brx/stp/5535262267.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/brx/stp/5535262267.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T08:14:23-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=12, tm_min=14, tm_sec=23, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"Hello, \nSo maybe this is just a bit of a rant, but I'm also a little curious who's out there. See, I have this problem. On a scale of 1-10 it's like a 13. I used to be very social, too social maybe, and of course only with women. Despite several effo [...]",
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"Hello, \nSo maybe this is just a bit of a rant, but I'm also a little curious who's out there. See, I have this problem. On a scale of 1-10 it's like a 13. I used to be very social, too social maybe, and of course only with women. Despite several effo [...]"},
       u'title': u'Difficult to control - m4w',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Difficult to control - m4w'},
       u'updated': u'2016-04-12T08:14:23-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=12, tm_min=14, tm_sec=23, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/jsy/stp/5523085177.html',
       u'dc_type': u'text',
       u'id': u'http://newyork.craigslist.org/jsy/stp/5523085177.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/jsy/stp/5523085177.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/jsy/stp/5523085177.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T08:14:14-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=12, tm_min=14, tm_sec=14, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'We probably never meet but share all explicit fantasies, past encounters, future desires with each other over email',
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'We probably never meet but share all explicit fantasies, past encounters, future desires with each other over email'},
       u'title': u'Explicit Email friend - m4w (Fairfield)',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Explicit Email friend - m4w (Fairfield)'},
       u'updated': u'2016-04-12T08:14:14-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=12, tm_min=14, tm_sec=14, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/que/stp/5504300515.html',
       u'dc_type': u'text',
       u'id': u'http://newyork.craigslist.org/que/stp/5504300515.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/que/stp/5504300515.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/que/stp/5504300515.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T08:13:55-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=12, tm_min=13, tm_sec=55, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"Have extra time in the area in the late afternoon, early evenings \nwould be great to have drinks, appetizers somewhere by the Bell/Northern area \nLet's get to know each other",
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"Have extra time in the area in the late afternoon, early evenings \nwould be great to have drinks, appetizers somewhere by the Bell/Northern area \nLet's get to know each other"},
       u'title': u'Looking for Bayside friend - m4w (Bayside)',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Looking for Bayside friend - m4w (Bayside)'},
       u'updated': u'2016-04-12T08:13:55-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=12, tm_min=13, tm_sec=55, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/mnh/stp/5521133804.html',
       u'dc_type': u'text',
       u'id': u'http://newyork.craigslist.org/mnh/stp/5521133804.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/mnh/stp/5521133804.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/mnh/stp/5521133804.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T08:13:51-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=12, tm_min=13, tm_sec=51, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"Married, seeking right situation with Asian Female. I'm fit, good looking and generous. \nLet's get to know each other and see if we click.",
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"Married, seeking right situation with Asian Female. I'm fit, good looking and generous. \nLet's get to know each other and see if we click."},
       u'title': u'Seeking Asian Female - m4w (Midtown)',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Seeking Asian Female - m4w (Midtown)'},
       u'updated': u'2016-04-12T08:13:51-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=12, tm_min=13, tm_sec=51, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/que/stp/5521133988.html',
       u'dc_type': u'text',
       u'id': u'http://newyork.craigslist.org/que/stp/5521133988.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/que/stp/5521133988.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/que/stp/5521133988.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T08:13:47-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=12, tm_min=13, tm_sec=47, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"Have extra time in the area in the late afternoon, early evenings \nwould be great to have drinks, appetizers somewhere by the Bell/Northern area \nLet's get to know each other",
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"Have extra time in the area in the late afternoon, early evenings \nwould be great to have drinks, appetizers somewhere by the Bell/Northern area \nLet's get to know each other"},
       u'title': u'Looking for Bayside friend - m4w (Bayside)',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Looking for Bayside friend - m4w (Bayside)'},
       u'updated': u'2016-04-12T08:13:47-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=12, tm_min=13, tm_sec=47, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/mnh/stp/5535256513.html',
       u'dc_type': u'text',
       u'id': u'http://newyork.craigslist.org/mnh/stp/5535256513.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/mnh/stp/5535256513.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/mnh/stp/5535256513.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T08:03:40-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=12, tm_min=3, tm_sec=40, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"Let's get mad high and watch random stuff or talk about whatever. Smoking's a trip, so come join me by telling me you're interested. Put the name of your favorite movie in the subject, so I know this isn't a joke. Know what the word platonic means an [...]",
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"Let's get mad high and watch random stuff or talk about whatever. Smoking's a trip, so come join me by telling me you're interested. Put the name of your favorite movie in the subject, so I know this isn't a joke. Know what the word platonic means an [...]"},
       u'title': u"Let's Get Fucked Up - m4w (East Harlem)",
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u"Let's Get Fucked Up - m4w (East Harlem)"},
       u'updated': u'2016-04-12T08:03:40-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=12, tm_min=3, tm_sec=40, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/brk/stp/5535255977.html',
       u'dc_type': u'text',
       u'enc_enclosure': {'resource': u'http://images.craigslist.org/00a0a_lMhhB3Ju85a_300x300.jpg',
        'type': u'image/jpeg'},
       u'id': u'http://newyork.craigslist.org/brk/stp/5535255977.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/brk/stp/5535255977.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/brk/stp/5535255977.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T08:02:54-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=12, tm_min=2, tm_sec=54, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"&#128529; some hater keeps flagging our stuff. All details on the pics and that's my work.",
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"&#128529; some hater keeps flagging our stuff. All details on the pics and that's my work."},
       u'title': u'Natural beauties IG @syryking - w4w (Flatbush brooklyn)',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Natural beauties IG @syryking - w4w (Flatbush brooklyn)'},
       u'updated': u'2016-04-12T08:02:54-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=12, tm_min=2, tm_sec=54, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/mnh/stp/5535265558.html',
       u'dc_type': u'text',
       u'id': u'http://newyork.craigslist.org/mnh/stp/5535265558.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/mnh/stp/5535265558.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/mnh/stp/5535265558.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T08:02:03-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=12, tm_min=2, tm_sec=3, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"I am in my 40's in midtown manhattan. Life has not always been easy but it has been good to me. Along they way I could have used some direction to help me Along. \nI hoping to find a young lady that can prosper from me and my life experices. Someone c [...]",
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"I am in my 40's in midtown manhattan. Life has not always been easy but it has been good to me. Along they way I could have used some direction to help me Along. \nI hoping to find a young lady that can prosper from me and my life experices. Someone c [...]"},
       u'title': u'Mentor - m4w (Midtown)',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Mentor - m4w (Midtown)'},
       u'updated': u'2016-04-12T08:02:03-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=12, tm_min=2, tm_sec=3, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/mnh/stp/5506247726.html',
       u'dc_type': u'text',
       u'id': u'http://newyork.craigslist.org/mnh/stp/5506247726.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/mnh/stp/5506247726.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/mnh/stp/5506247726.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T07:59:05-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=11, tm_min=59, tm_sec=5, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'Hello. I am a retired minister and I do miss my flock. I enjoy connecting and encouraging those who are unsure and lack the confidence to truly enjoy life. So if you feel the need and desire to respond to this post please put Yes Father in the subj a [...]',
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'Hello. I am a retired minister and I do miss my flock. I enjoy connecting and encouraging those who are unsure and lack the confidence to truly enjoy life. So if you feel the need and desire to respond to this post please put Yes Father in the subj a [...]'},
       u'title': u'Minister looking for you - m4w (NYC)',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Minister looking for you - m4w (NYC)'},
       u'updated': u'2016-04-12T07:59:05-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=11, tm_min=59, tm_sec=5, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/mnh/stp/5535240475.html',
       u'dc_type': u'text',
       u'id': u'http://newyork.craigslist.org/mnh/stp/5535240475.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/mnh/stp/5535240475.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/mnh/stp/5535240475.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T07:50:44-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=11, tm_min=50, tm_sec=44, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"Are you an interesting, educated, articulate, self-aware, curious, passionate, creative, attached female in NYC? \nHave you been in the same relationship for quite a while? \nHave you frequently thought about the fact that you're missing a close, speci [...]",
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"Are you an interesting, educated, articulate, self-aware, curious, passionate, creative, attached female in NYC? \nHave you been in the same relationship for quite a while? \nHave you frequently thought about the fact that you're missing a close, speci [...]"},
       u'title': u'Seeking a smart, interesting, attached female for close friendship - m4w (Manhattan)',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Seeking a smart, interesting, attached female for close friendship - m4w (Manhattan)'},
       u'updated': u'2016-04-12T07:50:44-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=11, tm_min=50, tm_sec=44, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/mnh/stp/5535255767.html',
       u'dc_type': u'text',
       u'id': u'http://newyork.craigslist.org/mnh/stp/5535255767.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/mnh/stp/5535255767.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/mnh/stp/5535255767.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T07:47:42-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=11, tm_min=47, tm_sec=42, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'Let me buy you coffee this morning on the west side. Smart , positive and encouraging guy. Tell me about your life. Looking forward to coffee, across from a warm smile',
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'Let me buy you coffee this morning on the west side. Smart , positive and encouraging guy. Tell me about your life. Looking forward to coffee, across from a warm smile'},
       u'title': u'Morning coffee - m4w (Upper West Side)',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Morning coffee - m4w (Upper West Side)'},
       u'updated': u'2016-04-12T07:47:42-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=11, tm_min=47, tm_sec=42, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/mnh/stp/5535238929.html',
       u'dc_type': u'text',
       u'enc_enclosure': {'resource': u'http://images.craigslist.org/00G0G_cfGsXAheC6_300x300.jpg',
        'type': u'image/jpeg'},
       u'id': u'http://newyork.craigslist.org/mnh/stp/5535238929.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/mnh/stp/5535238929.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/mnh/stp/5535238929.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T07:35:51-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=11, tm_min=35, tm_sec=51, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"I'm a sweet, compassionate girl that is just going through a bad patch. I am grateful for all of the good that I have but unfortunately I have pretty bad depression and anxiety and it's just so hard to cope lately. I'd love to spend some time with a  [...]",
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"I'm a sweet, compassionate girl that is just going through a bad patch. I am grateful for all of the good that I have but unfortunately I have pretty bad depression and anxiety and it's just so hard to cope lately. I'd love to spend some time with a  [...]"},
       u'title': u'Going through a hard time, visiting for work - w4w (Midtown)',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Going through a hard time, visiting for work - w4w (Midtown)'},
       u'updated': u'2016-04-12T07:35:51-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=11, tm_min=35, tm_sec=51, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/brk/stp/5516082933.html',
       u'dc_type': u'text',
       u'id': u'http://newyork.craigslist.org/brk/stp/5516082933.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/brk/stp/5516082933.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/brk/stp/5516082933.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T07:22:56-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=11, tm_min=22, tm_sec=56, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'--------------------------------------------------------------------------------------------------------------------------- \n37 yr old white guy married and extremely unhappy looking to get out and see a movie or something and have a little fun \nif y [...]',
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'--------------------------------------------------------------------------------------------------------------------------- \n37 yr old white guy married and extremely unhappy looking to get out and see a movie or something and have a little fun \nif y [...]'},
       u'title': u'Married & Unhappy - m4w',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Married & Unhappy - m4w'},
       u'updated': u'2016-04-12T07:22:56-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=11, tm_min=22, tm_sec=56, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/brk/stp/5505960931.html',
       u'dc_type': u'text',
       u'enc_enclosure': {'resource': u'http://images.craigslist.org/00e0e_bQScnNbhHL_300x300.jpg',
        'type': u'image/jpeg'},
       u'id': u'http://newyork.craigslist.org/brk/stp/5505960931.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/brk/stp/5505960931.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/brk/stp/5505960931.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T07:01:54-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=11, tm_min=1, tm_sec=54, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"( FEMALES ONLY ) \nbrown eyes,brown skin,165 lbs,5'7,6 pack abs,passionate,romantic,34 years old,got my own apt,business owner,out going,own car. . . I am a good friend to all & a good listener. I am very out going and like to go to places as in- beac [...]",
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"( FEMALES ONLY ) \nbrown eyes,brown skin,165 lbs,5'7,6 pack abs,passionate,romantic,34 years old,got my own apt,business owner,out going,own car. . . I am a good friend to all & a good listener. I am very out going and like to go to places as in- beac [...]"},
       u'title': u'Going with the flow for 2016 - m4w (new york city)',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Going with the flow for 2016 - m4w (new york city)'},
       u'updated': u'2016-04-12T07:01:54-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=11, tm_min=1, tm_sec=54, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/mnh/stp/5535215216.html',
       u'dc_type': u'text',
       u'enc_enclosure': {'resource': u'http://images.craigslist.org/01515_7uJPlT1p83a_300x300.jpg',
        'type': u'image/jpeg'},
       u'id': u'http://newyork.craigslist.org/mnh/stp/5535215216.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/mnh/stp/5535215216.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/mnh/stp/5535215216.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T06:54:02-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=10, tm_min=54, tm_sec=2, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'I am looking to meet a young King to worship and serve. I am not looking for sex. It is understood that You are superior. If you are a trim white male between 18 and 28 please allow me to kiss your feet and serve as your footrest. \nI am a masculine,  [...]',
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'I am looking to meet a young King to worship and serve. I am not looking for sex. It is understood that You are superior. If you are a trim white male between 18 and 28 please allow me to kiss your feet and serve as your footrest. \nI am a masculine,  [...]'},
       u'title': u'Worshipping at the Feet of a Young King 18-28 (manhattan brooklyn queens ny/nj)',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Worshipping at the Feet of a Young King 18-28 (manhattan brooklyn queens ny/nj)'},
       u'updated': u'2016-04-12T06:54:02-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=10, tm_min=54, tm_sec=2, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/mnh/stp/5530176616.html',
       u'dc_type': u'text',
       u'enc_enclosure': {'resource': u'http://images.craigslist.org/00o0o_9wGVvRGyt11_300x300.jpg',
        'type': u'image/jpeg'},
       u'id': u'http://newyork.craigslist.org/mnh/stp/5530176616.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/mnh/stp/5530176616.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/mnh/stp/5530176616.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T06:30:20-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=10, tm_min=30, tm_sec=20, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'Who more likely clicked on me? Women who value perfect grammar, or clean vaginas? Discuss...... \nNow that I grabbed your attention, how can I keep it? Being a stage "performer," I am a master of gripping AND keeping attention. It\'s a skill-set, embed [...]',
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'Who more likely clicked on me? Women who value perfect grammar, or clean vaginas? Discuss...... \nNow that I grabbed your attention, how can I keep it? Being a stage "performer," I am a master of gripping AND keeping attention. It\'s a skill-set, embed [...]'},
       u'title': u'I Swear on My Perfect Grammar That I Will Not Give You an STD. - m4w (Greenwich Village)',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'I Swear on My Perfect Grammar That I Will Not Give You an STD. - m4w (Greenwich Village)'},
       u'updated': u'2016-04-12T06:30:20-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=10, tm_min=30, tm_sec=20, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/fct/stp/5517933281.html',
       u'dc_type': u'text',
       u'id': u'http://newyork.craigslist.org/fct/stp/5517933281.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/fct/stp/5517933281.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/fct/stp/5517933281.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T06:21:03-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=10, tm_min=21, tm_sec=3, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'free summer housing offered for helpful ,discreet ,masculin dude in private room in vacant property in exchange of cleaning around house . \nsend pic/ stats for response. \nput your favorite color in title .',
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'free summer housing offered for helpful ,discreet ,masculin dude in private room in vacant property in exchange of cleaning around house . \nsend pic/ stats for response. \nput your favorite color in title .'},
       u'title': u'Free summer housing for helpful   ! - m4m (upper Manhattan)',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Free summer housing for helpful   ! - m4m (upper Manhattan)'},
       u'updated': u'2016-04-12T06:21:03-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=10, tm_min=21, tm_sec=3, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/stn/stp/5531287034.html',
       u'dc_type': u'text',
       u'enc_enclosure': {'resource': u'http://images.craigslist.org/00j0j_5b8ejNf8i25_300x300.jpg',
        'type': u'image/jpeg'},
       u'id': u'http://newyork.craigslist.org/stn/stp/5531287034.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/stn/stp/5531287034.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/stn/stp/5531287034.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T06:19:29-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=10, tm_min=19, tm_sec=29, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"Located (south beach area ) \nI Drive Drive Drive \nI'm 19 Hispanic/Black \n9.5 inches \nHIV NEGATIVE NO DISEASE I answer every email . Just make it clear what up!",
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"Located (south beach area ) \nI Drive Drive Drive \nI'm 19 Hispanic/Black \n9.5 inches \nHIV NEGATIVE NO DISEASE I answer every email . Just make it clear what up!"},
       u'title': u'Quickly - m4m',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Quickly - m4m'},
       u'updated': u'2016-04-12T06:19:29-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=10, tm_min=19, tm_sec=29, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/mnh/stp/5535189355.html',
       u'dc_type': u'text',
       u'enc_enclosure': {'resource': u'http://images.craigslist.org/00D0D_5xzBVqTffPV_300x300.jpg',
        'type': u'image/jpeg'},
       u'id': u'http://newyork.craigslist.org/mnh/stp/5535189355.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/mnh/stp/5535189355.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/mnh/stp/5535189355.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T05:42:46-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=9, tm_min=42, tm_sec=46, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'R/X Viagr@ \n(5 seven 0) eight 6 one 6 three four three \nGoing Quick \n100mg \n4 p/I/l/l bundle for $100.00 \na single p/i/l/l is $25.00 \n(5 seven 0) eight 6 one 6 three four three',
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'R/X Viagr@ \n(5 seven 0) eight 6 one 6 three four three \nGoing Quick \n100mg \n4 p/I/l/l bundle for $100.00 \na single p/i/l/l is $25.00 \n(5 seven 0) eight 6 one 6 three four three'},
       u'title': u'Male Performance Enhancer Going FAST - m4mw',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Male Performance Enhancer Going FAST - m4mw'},
       u'updated': u'2016-04-12T05:42:46-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=9, tm_min=42, tm_sec=46, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/mnh/stp/5535198330.html',
       u'dc_type': u'text',
       u'id': u'http://newyork.craigslist.org/mnh/stp/5535198330.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/mnh/stp/5535198330.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/mnh/stp/5535198330.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T05:28:55-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=9, tm_min=28, tm_sec=55, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'ALL LIMITS ABSOLUTELY RESPECTED! \nGood-looking, discreet and respectful experienced white professional seeks attractive, kind woman interested in adult breastfeeding (wet or dry), either on a one-time or ongoing basis. For me, milk or no milk, the ex [...]',
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'ALL LIMITS ABSOLUTELY RESPECTED! \nGood-looking, discreet and respectful experienced white professional seeks attractive, kind woman interested in adult breastfeeding (wet or dry), either on a one-time or ongoing basis. For me, milk or no milk, the ex [...]'},
       u'title': u'Platonic Breastfeeding -- Adult Nursing - m4w (Midtown)',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Platonic Breastfeeding -- Adult Nursing - m4w (Midtown)'},
       u'updated': u'2016-04-12T05:28:55-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=9, tm_min=28, tm_sec=55, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/wch/stp/5522003835.html',
       u'dc_type': u'text',
       u'id': u'http://newyork.craigslist.org/wch/stp/5522003835.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/wch/stp/5522003835.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/wch/stp/5522003835.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T04:38:32-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=8, tm_min=38, tm_sec=32, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"Fun easy going guy here \nLooking to chat with a female so we can push \nEach other to becoming mega rich \nIt's possible \nLet's talk",
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"Fun easy going guy here \nLooking to chat with a female so we can push \nEach other to becoming mega rich \nIt's possible \nLet's talk"},
       u'title': u'Brainstorm on pushing ourselves to getting rich - m4w (Westchester)',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Brainstorm on pushing ourselves to getting rich - m4w (Westchester)'},
       u'updated': u'2016-04-12T04:38:32-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=8, tm_min=38, tm_sec=32, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/lgi/stp/5527189763.html',
       u'dc_type': u'text',
       u'id': u'http://newyork.craigslist.org/lgi/stp/5527189763.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/lgi/stp/5527189763.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/lgi/stp/5527189763.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T03:26:46-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=7, tm_min=26, tm_sec=46, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"dude here looking to chat with a dudedette or better yet with someone from the opposite sex, so if you are a dude pay attention, I have dick and balls looking for someone who doesn't have that... got it? \nagain... hey you... yeah, stop jerking off an [...]",
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"dude here looking to chat with a dudedette or better yet with someone from the opposite sex, so if you are a dude pay attention, I have dick and balls looking for someone who doesn't have that... got it? \nagain... hey you... yeah, stop jerking off an [...]"},
       u'title': u'long day! - m4w (suffolk)',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'long day! - m4w (suffolk)'},
       u'updated': u'2016-04-12T03:26:46-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=7, tm_min=26, tm_sec=46, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://newyork.craigslist.org/brk/stp/5535167105.html',
       u'dc_type': u'text',
       u'id': u'http://newyork.craigslist.org/brk/stp/5535167105.html',
       u'language': u'en-us',
       u'link': u'http://newyork.craigslist.org/brk/stp/5535167105.html',
       u'links': [{u'href': u'http://newyork.craigslist.org/brk/stp/5535167105.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T03:20:29-04:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=7, tm_min=20, tm_sec=29, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'Seeking nice american female for business proposal asap, must be trustworthy, honest working citizen.',
       u'summary_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'Seeking nice american female for business proposal asap, must be trustworthy, honest working citizen.'},
       u'title': u'Bonding business proposal - m4w (NYC)',
       u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Bonding business proposal - m4w (NYC)'},
       u'updated': u'2016-04-12T03:20:29-04:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=7, tm_min=20, tm_sec=29, tm_wday=1, tm_yday=103, tm_isdst=0)}],
     'feed': {u'author': u'robot@craigslist.org',
      u'author_detail': {u'email': u'robot@craigslist.org'},
      u'authors': [{u'email': u'robot@craigslist.org'}],
      u'dc_source': u'https://newyork.craigslist.org/search/stp?format=rss',
      u'dc_type': u'Collection',
      u'entries': u'',
      u'language': u'en-us',
      u'link': u'https://newyork.craigslist.org/search/stp',
      u'links': [{u'href': u'https://newyork.craigslist.org/search/stp',
        u'rel': u'alternate',
        u'type': u'text/html'}],
      u'publisher': u'robot@craigslist.org',
      u'publisher_detail': {u'email': u'robot@craigslist.org'},
      u'rdf_li': {'rdf:resource': u'http://newyork.craigslist.org/brk/stp/5535167105.html'},
      u'rdf_seq': u'',
      u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
      u'rights_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
       u'language': None,
       u'type': u'text/html',
       u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
      u'subtitle': u'',
      u'subtitle_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
       u'language': None,
       u'type': u'text/html',
       u'value': u''},
      u'sy_updatebase': u'2016-04-12T09:15:29-04:00',
      u'sy_updatefrequency': u'1',
      u'sy_updateperiod': u'hourly',
      u'title': u'craigslist new york | strictly platonic search',
      u'title_detail': {u'base': u'http://newyork.craigslist.org/search/stp?format=rss',
       u'language': None,
       u'type': u'text/plain',
       u'value': u'craigslist new york | strictly platonic search'}},
     'headers': {'cache-control': 'max-age=900, public',
      'content-encoding': 'gzip',
      'content-length': '5213',
      'content-type': 'application/rss+xml; charset=utf-8',
      'date': 'Tue, 12 Apr 2016 13:15:29 GMT',
      'expires': 'Tue, 12 Apr 2016 13:30:29 GMT',
      'last-modified': 'Tue, 12 Apr 2016 13:15:29 GMT',
      'server': 'Apache',
      'vary': 'Accept-Encoding',
      'x-frame-options': 'SAMEORIGIN'},
     u'href': u'http://newyork.craigslist.org/search/stp?format=rss',
     u'namespaces': {u'': u'http://purl.org/rss/1.0/',
      u'admin': u'http://webns.net/mvcb/',
      u'content': u'http://purl.org/rss/1.0/modules/content/',
      u'dc': u'http://purl.org/dc/elements/1.1/',
      u'dcterms': u'http://purl.org/dc/terms/',
      u'enc': u'http://purl.oclc.org/net/rss_2.0/enc#',
      u'ev': u'http://purl.org/rss/1.0/modules/event/',
      u'rdf': u'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
      u'sy': u'http://purl.org/rss/1.0/modules/syndication/',
      u'taxo': u'http://purl.org/rss/1.0/modules/taxonomy/'},
     u'status': 301,
     u'updated': 'Tue, 12 Apr 2016 13:15:29 GMT',
     u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=13, tm_min=15, tm_sec=29, tm_wday=1, tm_yday=103, tm_isdst=0),
     u'version': u'rss10'}




```python
# 샌프란시스코에 사는 사람들의 커뮤니티에 올라온 글을 불러옵니다. 마찬가지로 RSS를 활용합니다.

sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
sf
```




    {'bozo': False,
     u'encoding': u'utf-8',
     'entries': [{u'dc_source': u'http://sfbay.craigslist.org/sby/stp/5535237202.html',
       u'dc_type': u'text',
       u'id': u'http://sfbay.craigslist.org/sby/stp/5535237202.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/sby/stp/5535237202.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/sby/stp/5535237202.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T04:15:42-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=11, tm_min=15, tm_sec=42, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"When I read these posts, or read responses to these posts, I always get the sense of deja vu. Why not start this relationship off by telling each other three things about eachother other people wouldn't think to say. (Do you know how many thirty-some [...]",
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"When I read these posts, or read responses to these posts, I always get the sense of deja vu. Why not start this relationship off by telling each other three things about eachother other people wouldn't think to say. (Do you know how many thirty-some [...]"},
       u'title': u'Tell me three things - w4m (sunnyvale)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Tell me three things - w4m (sunnyvale)'},
       u'updated': u'2016-04-12T04:15:42-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=11, tm_min=15, tm_sec=42, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/eby/stp/5535190268.html',
       u'dc_type': u'text',
       u'id': u'http://sfbay.craigslist.org/eby/stp/5535190268.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/eby/stp/5535190268.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/eby/stp/5535190268.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T03:52:30-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=10, tm_min=52, tm_sec=30, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'Hey you stop right there, if I was you I wouldnt click back I would actually be ready to click reply and this is why. \nI am going to be straight forward with you, I believe that I am the one you need in your life as a friend not only for the fact tha [...]',
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'Hey you stop right there, if I was you I wouldnt click back I would actually be ready to click reply and this is why. \nI am going to be straight forward with you, I believe that I am the one you need in your life as a friend not only for the fact tha [...]'},
       u'title': u"6'2 Chocolate and Handsome - m4w (vallejo / benicia)",
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u"6'2 Chocolate and Handsome - m4w (vallejo / benicia)"},
       u'updated': u'2016-04-12T03:52:30-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=10, tm_min=52, tm_sec=30, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/eby/stp/5535188849.html',
       u'dc_type': u'text',
       u'enc_enclosure': {'resource': u'http://images.craigslist.org/00J0J_ikJ7eLk9ipT_300x300.jpg',
        'type': u'image/jpeg'},
       u'id': u'http://sfbay.craigslist.org/eby/stp/5535188849.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/eby/stp/5535188849.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/eby/stp/5535188849.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T03:50:41-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=10, tm_min=50, tm_sec=41, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"I am just looking for a real friend and you would think that would be easy but it's not because too many women play games or lie on here. Now I am going to be straight up though I am looking for just a friend that doesn't involve sex but I would like [...]",
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"I am just looking for a real friend and you would think that would be easy but it's not because too many women play games or lie on here. Now I am going to be straight up though I am looking for just a friend that doesn't involve sex but I would like [...]"},
       u'title': u"Why can't I just find a real one ... - m4w (vallejo / benicia)",
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u"Why can't I just find a real one ... - m4w (vallejo / benicia)"},
       u'updated': u'2016-04-12T03:50:41-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=10, tm_min=50, tm_sec=41, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/eby/stp/5535186981.html',
       u'dc_type': u'text',
       u'id': u'http://sfbay.craigslist.org/eby/stp/5535186981.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/eby/stp/5535186981.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/eby/stp/5535186981.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T03:11:28-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=10, tm_min=11, tm_sec=28, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"It has been pretty cold lately hasn't it? Wouldn't you like a big strong man to hold you tight in his arms warming you up? Wouldn't you like a talk and muscular man that can give you a nice and tight hug making you feel secure and safe in his arms? S [...]",
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"It has been pretty cold lately hasn't it? Wouldn't you like a big strong man to hold you tight in his arms warming you up? Wouldn't you like a talk and muscular man that can give you a nice and tight hug making you feel secure and safe in his arms? S [...]"},
       u'title': u'SBM that will keep you warm - m4w (concord / pleasant hill / martinez)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'SBM that will keep you warm - m4w (concord / pleasant hill / martinez)'},
       u'updated': u'2016-04-12T03:11:28-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=10, tm_min=11, tm_sec=28, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/eby/stp/5535189941.html',
       u'dc_type': u'text',
       u'id': u'http://sfbay.craigslist.org/eby/stp/5535189941.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/eby/stp/5535189941.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/eby/stp/5535189941.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T02:49:10-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=9, tm_min=49, tm_sec=10, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"I'm twenty eight, no kids, never married I like to play sports, listen to music, watch movies etc. . . Despite how I write I'm a really silly guy and don't try to take too many things too seriously because of this, plus life's short. I live in Vallej [...]",
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"I'm twenty eight, no kids, never married I like to play sports, listen to music, watch movies etc. . . Despite how I write I'm a really silly guy and don't try to take too many things too seriously because of this, plus life's short. I live in Vallej [...]"},
       u'title': u'good guy seeking a sweet biggg booty- curvy thick chick - m4w (pittsburg / antioch)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'good guy seeking a sweet biggg booty- curvy thick chick - m4w (pittsburg / antioch)'},
       u'updated': u'2016-04-12T02:49:10-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=9, tm_min=49, tm_sec=10, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/eby/stp/5535178532.html',
       u'dc_type': u'text',
       u'id': u'http://sfbay.craigslist.org/eby/stp/5535178532.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/eby/stp/5535178532.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/eby/stp/5535178532.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T01:45:16-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=8, tm_min=45, tm_sec=16, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"I like a woman who is thick a nice curvy shape\nA girl that has weight but most in the right place (wink)\nShe is looking for a friend not for a soul mate\nShe makes time to hang with me even with a full plate;\n'm not looking for sex I'm not horny like  [...]",
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"I like a woman who is thick a nice curvy shape\nA girl that has weight but most in the right place (wink)\nShe is looking for a friend not for a soul mate\nShe makes time to hang with me even with a full plate;\n'm not looking for sex I'm not horny like  [...]"},
       u'title': u'Thick Woman Friend Rap - m4w (vallejo / benicia)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Thick Woman Friend Rap - m4w (vallejo / benicia)'},
       u'updated': u'2016-04-12T01:45:16-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=8, tm_min=45, tm_sec=16, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/eby/stp/5498701279.html',
       u'dc_type': u'text',
       u'id': u'http://sfbay.craigslist.org/eby/stp/5498701279.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/eby/stp/5498701279.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/eby/stp/5498701279.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T01:30:13-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=8, tm_min=30, tm_sec=13, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"I'm twenty eight, no kids, never married I like to play sports, listen to music, watch movies etc. . . Despite how I write I'm a really silly guy and don't try to take too many things too seriously because of this, plus life's short. I live in Vallej [...]",
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"I'm twenty eight, no kids, never married I like to play sports, listen to music, watch movies etc. . . Despite how I write I'm a really silly guy and don't try to take too many things too seriously because of this, plus life's short. I live in Vallej [...]"},
       u'title': u'good guy seeking a sweet biggg booty- curvy thick chick - m4w (vallejo / benicia)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'good guy seeking a sweet biggg booty- curvy thick chick - m4w (vallejo / benicia)'},
       u'updated': u'2016-04-12T01:30:13-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=8, tm_min=30, tm_sec=13, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/eby/stp/5535181762.html',
       u'dc_type': u'text',
       u'id': u'http://sfbay.craigslist.org/eby/stp/5535181762.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/eby/stp/5535181762.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/eby/stp/5535181762.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T01:27:25-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=8, tm_min=27, tm_sec=25, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'Why is it so hard to find a friend of the opposite sex? I have been living in the bay area for my whole life and it is next to impossible to find that 1 female friend i can talk to, laugh and joke with, talk shit and actually enjoy each others compan [...]',
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'Why is it so hard to find a friend of the opposite sex? I have been living in the bay area for my whole life and it is next to impossible to find that 1 female friend i can talk to, laugh and joke with, talk shit and actually enjoy each others compan [...]'},
       u'title': u'best friend of opposite sex - m4w (concord / pleasant hill / martinez)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'best friend of opposite sex - m4w (concord / pleasant hill / martinez)'},
       u'updated': u'2016-04-12T01:27:25-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=8, tm_min=27, tm_sec=25, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/eby/stp/5528307772.html',
       u'dc_type': u'text',
       u'id': u'http://sfbay.craigslist.org/eby/stp/5528307772.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/eby/stp/5528307772.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/eby/stp/5528307772.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T01:06:51-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=8, tm_min=6, tm_sec=51, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'Hey how are you doing I hope your Holidays were good and I am just posting an ad on here because I am looking to change things for the New Year and that means I am trying to gain a new friend for the New Year. I am looking for somebody who is real co [...]',
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'Hey how are you doing I hope your Holidays were good and I am just posting an ad on here because I am looking to change things for the New Year and that means I am trying to gain a new friend for the New Year. I am looking for somebody who is real co [...]'},
       u'title': u'Fun loving guy looking for fun loving girl - m4w (vallejo / benicia)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Fun loving guy looking for fun loving girl - m4w (vallejo / benicia)'},
       u'updated': u'2016-04-12T01:06:51-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=8, tm_min=6, tm_sec=51, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/eby/stp/5514453789.html',
       u'dc_type': u'text',
       u'id': u'http://sfbay.craigslist.org/eby/stp/5514453789.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/eby/stp/5514453789.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/eby/stp/5514453789.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T00:45:22-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=7, tm_min=45, tm_sec=22, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"As you can tell from my title I am looking for a thick curvy girl. I know that some guys don't really like them, but they just don't understand how great a curvy girl really is :). So about me, I am 28 years old working part time right now, and plann [...]",
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"As you can tell from my title I am looking for a thick curvy girl. I know that some guys don't really like them, but they just don't understand how great a curvy girl really is :). So about me, I am 28 years old working part time right now, and plann [...]"},
       u'title': u'I like thick curvy girls plain and simple - m4w (concord / pleasant hill / martinez)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'I like thick curvy girls plain and simple - m4w (concord / pleasant hill / martinez)'},
       u'updated': u'2016-04-12T00:45:22-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=7, tm_min=45, tm_sec=22, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/sby/stp/5531009293.html',
       u'dc_type': u'text',
       u'id': u'http://sfbay.craigslist.org/sby/stp/5531009293.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/sby/stp/5531009293.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/sby/stp/5531009293.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T00:41:27-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=7, tm_min=41, tm_sec=27, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'Seems like most of my good friends are in relationships or are married... i would love to meet some new women, who have their act together, are real and fun.. \nmyself, i love sports including golf,hiking,tennis, raquetball...love music, listening and [...]',
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'Seems like most of my good friends are in relationships or are married... i would love to meet some new women, who have their act together, are real and fun.. \nmyself, i love sports including golf,hiking,tennis, raquetball...love music, listening and [...]'},
       u'title': u'looking to make some new female friends - m4w (los gatos)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'looking to make some new female friends - m4w (los gatos)'},
       u'updated': u'2016-04-12T00:41:27-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=7, tm_min=41, tm_sec=27, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/eby/stp/5490112754.html',
       u'dc_type': u'text',
       u'id': u'http://sfbay.craigslist.org/eby/stp/5490112754.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/eby/stp/5490112754.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/eby/stp/5490112754.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T00:24:45-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=7, tm_min=24, tm_sec=45, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'Hey how are you doing I hope your new year has been good so far and if not maybe I can change that by offering you 2 choices. A: You can decide to get to know me and that would be a very smart choice on your part because not only am I tall dark and h [...]',
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'Hey how are you doing I hope your new year has been good so far and if not maybe I can change that by offering you 2 choices. A: You can decide to get to know me and that would be a very smart choice on your part because not only am I tall dark and h [...]'},
       u'title': u'You need me in your life trust me - m4w (fairfield / vacaville)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'You need me in your life trust me - m4w (fairfield / vacaville)'},
       u'updated': u'2016-04-12T00:24:45-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=7, tm_min=24, tm_sec=45, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/sby/stp/5535165832.html',
       u'dc_type': u'text',
       u'id': u'http://sfbay.craigslist.org/sby/stp/5535165832.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/sby/stp/5535165832.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/sby/stp/5535165832.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T00:15:59-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=7, tm_min=15, tm_sec=59, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"Trying meet new friends! \nI'm a female I'm 30 and love \nTo draw. If your my age ranged send a pic so I know \nYour real!",
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"Trying meet new friends! \nI'm a female I'm 30 and love \nTo draw. If your my age ranged send a pic so I know \nYour real!"},
       u'title': u'Making friends - w4m (milpitas)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Making friends - w4m (milpitas)'},
       u'updated': u'2016-04-12T00:15:59-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=7, tm_min=15, tm_sec=59, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/eby/stp/5530535118.html',
       u'dc_type': u'text',
       u'id': u'http://sfbay.craigslist.org/eby/stp/5530535118.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/eby/stp/5530535118.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/eby/stp/5530535118.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T00:13:05-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=7, tm_min=13, tm_sec=5, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'Just like the title says I am looking for a female patna in crime who is willing to laugh have fun and going on adventures and hopefully take it back to the fun childhood times lol. I am a grown man and I am fun, funny, goofy, playful, and flirtatiou [...]',
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'Just like the title says I am looking for a female patna in crime who is willing to laugh have fun and going on adventures and hopefully take it back to the fun childhood times lol. I am a grown man and I am fun, funny, goofy, playful, and flirtatiou [...]'},
       u'title': u'Female Patna in Crime - m4w (fairfield / vacaville)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Female Patna in Crime - m4w (fairfield / vacaville)'},
       u'updated': u'2016-04-12T00:13:05-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=7, tm_min=13, tm_sec=5, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/eby/stp/5530537008.html',
       u'dc_type': u'text',
       u'id': u'http://sfbay.craigslist.org/eby/stp/5530537008.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/eby/stp/5530537008.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/eby/stp/5530537008.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-12T00:09:57-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=7, tm_min=9, tm_sec=57, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'Just like the title says I am looking for a female patna in crime who is willing to laugh have fun and going on adventures and hopefully take it back to the fun childhood times lol. I am a grown man and I am fun, funny, goofy, playful, and flirtatiou [...]',
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'Just like the title says I am looking for a female patna in crime who is willing to laugh have fun and going on adventures and hopefully take it back to the fun childhood times lol. I am a grown man and I am fun, funny, goofy, playful, and flirtatiou [...]'},
       u'title': u'Female Patna in Crime - m4w (concord / pleasant hill / martinez)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Female Patna in Crime - m4w (concord / pleasant hill / martinez)'},
       u'updated': u'2016-04-12T00:09:57-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=7, tm_min=9, tm_sec=57, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/eby/stp/5535160658.html',
       u'dc_type': u'text',
       u'id': u'http://sfbay.craigslist.org/eby/stp/5535160658.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/eby/stp/5535160658.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/eby/stp/5535160658.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-11T23:59:39-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=6, tm_min=59, tm_sec=39, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'Hi! How are you doing today? Want to get hang out and meet over a massage? I am a massage student and I do not know many people in the area because I have not lived here long. In order to get my LMT I need to practice! Since I do not know many people [...]',
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'Hi! How are you doing today? Want to get hang out and meet over a massage? I am a massage student and I do not know many people in the area because I have not lived here long. In order to get my LMT I need to practice! Since I do not know many people [...]'},
       u'title': u'What are you doing today??? - w4m (berkeley)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'What are you doing today??? - w4m (berkeley)'},
       u'updated': u'2016-04-11T23:59:39-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=6, tm_min=59, tm_sec=39, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/eby/stp/5535158911.html',
       u'dc_type': u'text',
       u'enc_enclosure': {'resource': u'http://images.craigslist.org/00Y0Y_gFMfrXfvF3c_300x300.jpg',
        'type': u'image/jpeg'},
       u'id': u'http://sfbay.craigslist.org/eby/stp/5535158911.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/eby/stp/5535158911.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/eby/stp/5535158911.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-11T23:53:59-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=6, tm_min=53, tm_sec=59, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"I'm here, online, bored and wanting/willing to chat. \nHere are my screen names if you care to chat with me: \nYahoo: deranged_lizzie_borden187 \nKik: zombie_poppet \nIf you're bored and you wanna chat, I'm on now. \nPLEASE live in California (I plan on m [...]",
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"I'm here, online, bored and wanting/willing to chat. \nHere are my screen names if you care to chat with me: \nYahoo: deranged_lizzie_borden187 \nKik: zombie_poppet \nIf you're bored and you wanna chat, I'm on now. \nPLEASE live in California (I plan on m [...]"},
       u'title': u'The inside is always the same... - w4m (hayward / castro valley)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'The inside is always the same... - w4m (hayward / castro valley)'},
       u'updated': u'2016-04-11T23:53:59-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=6, tm_min=53, tm_sec=59, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/eby/stp/5535151499.html',
       u'dc_type': u'text',
       u'id': u'http://sfbay.craigslist.org/eby/stp/5535151499.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/eby/stp/5535151499.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/eby/stp/5535151499.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-11T23:53:55-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=6, tm_min=53, tm_sec=55, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"Hi im a tall athletic but single male.drug free drama free looking for any type of lady who is able to host or would like to get a motel tonight for some fun. I'm open to car play since I can not host due to roomates. \nI'm clean and a pretty cool guy [...]",
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"Hi im a tall athletic but single male.drug free drama free looking for any type of lady who is able to host or would like to get a motel tonight for some fun. I'm open to car play since I can not host due to roomates. \nI'm clean and a pretty cool guy [...]"},
       u'title': u'Pc technician looking for love - m4w (oakland hills / mills)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Pc technician looking for love - m4w (oakland hills / mills)'},
       u'updated': u'2016-04-11T23:53:55-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=6, tm_min=53, tm_sec=55, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/sby/stp/5535090233.html',
       u'dc_type': u'text',
       u'enc_enclosure': {'resource': u'http://images.craigslist.org/00P0P_5oMv5JQfnht_300x300.jpg',
        'type': u'image/jpeg'},
       u'id': u'http://sfbay.craigslist.org/sby/stp/5535090233.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/sby/stp/5535090233.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/sby/stp/5535090233.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-11T23:27:51-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=6, tm_min=27, tm_sec=51, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"when I say not too extra, I mean a gay male that doesn't love drama. I just want a low key, fun, respectful gay male to befriend. I love gay men that are comfortable in their skin, and in spite of being teased, or laughed at, etc, maintain a positive [...]",
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"when I say not too extra, I mean a gay male that doesn't love drama. I just want a low key, fun, respectful gay male to befriend. I love gay men that are comfortable in their skin, and in spite of being teased, or laughed at, etc, maintain a positive [...]"},
       u'title': u"seeking a chill, black GAY male that isn't too extra - w4m (santa clara)",
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u"seeking a chill, black GAY male that isn't too extra - w4m (santa clara)"},
       u'updated': u'2016-04-11T23:27:51-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=6, tm_min=27, tm_sec=51, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/sby/stp/5516191352.html',
       u'dc_type': u'text',
       u'enc_enclosure': {'resource': u'http://images.craigslist.org/00I0I_11qqXVKAKCI_300x300.jpg',
        'type': u'image/jpeg'},
       u'id': u'http://sfbay.craigslist.org/sby/stp/5516191352.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/sby/stp/5516191352.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/sby/stp/5516191352.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-11T23:23:25-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=6, tm_min=23, tm_sec=25, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'Need a movie buddy. \nPut your favorite superhero on the subject line \nKik mikesoriano18 \nSnapchat Mike_kent18 \n# Six six nine two six two seven two five one',
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'Need a movie buddy. \nPut your favorite superhero on the subject line \nKik mikesoriano18 \nSnapchat Mike_kent18 \n# Six six nine two six two seven two five one'},
       u'title': u'Friends ? - m4w (san jose east)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Friends ? - m4w (san jose east)'},
       u'updated': u'2016-04-11T23:23:25-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=6, tm_min=23, tm_sec=25, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/sby/stp/5523928395.html',
       u'dc_type': u'text',
       u'id': u'http://sfbay.craigslist.org/sby/stp/5523928395.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/sby/stp/5523928395.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/sby/stp/5523928395.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-11T23:18:05-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=6, tm_min=18, tm_sec=5, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'Seeking a cute, fun, petite, mature student for friendship and maybe more. Prefer someone focused on school or work, not into the party or wild type. Attractive fit fun clean WM businessman type here for support, mentor, dinners, movies, shopping, th [...]',
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'Seeking a cute, fun, petite, mature student for friendship and maybe more. Prefer someone focused on school or work, not into the party or wild type. Attractive fit fun clean WM businessman type here for support, mentor, dinners, movies, shopping, th [...]'},
       u'title': u'Older WM for cute petite 18-21 yo - m4w (saratoga)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Older WM for cute petite 18-21 yo - m4w (saratoga)'},
       u'updated': u'2016-04-11T23:18:05-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=6, tm_min=18, tm_sec=5, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/sby/stp/5535145852.html',
       u'dc_type': u'text',
       u'id': u'http://sfbay.craigslist.org/sby/stp/5535145852.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/sby/stp/5535145852.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/sby/stp/5535145852.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-11T23:16:41-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=6, tm_min=16, tm_sec=41, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"Indian female, techie, between jobs. Looking to connect with others during weekdays for some motivation and people time. We can meet at the local library or coffee shop couple of times a week and get something productive done:) . If you're from a tec [...]",
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"Indian female, techie, between jobs. Looking to connect with others during weekdays for some motivation and people time. We can meet at the local library or coffee shop couple of times a week and get something productive done:) . If you're from a tec [...]"},
       u'title': u'Weekday Study Buddy / Coffee Shop /Library - w4ww (santa clara)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Weekday Study Buddy / Coffee Shop /Library - w4ww (santa clara)'},
       u'updated': u'2016-04-11T23:16:41-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=6, tm_min=16, tm_sec=41, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/pen/stp/5535136723.html',
       u'dc_type': u'text',
       u'id': u'http://sfbay.craigslist.org/pen/stp/5535136723.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/pen/stp/5535136723.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/pen/stp/5535136723.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-11T22:55:12-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=5, tm_min=55, tm_sec=12, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u'Pretty straight forward. I am looking for someone to hangout and grab dinner. No expectation, drama and issues. Just good company, good food and a relaxing time. I am an Asian-American male (Chinese), professionally employed, easy going. Prefer someo [...]',
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'Pretty straight forward. I am looking for someone to hangout and grab dinner. No expectation, drama and issues. Just good company, good food and a relaxing time. I am an Asian-American male (Chinese), professionally employed, easy going. Prefer someo [...]'},
       u'title': u'Hanging Out / Dinner buddy - m4w (san mateo)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Hanging Out / Dinner buddy - m4w (san mateo)'},
       u'updated': u'2016-04-11T22:55:12-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=5, tm_min=55, tm_sec=12, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/eby/stp/5534200406.html',
       u'dc_type': u'text',
       u'id': u'http://sfbay.craigslist.org/eby/stp/5534200406.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/eby/stp/5534200406.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/eby/stp/5534200406.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-11T22:54:58-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=5, tm_min=54, tm_sec=58, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"HELLO I'M LATINO I WAS WHIT A BBW BEFORE AND I WOULD LIKE TO TRY AGAIN SO I'M LOOCKING FOR A \nNICE BBW UNDER 40 DESCREETE.FUNNY, MUST LOVE SEX, BE HOT AND JUCY EMAIL ME WENT YOU READY YOUR PIC GETS MY, \nIN THE SUBJET PUY (WET PUSSY) I'LL KNOW YOU ARE [...]",
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"HELLO I'M LATINO I WAS WHIT A BBW BEFORE AND I WOULD LIKE TO TRY AGAIN SO I'M LOOCKING FOR A \nNICE BBW UNDER 40 DESCREETE.FUNNY, MUST LOVE SEX, BE HOT AND JUCY EMAIL ME WENT YOU READY YOUR PIC GETS MY, \nIN THE SUBJET PUY (WET PUSSY) I'LL KNOW YOU ARE [...]"},
       u'title': u'MERRIED LATINO LOOCKING FOR DESCREETE BBW UNDER 40 - m4w (dublin / pleasanton / livermore)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'MERRIED LATINO LOOCKING FOR DESCREETE BBW UNDER 40 - m4w (dublin / pleasanton / livermore)'},
       u'updated': u'2016-04-11T22:54:58-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=5, tm_min=54, tm_sec=58, tm_wday=1, tm_yday=103, tm_isdst=0)},
      {u'dc_source': u'http://sfbay.craigslist.org/sby/stp/5535133755.html',
       u'dc_type': u'text',
       u'id': u'http://sfbay.craigslist.org/sby/stp/5535133755.html',
       u'language': u'en-us',
       u'link': u'http://sfbay.craigslist.org/sby/stp/5535133755.html',
       u'links': [{u'href': u'http://sfbay.craigslist.org/sby/stp/5535133755.html',
         u'rel': u'alternate',
         u'type': u'text/html'}],
       u'published': u'2016-04-11T22:48:50-07:00',
       u'published_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=5, tm_min=48, tm_sec=50, tm_wday=1, tm_yday=103, tm_isdst=0),
       u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
       u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
       u'summary': u"Hi, \nI am a woman in my 60's and I am seeking friends to hang out with, go visit, have dinner, the park, ect . I am Mexican/English and I have alot of time on my hands that I would like to fill it with good friends and good times. I have a hip proble [...]",
       u'summary_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/html',
        u'value': u"Hi, \nI am a woman in my 60's and I am seeking friends to hang out with, go visit, have dinner, the park, ect . I am Mexican/English and I have alot of time on my hands that I would like to fill it with good friends and good times. I have a hip proble [...]"},
       u'title': u'Any  older  ladies out there seeking friends? - w4w (san jose east)',
       u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
        u'language': None,
        u'type': u'text/plain',
        u'value': u'Any  older  ladies out there seeking friends? - w4w (san jose east)'},
       u'updated': u'2016-04-11T22:48:50-07:00',
       u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=5, tm_min=48, tm_sec=50, tm_wday=1, tm_yday=103, tm_isdst=0)}],
     'feed': {u'author': u'robot@craigslist.org',
      u'author_detail': {u'email': u'robot@craigslist.org'},
      u'authors': [{u'email': u'robot@craigslist.org'}],
      u'dc_source': u'https://sfbay.craigslist.org/search/stp?format=rss',
      u'dc_type': u'Collection',
      u'entries': u'',
      u'language': u'en-us',
      u'link': u'https://sfbay.craigslist.org/search/stp',
      u'links': [{u'href': u'https://sfbay.craigslist.org/search/stp',
        u'rel': u'alternate',
        u'type': u'text/html'}],
      u'publisher': u'robot@craigslist.org',
      u'publisher_detail': {u'email': u'robot@craigslist.org'},
      u'rdf_li': {'rdf:resource': u'http://sfbay.craigslist.org/sby/stp/5535133755.html'},
      u'rdf_seq': u'',
      u'rights': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>',
      u'rights_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
       u'language': None,
       u'type': u'text/html',
       u'value': u'&copy; 2016 <span class="desktop">craigslist</span><span class="mobile">CL</span>'},
      u'subtitle': u'',
      u'subtitle_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
       u'language': None,
       u'type': u'text/html',
       u'value': u''},
      u'sy_updatebase': u'2016-04-12T06:15:30-07:00',
      u'sy_updatefrequency': u'1',
      u'sy_updateperiod': u'hourly',
      u'title': u'craigslist SF bay area | strictly platonic search',
      u'title_detail': {u'base': u'http://sfbay.craigslist.org/search/stp?format=rss',
       u'language': None,
       u'type': u'text/plain',
       u'value': u'craigslist SF bay area | strictly platonic search'}},
     'headers': {'cache-control': 'max-age=900, public',
      'content-encoding': 'gzip',
      'content-length': '5457',
      'content-type': 'application/rss+xml; charset=utf-8',
      'date': 'Tue, 12 Apr 2016 13:15:30 GMT',
      'expires': 'Tue, 12 Apr 2016 13:30:30 GMT',
      'last-modified': 'Tue, 12 Apr 2016 13:15:30 GMT',
      'server': 'Apache',
      'vary': 'Accept-Encoding',
      'x-frame-options': 'SAMEORIGIN'},
     u'href': u'http://sfbay.craigslist.org/search/stp?format=rss',
     u'namespaces': {u'': u'http://purl.org/rss/1.0/',
      u'admin': u'http://webns.net/mvcb/',
      u'content': u'http://purl.org/rss/1.0/modules/content/',
      u'dc': u'http://purl.org/dc/elements/1.1/',
      u'dcterms': u'http://purl.org/dc/terms/',
      u'enc': u'http://purl.oclc.org/net/rss_2.0/enc#',
      u'ev': u'http://purl.org/rss/1.0/modules/event/',
      u'rdf': u'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
      u'sy': u'http://purl.org/rss/1.0/modules/syndication/',
      u'taxo': u'http://purl.org/rss/1.0/modules/taxonomy/'},
     u'status': 301,
     u'updated': 'Tue, 12 Apr 2016 13:15:30 GMT',
     u'updated_parsed': time.struct_time(tm_year=2016, tm_mon=4, tm_mday=12, tm_hour=13, tm_min=15, tm_sec=30, tm_wday=1, tm_yday=103, tm_isdst=0),
     u'version': u'rss10'}




```python
# 두 지역의 단어 사용 패턴의 차이를 수치로 출력합니다.

vocabList, pSF, pNY = naivebayes.localWords(ny, sf)
```

    the error rate is:  0.6
    


```python
# 재학습한 후 출력합니다.

vocabList, pSF, pNY = naivebayes.localWords(ny, sf)
```

    the error rate is:  0.55
    


```python
# 각 지역에서 자주 사용되는 것으로 추정되는 단어들을 출력합니다.

naivebayes.getTopWords(ny, sf)
```

    the error rate is:  0.35
    ***This is San Francisco***
    twenty
    hey
    tall
    seeking
    why
    pretty
    seems
    has
    trying
    when
    all
    befriend
    chinese
    lack
    four
    asian
    grateful
    skin
    milk
    issues
    relationships
    buddy
    father
    young
    send
    masculine
    under
    smile
    must
    brown
    woman
    very
    choice
    every
    decide
    telling
    word
    cool
    school
    respectful
    compassionate
    posts
    try
    quick
    guy
    enjoy
    says
    speci
    tec
    past
    across
    likely
    professionally
    click
    depression
    drinks
    even
    what
    abs
    reply
    compan
    seeks
    hasn
    full
    unsure
    desires
    ranged
    men
    wild
    let
    others
    clicked
    along
    strong
    change
    fantasies
    dry
    great
    kids
    northern
    thirty
    pics
    mentor
    social
    makes
    massage
    retired
    secure
    hiking
    extra
    prefer
    private
    mikesoriano18
    names
    support
    flagging
    confidence
    from
    positive
    dude
    arms
    next
    live
    raquetball
    music
    taken
    type
    until
    today
    more
    females
    posting
    successful
    company
    yahoo
    aware
    warm
    adult
    hold
    women
    involve
    keeping
    balls
    join
    room
    car
    work
    soul
    movies
    nine
    making
    male
    viagr
    performer
    want
    give
    share
    high
    something
    masculin
    sense
    times
    drama
    needs
    productive
    dinners
    six
    write
    hot
    low
    answer
    nyc
    beach
    indian
    petite
    stop
    mad
    plann
    guys
    response
    man
    short
    maybe
    explicit
    childhood
    wink
    maintain
    allow
    keeps
    order
    talk
    cute
    mexican
    help
    over
    years
    playful
    held
    including
    cold
    horny
    before
    perfect
    superior
    roomates
    fit
    located
    somewhere
    interesting
    appetizers
    actually
    late
    willing
    jucy
    wouldn
    them
    somebody
    food
    jerking
    safe
    snapchat
    they
    hands
    now
    discuss
    term
    gets
    grammar
    name
    always
    goofy
    each
    went
    friendship
    side
    mean
    weight
    doing
    house
    hard
    yeah
    used
    connect
    year
    our
    whit
    bay
    really
    living
    open
    motel
    since
    seriously
    mate
    got
    lmt
    free
    quite
    ect
    california
    puy
    put
    hand
    care
    visit
    could
    keep
    american
    place
    isn
    deja
    think
    south
    tennis
    platonic
    feel
    encouraging
    one
    feet
    rant
    done
    lol
    owner
    impossible
    thick
    miss
    little
    toy
    start
    weekdays
    their
    master
    too
    white
    listen
    gripping
    hug
    relationship
    serve
    part
    believe
    king
    kind
    anxiety
    unfortunately
    midtown
    future
    silly
    were
    lbs
    mine
    comfortable
    tonight
    say
    hangout
    pay
    need
    any
    lie
    offering
    person
    self
    cope
    able
    lik
    also
    take
    proble
    online
    wanting
    latino
    play
    experienced
    opposite
    park
    most
    eight
    plan
    superhero
    extremely
    don
    eachother
    drive
    clean
    businessman
    fact
    shop
    golf
    random
    effo
    find
    busy
    grabbed
    spite
    title
    dick
    unhappy
    crime
    athletic
    only
    black
    employed
    local
    hope
    hip
    plate
    his
    hiv
    means
    hopefully
    watch
    expectation
    truly
    listener
    despite
    during
    having
    footrest
    beac
    girl
    course
    married
    patch
    bad
    stuff
    she
    through
    grab
    respond
    set
    sex
    see
    prosper
    close
    yes
    subject
    stats
    movie
    exchange
    please
    worship
    between
    probably
    encounters
    email
    never
    jobs
    responses
    missing
    zombie_poppet
    screen
    attention
    key
    come
    problem
    limits
    many
    adventures
    drug
    etc
    games
    alot
    focused
    whole
    bell
    color
    sweet
    motivation
    whatever
    100mg
    simply
    laugh
    100
    better
    basis
    west
    due
    been
    much
    far
    bbw
    proposal
    direction
    educated
    shopping
    gay
    offered
    wet
    new
    understand
    muscular
    smoking
    those
    honest
    myself
    mega
    these
    straight
    value
    choices
    while
    ongoing
    joke
    situation
    property
    pack
    seven
    hoping
    coffee
    respected
    someone
    128529
    ready
    grown
    funny
    things
    make
    same
    trip
    party
    several
    week
    practice
    hang
    holidays
    kik
    cleaning
    student
    frequently
    warming
    dark
    off
    vacant
    thought
    hispanic
    someo
    english
    spend
    tha
    meet
    breastfeeding
    summer
    kiss
    being
    descreete
    shape
    skill
    yet
    generous
    other
    attractive
    easy
    citizen
    smart
    hater
    real
    around
    read
    big
    couple
    laughed
    possible
    early
    discreet
    five
    apt
    listening
    bit
    lady
    desire
    helpful
    either
    passionate
    two
    because
    old
    people
    absolutely
    some
    back
    library
    understood
    curious
    165
    happens
    scale
    though
    creative
    connecting
    subjet
    three
    loocking
    business
    asap
    host
    post
    stage
    about
    working
    wouldnt
    getting
    vaginas
    favorite
    trustworthy
    dinner
    plus
    afternoon
    act
    wanna
    own
    techie
    pussy
    into
    articulate
    negative
    right
    sports
    subj
    area
    vallej
    housing
    there
    long
    experices
    patna
    way
    forward
    bored
    was
    buy
    inches
    becoming
    gain
    mike_kent18
    line
    romantic
    places
    morning
    attached
    tell
    mature
    embed
    clear
    trim
    pic
    doesn
    disease
    single
    lived
    chat
    teased
    flock
    shit
    fill
    again
    spell
    relaxing
    lately
    tight
    interested
    details
    evenings
    out
    manhattan
    nice
    draw
    eyes
    flirtatiou
    bundle
    professional
    friends
    younger
    deranged_lizzie_borden187
    rich
    age
    together
    dudedette
    curvy
    push
    hello
    minister
    ***This is New York***
    let
    dude
    viagr
    located
    seeking
    midtown
    sweet
    old
    all
    befriend
    chinese
    lack
    four
    asian
    grateful
    skin
    milk
    issues
    relationships
    buddy
    father
    young
    send
    masculine
    under
    smile
    must
    brown
    woman
    very
    choice
    every
    decide
    telling
    word
    cool
    school
    respectful
    compassionate
    posts
    try
    quick
    guy
    enjoy
    says
    speci
    tec
    past
    across
    likely
    professionally
    click
    depression
    drinks
    even
    what
    abs
    reply
    compan
    seeks
    hasn
    full
    unsure
    desires
    ranged
    men
    wild
    others
    clicked
    along
    strong
    change
    fantasies
    dry
    great
    kids
    northern
    thirty
    pics
    mentor
    social
    makes
    massage
    retired
    secure
    hiking
    extra
    prefer
    private
    mikesoriano18
    names
    support
    flagging
    confidence
    from
    positive
    arms
    next
    live
    raquetball
    music
    taken
    type
    until
    today
    more
    females
    posting
    successful
    company
    yahoo
    aware
    warm
    adult
    hold
    women
    involve
    keeping
    balls
    join
    room
    car
    work
    soul
    movies
    nine
    making
    male
    performer
    want
    give
    share
    high
    something
    masculin
    sense
    times
    drama
    needs
    productive
    dinners
    six
    write
    hot
    low
    answer
    nyc
    beach
    indian
    petite
    stop
    mad
    plann
    guys
    response
    man
    short
    maybe
    explicit
    childhood
    wink
    maintain
    allow
    keeps
    tall
    order
    talk
    cute
    mexican
    help
    over
    years
    playful
    held
    including
    cold
    horny
    before
    perfect
    superior
    roomates
    fit
    somewhere
    interesting
    appetizers
    actually
    late
    willing
    jucy
    wouldn
    them
    somebody
    food
    jerking
    safe
    snapchat
    they
    hands
    now
    discuss
    term
    gets
    grammar
    name
    always
    goofy
    each
    went
    friendship
    side
    mean
    weight
    doing
    house
    hard
    yeah
    used
    connect
    year
    our
    whit
    bay
    really
    living
    open
    motel
    since
    seriously
    mate
    got
    lmt
    free
    quite
    ect
    california
    puy
    put
    hand
    care
    visit
    could
    keep
    american
    place
    isn
    deja
    think
    south
    tennis
    platonic
    feel
    encouraging
    one
    feet
    rant
    done
    lol
    owner
    impossible
    thick
    miss
    little
    toy
    start
    twenty
    weekdays
    their
    master
    too
    white
    listen
    gripping
    hug
    relationship
    serve
    part
    believe
    king
    kind
    anxiety
    unfortunately
    future
    silly
    were
    lbs
    mine
    comfortable
    tonight
    say
    hangout
    pay
    need
    any
    lie
    offering
    person
    self
    cope
    able
    lik
    also
    take
    proble
    online
    wanting
    latino
    play
    experienced
    opposite
    park
    most
    eight
    plan
    superhero
    extremely
    why
    don
    eachother
    drive
    clean
    businessman
    fact
    shop
    golf
    random
    effo
    find
    busy
    grabbed
    spite
    title
    dick
    unhappy
    crime
    athletic
    only
    black
    pretty
    employed
    local
    hope
    hip
    plate
    his
    hiv
    means
    hopefully
    watch
    expectation
    truly
    listener
    despite
    during
    having
    footrest
    beac
    girl
    course
    married
    patch
    bad
    stuff
    she
    through
    grab
    respond
    set
    sex
    see
    prosper
    close
    yes
    subject
    stats
    movie
    exchange
    please
    worship
    between
    probably
    encounters
    email
    never
    jobs
    responses
    missing
    zombie_poppet
    screen
    attention
    key
    come
    problem
    limits
    many
    adventures
    drug
    etc
    games
    alot
    focused
    whole
    bell
    color
    motivation
    whatever
    100mg
    simply
    laugh
    100
    better
    basis
    west
    due
    been
    much
    far
    bbw
    proposal
    direction
    educated
    shopping
    gay
    offered
    wet
    new
    understand
    muscular
    smoking
    those
    honest
    myself
    mega
    these
    straight
    value
    choices
    while
    ongoing
    joke
    situation
    property
    pack
    seven
    hoping
    coffee
    respected
    someone
    128529
    ready
    grown
    funny
    things
    make
    same
    trip
    party
    several
    week
    practice
    hang
    holidays
    kik
    cleaning
    student
    frequently
    warming
    dark
    off
    vacant
    thought
    hispanic
    someo
    english
    spend
    tha
    meet
    breastfeeding
    summer
    kiss
    being
    descreete
    shape
    skill
    yet
    generous
    seems
    other
    attractive
    easy
    citizen
    has
    smart
    hater
    real
    around
    read
    big
    couple
    laughed
    possible
    early
    discreet
    five
    apt
    listening
    bit
    lady
    desire
    helpful
    either
    passionate
    two
    because
    people
    absolutely
    some
    back
    library
    understood
    curious
    165
    happens
    scale
    though
    creative
    connecting
    subjet
    three
    loocking
    business
    asap
    host
    post
    stage
    about
    working
    wouldnt
    getting
    vaginas
    favorite
    trustworthy
    dinner
    plus
    afternoon
    act
    wanna
    own
    techie
    pussy
    into
    articulate
    negative
    right
    sports
    subj
    area
    vallej
    housing
    there
    hey
    long
    experices
    patna
    way
    forward
    bored
    was
    buy
    inches
    becoming
    gain
    mike_kent18
    line
    trying
    romantic
    places
    morning
    attached
    tell
    mature
    embed
    clear
    trim
    pic
    doesn
    disease
    single
    lived
    chat
    teased
    flock
    shit
    fill
    again
    spell
    relaxing
    when
    lately
    tight
    interested
    details
    evenings
    out
    manhattan
    nice
    draw
    eyes
    flirtatiou
    bundle
    professional
    friends
    younger
    deranged_lizzie_borden187
    rich
    age
    together
    dudedette
    curvy
    push
    hello
    minister
    
