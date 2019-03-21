import urllib
from bs4 import BeautifulSoup

url = 'http://arxiv.org/list/cs.{}/{}{}?show=1000'
fields = ['CV']
keywords = ["deep", "learn", "convolution", "recurrent", "neural", "network"]
months = ['{:0>2d}'.format(i+1) for i in range(12)]
years = ['{:0>2d}'.format(i) for i in range(15, 19)]

f = open("paperlinks.txt", "wt")

for field in fields:
    for year in years:
        for month in months:
            query_url = url.format(field, year, month)
            print('Retrieving {}'.format(query_url))
            uh = urllib.request.urlopen(query_url)
            data = uh.read()
            soup = BeautifulSoup(str(data), features="html.parser")
            titles = soup.findAll('div', {'class': 'list-title'})
            authors = soup.findAll('div', {'class': 'list-authors'})
            paper_urls = soup.findAll('span', {'class': 'list-identifier'})
            if len(titles) != len(authors):
                print(str(len(titles)) + " != " + str(len(titles)))
                print('number of titles and authors mismatch')
            else:
                for title, author, paper_url in zip(titles, authors, paper_urls):
                    title = title.contents[-1].strip()
                    paper_url = 'http://arxiv.org' + paper_url.contents[0].attrs['href']
                    paper_authors = [au.string.strip() for au in author.findAll('a')]
                    low_title = title.lower()
                    if any(k in low_title for k in keywords):
                        f.write(title + "\n")
                        f.write(paper_url + "\n")

f.close()