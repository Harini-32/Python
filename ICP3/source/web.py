from bs4 import BeautifulSoup
import urllib.request

output_file = open("output", "w" )
def search_spider():
    url = "https://en.wikipedia.org/wiki/Deep_learning"
    source_code = urllib.request.urlopen(url)
    soup = BeautifulSoup(source_code, "html.parser")
    title1 = soup.title.string
    result = soup.findAll('a')
    print(title1)
    for a in result:
        link = a.get('href')
        output_file.writelines(str(link) + "\n")

search_spider()
