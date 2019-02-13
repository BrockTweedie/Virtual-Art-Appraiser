"""
Crawls through Sotheby's auctions of interest (identified manually via their search form, see 
base_urls.py) and stores their full html in Cache/

A separate script will then pass over these cached files and download the images, and another script
will parse the html.
"""


from   requests import get
from   bs4 import BeautifulSoup
import re
import os.path
from   time import time, sleep
import random
from   base_urls import base_urls


"""
TO DO
* More systematic painting-finding? (Orientalism, Paris auctions, additional auction titles) Or 
  just go ArtNet/ArtPrice
* The 2005 19th Century auction was a good exmaple of one where we'd benefit from a better database.
  The painting titles are *not* included, and images are severly shrunk and given huge white 
  borders.
"""


"""
19 June 200712:30 PM CEST Paris    Old Master and 19th Century Drawings and Paintings
             '2007/old-master-and-19th-century-drawings-and-paintings-pf7008',
can't find nationalities
dimensions are  "HH x WW" instead of "HH by WW", and not found by current regex

26 January 200810:00 AM EST New York       Old Master and 19th Century European Art  
can't find nationalities

25 June 20085:30 PM CEST Paris    19th Century Paintings and Drawings
super-weird nationalities

31 January 200910:00 AM EST New York    Old Master and 19th Century European Art
can't find nationalities
"""


scraping_time_delay = 5  # Sotheby's website requests 15s delay
scraping_time_variation = 1  # optionally, add a little randomness
scraping_time_min = 5  # optionally, add a floor

cache_dir = '/home/brock/Sothebys/Cache'

skip_paris = False   # Paris online record-keeping is garbage...skipping for now until I can figure out a dedicated algorithm to unpack them!


headers={
"User-Agent":"Mozilla/5.0"
#"User-Agent":"Chrome/67.0.3396.99"
#'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
}

# Sometimes Sotheby's lot numbers will skip. Allow the scraper to handle this by looking ahead n lots
#   until it gives up.
n_skip_max = 10

Sothebys_lot_url = 'http://www.sothebys.com/en/auctions/ecatalogue/'

# for testing
#base_urls = ['2003/old-master-and-19th-century-paintings-and-drawings-pf3007']
#base_urls = ['2017/19th-century-european-paintings-l17101']
#base_urls = ['2016/19th-century-european-art-n09499']
#base_urls = ['2009/19th-century-european-paintings-including-spanish-painting-the-orientalist-sale-and-german-austrian-scandinavian-and-symbolist-works-l09663']
#base_urls = ['2009/19th-century-european-paintings-and-modern-contemporary-art-am1086']
#base_urls = ['2006/old-master-and-19th-century-european-art-n08166']
#base_urls = ['2005/19th-century-european-paintings-includingbrgerman-austrian-hungarian-slavic-paintingsbrthe-orientalist-sale-andbrthe-scandinavian-sale-l05101']
#base_urls = ['2006/19th-century-european-paintings-am0995']

#base_urls = ['2007/19th-century-paintings-including-spanish-painting-and-symbolism-the-poetic-vision-l07103']

#base_urls = base_urls[-5:]

# Start scraping!
t_start = time()
start_url = base_urls[0]  # in case need to restart, can choose a different starting point
#start_url = '2003/19th-century-paintings-n07886'
begin = False
for base_url in base_urls:

    # Scan through the list until we hit the desired starting point
    if base_url == start_url:
        begin = True
    if not begin:
        continue

    print()
    print("---------------------------------------------")
    print(base_url)
    print("---------------------------------------------")
    print()

    auction_dir = cache_dir + '/' + re.sub(r'/', '--', base_url)
    done = False
    lot_number = 0
    lot_index = -1  # in case we are blessed with an explicit list, e.g. [1, 2, 4, 6, 7, 8, ...]
    n_skipped = 0

    # First, check if there is a master page for the auction, from which we can just
    #   get a list of the lot #s (rather than having to search page-by-page to check
    #   whether they exist!), and then walk through them by list index.
    master_url = 'http://www.sothebys.com/en/auctions/' + base_url + \
                 '.html#&page=all&sort=lotSortNum-asc&viewMode=list'
    response = get(master_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    lot_number_strings = re.findall(r"'id':'(\d+)'", soup.text)
    lot_number_list = list(map(int, filter(str.isdigit, lot_number_strings)))

    # Otherwise, try to find a valid lot, and extract the full auction list from its html
    lots_to_try = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    100, 101, 200, 201, 300, 301, 400, 401, 500, 501, 600, 601,
                    700, 701, 150, 151, 250, 251, 350, 351, 450, 451, 550, 551, 650, 651,
                    750, 751, 800, 801, 850, 851, 900, 901, 950, 951 ]
    if not lot_number_list:
        print(" -- NO MASTER LIST...LOOKING FOR A VALID LOT TO GRAB ONTO")
        good_url = ''
        # first check if there's anything already in the cache, and use the first lot # we find
        if os.path.exists(auction_dir):
            for name in os.listdir(auction_dir):
                nums = re.findall(r'lot\.(\d+)\.html', name)
                if len(nums) == 1 and nums[0].isdigit():
                    good_url = Sothebys_lot_url + base_url + '/lot.' + nums[0] + '.html'
                    break
        # still cannot find? then ping the web for lots in lots_to_try
        if not good_url:
            for lot_try in lots_to_try:
                print(" -- CHECK FOR LOT", lot_try)
                url = Sothebys_lot_url + base_url + '/lot.' + str(lot_try) + '.html'
                print(url)
                response = get(url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                lot_header_containers = soup.find_all('div', class_ = 'lotdetail-header-block')
                if lot_header_containers:
                    print(" -- GOT ONE!!! PROCEED TO SCRAPE")
                    good_url = url
                    break
        # STILL cannot find? then print an error message and skip this auction
        if not good_url:
            print(" -- CANNOT FIND ANY LOTS FOR THIS AUCTION THROUGH CURSORY SEARCH....SKIPPING!")
            continue
        print(good_url)
        response = get(good_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        lot_numbers_string = re.findall(r"var\s+lotIds\s+=\s+\[(.+)\s+]", soup.text)[0]
        lot_number_strings = lot_numbers_string.strip().split(',')
        lot_number_strings = list(map(lambda x: x.strip("'"), lot_number_strings))
        lot_number_list = list(map(int, filter(str.isdigit, lot_number_strings)))

    print(lot_number_list)
    
    while not done:

        # ** This is now a bit deprecated, since we search above for an explicit list of valid lots
        #    However, I keep it in case I need to revert back to the "walking" strategy for some reason
        if lot_number_list:  # walk through explicit list
            lot_index += 1
            if lot_index < len(lot_number_list):
                lot_number = lot_number_list[lot_index]
            else:
                break
        else:  # or just keep trying to increment the lot number and see if page exists
            lot_number += 1

        url = Sothebys_lot_url + base_url + '/lot.' + str(lot_number) + '.html'
        cache_file_name = auction_dir + '/lot.' + str(lot_number) + '.html'
        print(url)
        print(cache_file_name)
        if os.path.exists(cache_file_name):
            print(" -- FILE EXISTS, SKIPPING -- ")
            continue
        response = get(url, headers=headers)
        print(response)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Introduce a random time lag to comply with robots.txt and to reduce chance of being blocked!
        delay =  abs(random.uniform(scraping_time_delay - scraping_time_variation,
                                    scraping_time_delay + scraping_time_variation))
        delay = max(delay, scraping_time_min)
        print("...sleep", delay)
        sleep(delay)

        # Header block (basic summary info at top of catalogue entry page, skip item if header is nonexistant!)
        lot_header_containers = soup.find_all('div', class_ = 'lotdetail-header-block')
        if not lot_header_containers:
            print("  NO INFORMATION HERE, SKIPPING")
            n_skipped += 1
            if n_skipped == n_skip_max:
                done = True
            continue
        n_skipped = 0

        # Cache html to file for later parsing
        if not os.path.exists(auction_dir):
            os.makedirs(auction_dir)
        cache_file = open(cache_file_name, 'w')
        cache_file.write(response.text)
        cache_file.close()
        
        #if lot_number > 5:  break


print()
print("Done scraping! Time elapsed =", time() - t_start)


