"""
Extract painting information from our database of cached Artnet websites/images and store as pickled dataframe for downstream cleaning/analysis.

This is based on 100-per-page summary search results manually copied from Artnet, supplemented by revisiting/modifying the image file links for each item.

** Information like artist lifespan/nationality may also be missing from the Artnet summaries. I just provide this here by hand. (Can make more sophisticated if I ever build an automated scraper, e.g., for Artprice database.)

Sorry, the file naming format here was pretty crappy...I rushed this out in one Sunday evening while I had connection to the Clark's network!
"""

#######################
from   bs4 import BeautifulSoup
import re
import numpy as np
from   numpy import nan
import pandas as pd
from   string import capwords
import os.path
import os
from   time import time, sleep
from   random import randint
########################


output_file_code = 'an_data_raw'
allow_overwrite = True
cache_dir = '/home/brock/Sothebys/Cache/Artnet'
data_dir = '/home/brock/Sothebys/DataFrames'


## The "codenames" and Sotheby's-like bio info of the artists: the former for file and directory
##   names, the latter for proper Sotheby's integration
## Since we deal with a small number of Artnet artists for now, it's sufficient to do this
##   manually!
artist_info = {
    'Corot':{'Artist':'JEAN-BAPTISTE-CAMILLE COROT', 'Nationality':'French',
             'Born':1796., 'Died':1875.},
    'Sorolla':{'Artist':'JOAQUÍN SOROLLA', 'Nationality':'Spanish',
               'Born':1863., 'Died':1923.},
    'Gerome':{'Artist':'JEAN-LÉON GÉRÔME', 'Nationality':'French',
              'Born':1824., 'Died':1904.},
    'Bouguereau':{'Artist':'WILLIAM BOUGUEREAU', 'Nationality':'French',
                  'Born':1825., 'Died':1905.},
    'Breton':{'Artist':'JULES BRETON', 'Nationality':'French',
              'Born':1827., 'Died':1906.},
}


# Initialize the dataframe
columns = ['Lot', 'Artist', 'Nationality', 'Born', 'Died',
           'Title', 'Est Low', 'Est High', 'Sale Price', 'Currency',
           'Height', 'Width', 'Date', 'Signed', 'Catalogue Note',
           'Auction Date', 'Auction Location', 'Auction Title', 'Auction Code',
           'Local Image Path']
df = pd.DataFrame(columns=columns)


# Start scraping!
t_start = time()
artist_dirs = [ os.path.join(cache_dir, subdir) for subdir in os.listdir(cache_dir) 
                 if os.path.isdir(os.path.join(cache_dir, subdir)) ]
for it, artist_dir in enumerate(artist_dirs):
    print(it, "  ", artist_dir)
start_dir = artist_dirs[0]  # in case need to restart, can choose a different starting point
begin = False
for artist_dir in artist_dirs:

    # Special/temp exception
    if 'BADFILE' in artist_dir:
        continue
    
    # Scan through the list until we hit the desired starting point
    if artist_dir == start_dir:
        begin = True
    if not begin:
        continue

    # *Using explicit list of artists that I scraped from ArtNet
    artist = ''
    nationality = ''
    born = nan
    died = nan
    for artist_name, info in artist_info.items():
        if artist_name in artist_dir:
            artist = info['Artist']
            nationality = info['Nationality']
            born = info['Born']
            died = info['Died']

    print()
    print("---------------------------------------------")
    print(artist_dir)
    print(artist, nationality, born, died)
    print("---------------------------------------------")
    print()

    if not os.path.exists(artist_dir):
        print(" -- NONEXISTANT CACHE DIRECTORY. SKIPPING!")
        continue
            
    for file_name in os.listdir(artist_dir):
        
        # At an intermediate stage, I had copied the website html from a browser view of the html
        #   code, and the html of *that* turned out to be a bunch of crazy code where, e.g., each
        #   tag had to be wrapped in html to format it for browser display. I fixed that by copying
        #   the browser display of the code into emacs. However, since these "wrapped" files are
        #   my originals, I keep them around.
        # (This was a useful mistake...the "wrapped" files contain the links into ArtNet's
        #   online image database!)
        if file_name.endswith(".html") and 'wrapped' not in file_name:
            cache_file_name = os.path.join(artist_dir, file_name)
        else:
            continue

        print(cache_file_name)
        soup = BeautifulSoup(open(cache_file_name), "html.parser")
        artwork_containers = soup.find_all('div', class_='artworkItem')
        print(f"   {len(artwork_containers)} paintings")

        for artwork_container in artwork_containers:

            text = artwork_container.text
            #print(text)
            
            title = re.findall('Title(.*)',text)[0]

            date = nan
            date_blocks = re.findall(r'\nYear\sof\sWork(.*)\n', text)
            if date_blocks:
                block = date_blocks[0]
                single_date_re = re.findall('\d\d\d\d', block)
                double_date_re = re.findall('(\d\d\d\d)-(\d\d\d\d)', block)
                if double_date_re:
                    date = (int(double_date_re[0][0]) + int(double_date_re[0][1])) / 2
                elif single_date_re:
                    date = int(single_date_re[0])

            is_signed = ('\nMisc.Signed' in text)
            
            # Check the material
            # For now, we will only consider oil paintings!
            medium_blocks = re.findall(r'\nMedium(.*)\n', text)
            medium = None
            if medium_blocks and 'oil' in medium_blocks[0].lower():
                medium = 'oil'
            if medium != 'oil':
                continue
            
            USD_conversion_blocks = re.findall('\((.*?USD)\)', text)
            
            estimate_blocks = re.findall(r'\nEstimate(.*)', text)
            low_estimate = nan
            high_estimate = nan
            currency = 'NA'
            if estimate_blocks:
                block = estimate_blocks[0] if not USD_conversion_blocks else \
                        USD_conversion_blocks[0]
                low_estimate_re = re.findall('^(.*?)\s', block)
                high_estimate_re = re.findall('-\s*(.*)\s', block)
                currency_re = re.findall('\s([A-Z]+)$', block)
                if currency_re:
                    currency = currency_re[0]
                    if low_estimate_re and low_estimate_re[0]:
                        low_estimate = int(low_estimate_re[0].replace(',',''))
                        high_estimate = int(high_estimate_re[0].replace(',','')) \
                                        if high_estimate_re else low_estimate
                    elif high_estimate_re and high_estimate_re[0]:
                        high_estimate = int(high_estimate_re[0].replace(',',''))
                        low_estimate = high_estimate

            # *This ignores the possible presence of a "Premium" qualifier, which I think indicates
            #    that a buyer's premium is included in the price (but does not give the amount of
            #    the premium)
            sold_blocks = re.findall(r'\nSold For([0-9,]+\s*[A-Z]+)', text)
            sale_price = nan
            if sold_blocks:
                block = sold_blocks[0] if not len(USD_conversion_blocks) > 1 else \
                        USD_conversion_blocks[1]
                sale_price_re = re.findall('^(.*?)\s', block)
                if sale_price_re:
                    sale_price = int(sale_price_re[0].replace(',',''))

            height = nan
            width = nan
            height_re = re.findall('Height\s*(\d+\.?\d*)\s*cm', text)
            width_re = re.findall('Width\s*(\d+\.?\d*)\s*cm', text)
            if height_re and width_re:
                height = float(height_re[0])
                width = float(width_re[0])

            auction_info_blocks = re.findall(r'\nSale\sof(.*)', text)
            location = ''
            timecode_raw = ''
            auction_title = ''
            lot_number = nan
            auction_code = 'NA'  # specialized to Sotheby's...ignore
            if auction_info_blocks:
                block = auction_info_blocks[0]
                location_re = re.findall('^(.*):', block)
                timecode_re = re.findall(':\s*(.*)\s*\[', block)
                auction_title_re = re.findall('\]\s*(.*)$', block)
                lot_number_re = re.findall('\[Lot\s*(\d+)\s*\]', block)
                if location_re:
                    location = location_re[0]
                if timecode_re:
                    timecode_raw = timecode_re[0]
                if auction_title_re:
                    auction_title = auction_title_re[0]
                if lot_number_re:
                    lot_number = int(lot_number_re[0])

            # Adjust to Sotheby's-like timecodes, with fake hour/minute/timezone
            timecode = ''
            if timecode_raw:
                day_re = re.findall('\s(\d+),', timecode_raw)
                month_re = re.findall(',\s*([A-Z][a-z]+)\s', timecode_raw)
                year_re = re.findall('\d\d\d\d', timecode_raw)
                if day_re and month_re and year_re:
                    timecode = day_re[0] + ' ' + month_re[0] + ' ' + year_re[0] + \
                               ' 12:00 PM EST'                

            # Since this data is pulled from a summary page, we don't have access to the individual
            #   catalogue notes
            catalogue_note = ''

            ####  COME BACK TO THIS!!!  ####
            image_file_name = ''
            
            # Accumulate into data frame
            df_this = pd.DataFrame(
                [[lot_number, artist, nationality, born, died, title,
                  low_estimate, high_estimate, sale_price, currency, height, width, date, is_signed,
                  catalogue_note,
                  timecode, location, auction_title, auction_code, image_file_name]],
                columns=columns )
            df = df.append(df_this)


print()
print("Done scraping! Time elapsed =", time() - t_start)

# Write the dataframe to file (pickle)
df = df.reset_index(drop=True)
wrote_file = False
while not wrote_file:
    df_file_name = data_dir + '/' + output_file_code + '.pkl'
    if not os.path.isfile(df_file_name) or allow_overwrite:
        df.to_pickle(df_file_name)
        wrote_file = True
    else:
        output_file_code += '_'


