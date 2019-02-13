"""
Extract painting information from our database of cached Sotheby's lot websites and store as pickled
dataframe for downstream cleaning/analysis.

Note that many assumptions go into the scraping, and this is very much geared toward extracting 19th
Century (or a little before/after) oil paintings. Do be aware!! E.g., there may be an explicit 
constraint on painting dates below, especially to avoid regex screwups propagating to our fits 
later.

Only light attempt is made at this first pass to exclude non-desired objects like older/more 
contemporary works, sculptures, etc. 
"""


########################
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


output_file_code = 'sb_data_raw'
allow_overwrite = False
cache_dir = '/home/brock/Sothebys/Cache'
data_dir = '/home/brock/Sothebys/DataFrames'


# Some cities have terrible record-keeping, and might be better to just skip them
# Paris: inconsistent artist+title+info in headers...difficult to consistently parse
# Zurich: artist+title conjoined, no dimensions or dates *or* reports
skip_paris = True
skip_zurich = True

# Some auctions just happened to have bad records. Skip these, too.
skip_auctions = ['2000--19th-century-european-paintings-am0762']
skip_auctions = [ cache_dir + '/' + auc for auc in skip_auctions ]


# Explicit list of "known" (to me) nationalities, for digging out nationality info,
#   which isn't always presented in a consistent way!
list_of_nationalities = \
    ['American', 'Australian', 'Austrian', 'British', 'Belgian', 'Canadian', 'Czech',
     'Danish', 'Dutch', 'Finnish', 'French', 'German', 'Greek', 'Hungarian', 'Italian',
     'Mexican', 'Norwegian', 'Japanese', 'Polish', 'Swedish', 'Peruvian', 'Spanish', 'Turkish']


# Initialize the dataframe
columns = ['Lot', 'Artist', 'Nationality', 'Born', 'Died',
           'Title', 'Est Low', 'Est High', 'Sale Price', 'Currency',
           'Height', 'Width', 'Date', 'Signed', 'Catalogue Note',
           'Auction Date', 'Auction Location', 'Auction Title', 'Auction Code',
           'Local Image Path',]
df = pd.DataFrame(columns=columns)


# Start scraping!
t_start = time()
auction_dirs = [ os.path.join(cache_dir, subdir) for subdir in os.listdir(cache_dir) 
                 if os.path.isdir(os.path.join(cache_dir, subdir)) ]
for it, auction_dir in enumerate(auction_dirs):
    print(it, "  ", auction_dir)
start_dir = auction_dirs[0]  # in case need to restart, can choose a different starting point
begin = False
for auction_dir in auction_dirs:

    # Scan through the list until we hit the desired starting point
    if auction_dir == start_dir:
        begin = True
    if not begin:
        continue

    print()
    print("---------------------------------------------")
    print(auction_dir)
    print("---------------------------------------------")
    print()

    # Check for special-case auctions
    auction_code = 'NA'
    auction_code_candidates = re.findall(r'-[a-z]+\d+', auction_dir)
    if auction_code_candidates:
        auction_code = auction_code_candidates[-1].replace('-', '')
    is_paris = ('pf' in auction_code)
    is_zurich = ('zh' in auction_code)
    if skip_paris and is_paris or \
       skip_zurich and is_zurich or \
       auction_dir in skip_auctions:
        print(" -- SKIPPING THIS AUCTION (FLAGGED FOR BAD RECORD-KEEPING)")
        continue
    
    if not os.path.exists(auction_dir):
        print(" -- NONEXISTANT CACHE DIRECTORY. SKIPPING!")
        continue

    done_printing_auction_details = False
    
    for file_name in os.listdir(auction_dir):
        if file_name.endswith(".html"):
            cache_file_name = os.path.join(auction_dir, file_name)
        else:
            continue

        print(cache_file_name)
        soup = BeautifulSoup(open(cache_file_name), "html.parser")

        # For possible later use, keep a record of the corresponding image file name on the local system
        image_file_name = re.sub('.html', '.jpg', cache_file_name)
        if not os.path.exists(image_file_name):
            image_file_name = re.sub('.jpg', '.jpg.txt', image_file_name)
        
        # Header block (basic summary info at top of catalogue entry page, skip item if nonexistant!)
        lot_header_containers = soup.find_all('div', class_ = 'lotdetail-header-block')
        if not lot_header_containers:
            print(" -- COULDN'T FIND HEADER INFORMATION, SKIPPING")
            continue
        lot_header = lot_header_containers[0]

        # Lot number
        lot_number = int(re.findall(r'\.(\d+)\.html', cache_file_name)[0])
        
        # Auction info (stored redundantly for each item)
        timecode = 'NA'
        time_container = soup.find('time', class_ = 'dtstart')
        if time_container and time_container.has_attr('datetime'):
            timecode = time_container['datetime']
        location = ''
        location_container = soup.find('div', class_ = 'location')
        if location_container:
            location = location_container.text
        auction_title = ''
        auction_info_container = soup.find('h5', class_ = 'alt')
        if auction_info_container:
            auction_title = auction_info_container.text      
            
        # Artist info, first attempt (there are some secondary tries later due to differing conventions for earlier/later auctions)
        artist = 'NA'
        nationality = 'NA'
        artist_born = nan
        artist_died = nan
        artist_candidate = soup.find('div', class_ = 'lotdetail-guarantee')
        if artist_candidate:
            for it, line_raw in enumerate(artist_candidate):
                line = str(line_raw)
                if it == 0:
                    artist = line
                    continue
                # check for nationality against a list of known values
                # it is listed in the same block as the artist name in older auctions
                for possible_nationality in list_of_nationalities:
                    if possible_nationality.capitalize() in line.capitalize():
                        nationality = possible_nationality
                # check for artist lifespan...also listed in the artist name block in older auctions
                lifespan_candidate = re.findall(r'\d\d\d\d', line)
                if len(lifespan_candidate) >= 1:
                    artist_born = int(lifespan_candidate[0])
                if len(lifespan_candidate) == 2:
                    artist_died = int(lifespan_candidate[1])

        # Special cleanup for Paris auctions
        if is_paris and artist != 'NA':
            #name_and_details = re.findall(r'([\w\s]+)\s*\((.*)\)', artist)
            name_and_details = re.findall(r'(.*)\((.*)\)', artist)
            print(name_and_details)
        
        # Title, and possibly nationality if stored in the lot's "subtitle" block
        title = 'NA'
        lotdetail_subtitle_block = lot_header.find('div', class_ = 'lotdetail-subtitle')
        if (lotdetail_subtitle_block):
            for lotdetail_subtitle_entry in lotdetail_subtitle_block:
                if str(type(lotdetail_subtitle_entry)) == "<class 'bs4.element.Tag'>":
                    # skip <br/> tags
                    continue
                # for shorter headers, assume it's only listing title (no nationality...I've seen examples of this)
                if nationality == 'NA' and len(lotdetail_subtitle_block) > 2:  
                    nationality = str(lotdetail_subtitle_entry).capitalize()
                else:
                    title = capwords(str(lotdetail_subtitle_entry))

        # Check again for artist lifespan, for lots with an explicitly-denoted entry for them
        lifespan_container = soup.find('div', class_ = 'lotdetail-artist-dates')
        if lifespan_container:
            lifespan_candidate = re.findall(r'\d\d\d\d', lifespan_container.text)
            if len(lifespan_candidate) >= 1:
                artist_born = int(lifespan_candidate[0])
            if len(lifespan_candidate) == 2:
                artist_died = int(lifespan_candidate[1])
 
        # Appraisal estimated values
        low_estimate = nan
        high_estimate = nan
        low_block = lot_header.find('span', class_ = 'range-from')
        high_block = lot_header.find('span', class_ = 'range-to')
        if (low_block):
            # no idea why I can't set low_estimate directly, but it doesn't work!
            moop = int(low_block.text.replace(',', ''))
            low_estimate = moop
        if (high_block):
            moop = int( high_block.text.replace(',', '') )
            high_estimate = moop

        # Actual sale price in this auction
        sale_price = nan
        sold_container = soup.find('div', class_ = 'price-sold')
        if (sold_container):
            for line in sold_container:
                integers_from_line = re.findall(r'\d+', str(line))
                if len(integers_from_line) == 1:
                    sale_price = int(integers_from_line[0])

        # Currency used (will correct these all back to common currency, accounting for inflation, in downstream cleaning script)
        currency = "NA"
        currency_container = soup.find('div', class_ = 'dropdown currency-dropdown inline')
        if currency_container and currency_container.has_attr('data-default-currency'):
            currency = currency_container['data-default-currency']

        # Dimensions (from the description block)
        height = nan
        width = nan
        details_container = soup.find('div', class_ = 'lotdetail-description-text')
        details_lines = details_container.get_text(separator='\n').split('\n')
        for line in details_lines:
            line = line.replace('by by', 'by')   # yes, sometimes replace "by by" by "by"...sigh
            line = line.replace('¼','.25')  # ugh, unicode fractions!
            line = line.replace('½','.5')
            line = line.replace('¾','.75')
            dimensions_string_candidate = re.findall(r'\d+[\.\,]?\d*\s+by\s+\d+[\.\,]?\d*\s*cm|'
                                                      '\d+[\.\,]?\d*\s+by\s+\d+[\.\,]?\d*\s*,|'
                                                      'cm\.?\s+\d+[\.\,]?\d*\s+x\s+\d+[\.\,]?\d*', line)
            dimensions_string_candidate = [ candidate.replace(',', '.') for candidate in dimensions_string_candidate]  # fix euro-decimal
            
            if dimensions_string_candidate:
                dimensions = re.findall(r'\d+\.?\d*', dimensions_string_candidate[0])
                if len(dimensions) >= 1:
                    height = float(dimensions[0])
                if len(dimensions) >= 2:
                    width = float(dimensions[1])
        if np.isnan(height) or np.isnan(width):
                print(" -- MISSING DIMENSIONS?")
                    
        # Check the material
        # For now, we will only consider oil paintings!
        is_oil = False
        is_gouache = False
        is_watercolor = False
        # pastel, pencil, "color"/"colour" ....
        for line in details_lines:
            if re.findall(r'\boil\b', line):
                is_oil = True
            if re.findall(r'\bolio\b', line):
                # Italiano!
                is_oil = True
            if re.findall(r'\bgouache\b', line):
                is_gouache = True
            if re.findall(r'\bwatercolor\b', line):
                is_watercolor = True
        if not is_oil:
            print(" -- THIS ISN'T AN OIL PAINTING, SKIPPING")
            continue

        # Check if signed
        signed_words = ['signed', 'signature', 'siglato', 'firmato']
        is_signed = any([ any([ signed_word in line.lower() for signed_word in signed_words ]) for line in details_lines ])
        
        # Try to get the date from the description block, when explicitly signed (sometimes just two numbers)
        # ** Not immune to (rare) cases where it has some kind of "item number" stamp or writing
        date = nan
        for line in details_lines:
            # digit or whitespace in front of "cm" or "in", or separeted by "by" or "x"
            # (originally tried blanket search for "cm" or "in", but found weird cases where these character
            #  sequences are embedded in words near the date!)
            has_dimensions = re.findall(r'[\d\s]cm|[\d\s]in|cm\.?[\d\s]|in\.?[\d\s]|'
                                        '\d+\s+by\s+\d+|\d+\s+x\s+\d+', line)  
            if has_dimensions:
                continue
            print("    ", line)
            # If the line contains words like 'dated', we will not continue looking into later lines
            #   set flag accordingly
            date_words = ['dated', 'date', 'datato', 'signed', 'inscribed']
            this_is_the_date_line = any([ date_word in line.lower() for date_word in date_words ])
            date_string_candidates = re.findall(r'\d\d+\(|\d\d+', line)
            for date_string_candidate in date_string_candidates:
                print("      --> ", date_string_candidate)
                fail = False
                if "(" in date_string_candidate:
                    # includes 190(?) -> 1900, 19(?) -> 190
                    date_string_candidate = date_string_candidate.replace('(', '0')
                if len(date_string_candidate) == 3:
                    # correction for Italian-style 1854 -> 854 notation...blah!
                    # (will be checked later for sanity)
                    date_string_candidate = '1' + date_string_candidate
                if len(date_string_candidate) % 2 == 1 or \
                   len(date_string_candidate) > 4:
                    # disallow things like 19(?) -> 190...better ways to estimate!
                    fail = True
                tentative_date = int(date_string_candidate)
                if tentative_date < 100:
                    # for 2-digit dates, try to rebuild the century using the artist's lifespan info
                    # (this is not very sophisticated...could be written to make sure it wasn't apparently
                    #  painted when the artist was 5 years old or whatever, but they would need to be very
                    #  long-lived for any ambiguity!)
                    if not np.isnan(artist_born) and not np.isnan(artist_died):
                        for i in range(1,21):  # check 2nd thru 21st centuries
                            if artist_born < i*100 + tentative_date <= artist_died:
                                tentative_date = i*100 + tentative_date
                                break
                    elif not np.isnan(artist_born) and tentative_date + (artist_born//100)*100 > artist_born:
                        tentative_date += (artist_born//100)*100
                    elif not np.isnan(artist_died) and tentative_date + (artist_died//100)*100 <= artist_died:
                        tentative_date += (artist_died//100)*100
                    else:
                        fail = True
                # Idiot-check against artist lifespan
                if not np.isnan(artist_born) and tentative_date <= artist_born or \
                   not np.isnan(artist_died) and tentative_date >  artist_died:
                    fail = True
                # Idiot check against global scope of the scraping dates!
                # (Would especially screw up 
                if not (1700 < tentative_date < 2000):
                    fail = True
                if fail:
                    if this_is_the_date_line:
                        # this line contained a word like 'dated' and we didn't find a good date...bail out!
                        break
                    else:
                        # perhaps this was just the wrong line...keep looking
                        continue
                # okay...let's guess this is the actual date that the appraiser wrote down
                date = tentative_date
                
                
        # The catalogue note in full (useful later for finding dates, keywords, gauging appraiser's interest level, etc)
        catalogue_note = ''
        catalogue_note_outer_container = soup.find('div', class_ = 'lotdetail-catalogue-notes-holder')
        catalogue_note_inner_container = []
        if catalogue_note_outer_container:
            catalogue_note_inner_container = catalogue_note_outer_container.find('div', class_ = 'readmore-content')
        if catalogue_note_inner_container:
            # *replaces non-breaking spaces (unicode analog of LaTeX ~)
            catalogue_note = catalogue_note_inner_container.text.replace(u'\xa0', u' ')

        
        ##############
        # Other things to look out for:
        # * check if signed
        # * lists of exhibitions and entries in other catalogues (can sometimes infer date from first exhibition)
        ##############
        
        # Printout
        #if not done_printing_auction_details:
        #    print()
        #    print(timecode, location, "  ", auction_title, "  ", auction_code)
        #    print("--------------------------------")
        #    done_printing_auction_details = True
        print(lot_number, artist, nationality, artist_born, artist_died, "\""+title+"\"", low_estimate, high_estimate, sale_price, currency, height, width, date, is_signed, sep="  ")

        # Accumulate into data frame
        df_this = pd.DataFrame(
            [[lot_number, artist, nationality, artist_born, artist_died, title,
              low_estimate, high_estimate, sale_price, currency, height, width, date, is_signed,
              catalogue_note,
              timecode, location, auction_title, auction_code, image_file_name]],
            columns=columns )
        df = df.append(df_this)


print()
print("Done scraping! Time elapsed =", time() - t_start)

# Write the dataframe to file (pickle)
df = df.apply(pd.to_numeric, errors='ignore')
df = df.reset_index(drop=True)
wrote_file = False
while not wrote_file:
    df_file_name = data_dir + '/' + output_file_code + '.pkl'
    if not os.path.isfile(df_file_name) or allow_overwrite:
        df.to_pickle(df_file_name)
        wrote_file = True
    else:
        output_file_code += '_'


