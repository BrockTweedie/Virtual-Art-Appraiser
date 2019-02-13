"""
Read "raw" Sotheby's auction data and perform some basic cleanup and preliminary analysis.


*** MAKE SURE THAT EACH ARTIST HAS CONSISTENT NATIONALITY ATTACHED

*** MAKE A FILTER FOR EMPTY IMAGES THAT ARE ACTUALLY "IMAGE UNDER ARTIST COPYWRITE, Sotheby's" 
    Google will spot these with Labels=Text and Logos=Sotheby's

* Find a way to identify "very similar" names, e.g. 'JOSÉ NAVARRO LLORÉNS', 'JOSÉ NAVARRO Y LLORENS'  <-  check by doing  print(*sorted(df.groupby('Artist').groups.keys()), sep='\n')
     (note different both in presence/absence of 'Y' and in accenting)
* Fix "names" like 'FRANCISCO MASRIERA Y MANOVENS, SPANISH 1842-1902' or 'FREDERICK HENDRIK KAEMMERER 1839-1902', where also nationality and lifespan are screwed up (should be straightforward regex)


"""

#####################
import numpy as np
import pandas as pd
from   currency_converter import CurrencyConverter
from   datetime import date
import re
#####################


df_dir = '/home/brock/Sothebys/DataFrames'
df_file_code = df_dir + '/an_data'


# Notes on inflation:
# I can also get country-specific (and aggregate) CPI information very easily from
# https://data.oecd.org/price/inflation-cpi.htm
# This can be output into a csv form, and would be *vastly* faster to run here, as well
# as more accurate. (For the 2000's, the CPI's of Europe, UK, and US are all usually well
# within 10% of each other when normalized at, say 2010. However, major differences develop
# as we scan further back in time, and should definitely be corrected for as I go deeper
# into the historical record.)
# All of this comes with some level of caveat, since the effective "CPI" for high-end consumers
# can be different than for us Normals, due to a much different composition of goods that they
# spend most of their money on. (See bookmarked Forbes article.) The tiny bit that
# I've read about this suggests that it inflates at the same level of the normal CPI, but I'd
# be happier if I had more robust information on that.
do_inflation_adjustment = True
if do_inflation_adjustment:
    import cpi   # it's a bit slow, but important to properly normalize older prices <- later build a separate table/interpolation yourself, if there is time
    print("Loaded CPI API for inflation adjustments")



# Get the raw dataframe from file
df_raw = pd.read_pickle(df_file_code + '_raw.pkl')
df = df_raw.copy()
print(f"original raw number of records: {len(df)}")


# Currency converter API
# Last updated 2018-08-01, date range 1999-01-04 thru 2018-06-12
# If this project goes on for a long time, consider setting it to auto-update from web!
# ** Also bear in mind that you need a special code to *guess* values before 1999
#    For much older auctions, consider using another tool?
converter = CurrencyConverter(fallback_on_missing_rate=True)
def convert_to_USD(amount, currency, dt):
    new_amount = 0
    if currency == 'USD':
        new_amount = amount
    else:
        try:
            new_amount = converter.convert(amount, currency, 'USD', date=dt)
        except:
            return np.nan
    if do_inflation_adjustment:
        try:
            new_amount = cpi.inflate(new_amount, dt)
        except:
            return np.nan
    return new_amount

# Get rid of redundant "index" column if it's there
df = df.drop(columns='index', errors='ignore')

# Get timezone as a separate column
def get_timezone(dt_string):
    return dt_string.split()[-1]
df['Auction Timezone'] = df['Auction Date'].map(get_timezone)

# Repair the dates/times (there is probably a more efficient/vectorized algorithm!)
def repair_dt_string(dt_string):
    substrings = dt_string.split()
    for it, ss in enumerate(substrings[:-1]):  # ignore last piece, assuming it's timezone
        ss_YYYYM_candidates = re.findall(r'\d\d\d\d\d', ss)  # fix YYYYHH:MM:SS -> YYYY HH:MM:SS
        if len(ss_YYYYM_candidates) == 1:
            ss_YYYYM = ss_YYYYM_candidates[0]
            ss_split = ss.split(ss_YYYYM)
            substrings[it] = ss_YYYYM[:-1] + ' ' + ss_YYYYM[-1] + ss_split[-1]
    new_dt_string = ' '.join(substrings[:-1])
    return pd.to_datetime(new_dt_string)
df['Auction Date'] = df['Auction Date'].map(repair_dt_string)

# Normalize currencies to USD
df['Est Low'   ] = ( df[['Est Low'   , 'Currency', 'Auction Date']]
                     .apply(lambda x: convert_to_USD(x['Est Low'   ],
                                                     x['Currency'],
                                                     x['Auction Date']), axis=1) )
df['Est High'  ] = ( df[['Est High'  , 'Currency', 'Auction Date']]
                     .apply(lambda x: convert_to_USD(x['Est High'  ],
                                                     x['Currency'],
                                                     x['Auction Date']), axis=1) )
df['Sale Price'] = ( df[['Sale Price', 'Currency', 'Auction Date']]
                     .apply(lambda x: convert_to_USD(x['Sale Price'],
                                                     x['Currency'],
                                                     x['Auction Date']), axis=1) )
df = df.drop(columns='Currency', errors='ignore')

# Make a column that describes how the date was determined
df['Dating Method'] = df['Date'].map(lambda x: 'signature' if pd.notnull(x) else 'unknown')

# Figure out date if signature is 2-digit (based on artist lifetime)
def fix_signature_date(date, born):
    if 99 < date <= 999:
        return np.nan  # 3-digits assumed invalid
    elif date > 999:
        return date  # nothing to do (we assume...)
    for century in range(10,21):
        candidate_date = 100*century + date
        if born < candidate_date:
            return candidate_date # first possible date after birth (assuming nobody is productive past age 100!)
    return np.nan
df['Date'] = df.apply(lambda x: fix_signature_date(x['Date'], x['Born']), axis=1)

# Figure out date from catalogue note if not in signature, based on simple regex search
# What does it miss? Examples:
#  *  This newly rediscovered work of 1883-1884
#  *  Painted circa 1880-84  [picks only lower value, not a big deal]
def get_catalogue_note_date(x):
    if pd.notnull(x['Date']):
        return x
    date_phrase_candidates = re.findall(r'Painted[a-zA-ZÀ-ÿ\s\,]*\d\d\d\d|circa [a-zA-Z]* ?\d\d\d\d|'
                                        'Dated [a-zA-Z]* ?\d\d\d\d', x['Catalogue Note'])
    if date_phrase_candidates:
        date_candidates = re.findall(r'\d\d\d\d', date_phrase_candidates[0])
        if date_candidates:
            x['Date'] = float(date_candidates[0])
            x['Dating Method'] = 'catalogue note'
    return x
df = df.apply(get_catalogue_note_date, axis=1)

# Impute a reasonable default date if other methods have failed.
# The "impute" dating method will be kept track of, in case we want to associate it with an
#   indicator variable later, or treat undated paintings in some other way.
def impute_date(x, artist_groups):
    if pd.notnull(x['Date']):
        return x
    median_date = artist_groups.get_group(x['Artist'])['Date'].median()  
    if pd.notnull(median_date):
        x['Date'] = median_date
        x['Dating Method'] = 'impute median'
    elif pd.notnull(x['Born']) and pd.notnull(x['Died']):
        x['Date'] = (x['Born'] + x['Died']) / 2
        x['Dating Method'] = 'impute lifespan midpoint'    
    elif pd.notnull(x['Born']):
        x['Date'] = x['Born'] + 30
        x['Dating Method'] = 'impute birth + 30'
    else:
        x['Date'] = 1850
        x['Dating Method'] = 'impute 1850'
    return x
artist_groups = df.groupby('Artist')
df = df.apply(lambda x: impute_date(x, artist_groups), axis=1)

# Get rid of (rare) artist names with asterisk (unknown attribution or semi-unknown names)
df = df[~df['Artist'].str.contains("\*")]
print(f"after dropping asterisk-artists: {len(df)}")

# Remove extra whitespaces and garbage from artist names
df['Artist'] = df['Artist'].map(lambda x: ' '.join(x.split()))

# Set the artist name and piece title to all uppercase
df['Artist'] = df['Artist'].map(lambda x: x.upper())
df['Title'] = df['Title'].map(lambda x: x.upper())

# Drop rows with weirdly low or nonexistant estimates (very rare)
df = df[(df['Est Low'] > 500) & (df['Est High'] > 500)]
print(f"after dropping weirdly low estimates: {len(df)}")

# Fix weirdo nationalities and define some useful aggregates for under-represented countries
df['Nationality'] = df['Nationality'].map(lambda x: 'Belgian' if x in ['Belg'] else x)
df['Nationality'] = df['Nationality'].map(lambda x: 'Czech' if x in ['Bohe', 'Bohemian'] else x)
df['Nationality'] = df['Nationality'].map(lambda x: 'Austrian' if x in ['Ausi'] else x)
df['Nationality'] = df['Nationality'].map(lambda x: 'British' if x in ['Brit', 'English', 'Scottish', 'Irish'] else x)
df['Nationality'] = df['Nationality'].map(lambda x: 'French' if x in ['Fren'] else x)
df['Nationality'] = df['Nationality'].map(lambda x: 'Polish' if x in ['Poli'] else x)
df['Nationality'] = df['Nationality'].map(lambda x: 'Dutch' if x in ['Dutc', 'Netherlander'] else x)
df['Nationality'] = df['Nationality'].map(lambda x: 'Danish' if x in ['Dani'] else x)
df['Nationality'] = df['Nationality'].map(lambda x: 'Danish' if x in ['Dani'] else x)
df['Nationality'] = df['Nationality'].map(lambda x: 'Spanish' if x in ['Span'] else x)
df['Nationality'] = df['Nationality'].map(lambda x: 'German' if x in ['Germ'] else x)
df['Nationality'] = df['Nationality'].map(lambda x: 'Italian' if x in ['Ital'] else x)
Eastern_nationalities = ['Polish', 'Hungarian', 'Czech', 'Armenian', 'Serbian', 'Turkish', 'Croatian',
                         'Russian', 'Ukrainian', 'Romanian', 'Bulgarian', 'Greek', 'Israeli']
df['Nationality'] = df['Nationality'].map(lambda x: 'Eastern' if x in Eastern_nationalities else x)
Scandanavian_nationalities = ['Norwegian', 'Finnish', 'Swedish', 'Danish']
df['Nationality'] = df['Nationality'].map(lambda x: 'Scandanavian' if x in Scandanavian_nationalities else x)
NewWorld_nationalities = ['American', 'Peruvian', 'Brazilian', 'Dominican']
df['Nationality'] = df['Nationality'].map(lambda x: 'NewWorld' if x in NewWorld_nationalities else x)
df['Nationality'] = df['Nationality'].map(lambda x: 'Spanish' if x in ['Portuguese'] else x)



# Keep only rows that can be dated to ~19th Century
df = df[(df['Date'] > 1780) & (df['Date'] < 1920)]
print(f"after forcing ~19th century: {len(df)}")

# Shuffle the rows (better for CV later) and reset the global index
df = df.sample(frac=1).reset_index(drop=True)
print(f"after shuffling: {len(df)}")
      
# Write cleaned output
df.to_pickle(df_file_code + '_cleaned.pkl')

# Stats
print("------------------------------------------")
print(f"Raw   data = {len(df_raw)} records")
print(f"Clean data = {len(df)} records")
