# pdf scanner
import PyPDF2
from pypdf import PdfReader
import pdfminer
# arrays
import pandas as pd
import numpy as np
# os
import os
# NLP
    # regex
import re
    # unicode
import unidecode

# scaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
# splitter
from sklearn.model_selection import train_test_split
# imputer
from sklearn.impute import SimpleImputer

def pull_images(pdf, name):
    '''
    Pulls images from pdf and saves them as jpgs -- 
    utilizes pypdf2 and pypdf
    '''
    # new directory for each PDF
    if not os.path.exists(name):
        os.makedirs(name)

    # reader object
    reader = PdfReader(pdf)
    for page in reader.pages:
        # number of pages within data pdf
        for image in page.images:
            # open as write binary
            with open(os.path.join(name, image.name), "wb") as fp:
                fp.write(image.data)

def pull_text(pdf):
    '''
    Pulls text from pdf and saves them as txts --
    utilizes pypdf2
    '''
    bodies = []
    # creating a pdf file object
    pdfFileObj = open(f'./{pdf}', 'rb')

    # creating a pdf reader object
    pdfReader = PyPDF2.PdfReader(pdfFileObj)

    # printing number of pages in pdf file
    num_pages = len(pdfReader.pages)
    print(f"Number of pages: {num_pages}")

        # loop through all pages and extract text
    for page_num in range(num_pages):
        pageObj = pdfReader.pages[page_num]
        text = pageObj.extract_text()
        bodies.append(text)
        print(f"Page {page_num+1}:\ndata.pdf")
        # validate the process body of text is correct format
        if type(text) is str:
            print('Successfully pulled text from page.\n')
        else:
            print('Failed to pull text.')
    # closing the pdf file object
    pdfFileObj.close()

    return bodies


def clean_bodies(text):
    '''
    Removal of extra text at top of document.

    A quick replace of any additional text.
    '''

    # decoding letters with accents into regular letters & removing character symbols
    clean_body = []
    for body in text:
        body = unidecode.unidecode(body)
        # removing the title of pdf from the body of text
        body = body.replace('Price Database31 October 2023','')
        clean_body.append(body)
    
    return clean_body 

def kaws(kaws_clean):
    '''
    This function cleans up the KAWS entries by adding a space after the artist name and removing double spaces.
    It also replaces variations of the artist name with the artist name.
    '''
    kaws_clean = list(set(kaws_clean))

    kaws_ = []
    for i in kaws_clean:
        # splitting by new line
        entries = i.split('\n')
        for entry in entries:
            # isolating single that has banksy in the lot name
            if 'DISSECTEDCOMPANION' in entry:
            # adding space after artist name for ease of regex
                entry = entry.replace('KAWS','KAWS ')
                # removing double spaces
                entry = entry.replace('  ', ' ')
                kaws_.append(entry)
            else:
                pass

    # cleaned KAWS entries with multiple appearances of names
    return kaws_

def banksy(banksy_clean):
    '''
    This function cleans up the Banksy entries by adding a space after the artist name and removing double spaces.
    It also replaces variations of the artist name with the artist name.
    '''
    banksy_clean = list(set(banksy_clean))

    banksy_ = []
    for i in banksy_clean:
        # splitting by new line
        entries = i.split('\n')
        for entry in entries:
            # adding space after artist name for ease of regex
            entry = entry.replace('Banksy', 'Banksy ')
            if 'KAWS' in entry:
                pass
            else:
                # removing double spaces
                entry = entry.replace('  ', ' ')
                banksy_.append(entry)

    # cleaned Banksy entries with multiple appearances of names
    return banksy_

def rembrandt(rembrandt_clean):
    '''
    This function cleans up the Rembrandt entries by adding a space after the artist name and removing double spaces.
    It also replaces variations of the artist name with the artist name.
    '''
    rembrandt_clean = list(set(rembrandt_clean))

    rembrandt_ = []
    for i in rembrandt_clean:
        # splitting by new line
        entries = i.split('\n')
        for entry in entries:
            # adding space after artist name for ease of regex
            entry = entry.replace('Rembrandt van Rijn', 'Rembrandt van Rijn ')
            entry = entry.replace('Circle of Rembrandt van Rijn', 'Rembrandt van Rijn ')
            entry = entry.replace('Studio of Rembrandt van Rijn', 'Rembrandt van Rijn ')
            entry = entry.replace('School of Rembrandt van Rijn', 'Rembrandt van Rijn ')
            entry = entry.replace('Follower of Rembrandt van Rijn', 'Rembrandt van Rijn ')
            entry = entry.replace('Workshop of Rembrandt van Rijn', 'Rembrandt van Rijn ')
            # removing double spaces
            entry = entry.replace('  ', ' ')
            rembrandt_.append(entry)

    rembrandt_ = [body for body in rembrandt_clean if len(body) <= 450]
    # cleaned Rembrandt entries with multiple appearances of names
    return rembrandt_

def marc(marc_clean):
    '''
    This function cleans up the Marc Chagall entries by adding a space after the artist name and removing double spaces.
    It also replaces variations of the artist name with the artist name.
    '''
    marc_clean = list(set(marc_clean))

    marc_ = []
    for i in marc_clean:
        # splitting by new line
        entries = i.split('\n')
        for entry in entries:
            # adding space after artist name for ease of regex
            entry = entry.replace('Marc Chagall', 'Marc Chagall ')
            entry = entry.replace('After Marc Chagall', 'Marc Chagall ')
            # removing double spaces
            entry = entry.replace('  ', ' ')
            if 'Marc Chagall' in entry:
                marc_.append(entry)

    # cleaned Marc Chagall entries with multiple appearances of names
    return marc_

def pablo(pablo_clean):
    '''
    This function cleans up the Pablo Picasso entries by adding a space after the artist name and removing double spaces.
    It also replaces variations of the artist name with the artist name.
    '''
    pablo_clean = list(set(pablo_clean))

    pablo_ = []
    for i in pablo_clean:
        # splitting by new line
        entries = i.split('\n')
        for entry in entries:
            # adding space after artist name for ease of regex
            entry = entry.replace('Pablo Picasso', 'Pablo Picasso ')
            entry = entry.replace('After Pablo Picasso', 'Pablo Picasso ')
            # removing double spaces
            entry = entry.replace('  ', ' ')
            pablo_.append(entry)

    # cleaned Pablo Picasso entries with multiple appearances of names
    return pablo_

def dali(dali_clean):
    '''
    This function cleans up the Salvador Dali entries by adding a space after the artist name and removing double spaces.
    It also replaces variations of the artist name with the artist name.
    '''
    dali_clean = list(set(dali_clean))

    dali_ = []
    for i in dali_clean:
        # splitting by new line
        entries = i.split('\n')
        for entry in entries:
            # adding space after artist name for ease of regex
            entry = entry.replace('Dali', 'Dali ')
            entry = entry.replace('After Salvador Dali', 'Salvador Dali ')
            # removing double spaces
            entry = entry.replace('  ', ' ')
            dali_.append(entry)

    # cleaned Salvador Dali entries with multiple appearances of names
    dali_ = dali_[1:]
    return dali_

def add_ins(new_body,add_ins):
    '''
    Adds values that had multiple counts of artist names --
    As well as removing entries that do not have complete data.
    '''
    print(f'Before adding additional values: {len(new_body)}')

    # new_body.extend(add_ins)
    # Convert each item to a set of words
    set_new_body = [set(item.split()) for item in new_body]
    set_add_ins = [set(item.split()) for item in add_ins]

    # Find items in new_body that are not similar to any item in add_ins
    unique_items = [item for item, s in zip(new_body, set_new_body) if s not in set_add_ins]

    # Merge unique items from new_body with add_ins
    new_body = unique_items + add_ins
    print(f'After adding additional values: {len(new_body)}')

    print('Some prep work before regex.')

    # Removing entries from new_body that do not have complete data
    print(f'Before: {len(new_body)}')
    new_body = [body for body in new_body if len(body) >= 150]
    new_body = [body for body in new_body if len(body) <= 530]
    print(f'After: {len(new_body)}')
    return new_body

def regex_foreign_currency(new_body):
    '''
    This function applies regular expressions to extract specific fields from the text data.
    
    Parameters:
    new_body (list): A list of text data to be processed.

    Returns:
    df (DataFrame): A pandas DataFrame containing the extracted fields as columns.
    '''
    data = []
    patterns = {
        'artist': r'^(Zhang Daqian|Andy Warhol|Banksy|Salvador Dali|Marc Chagell|Pablo Picasso|\
                        Rembrandt van Rijn|KAWS|Leonard Tsuguharu Foujita|Yayoi Kusama)',  # Matches any of the provided artist names at the start of the string
        'dimension_cm': r'(Height.*?cm)',  # dimensions in cm start with 'Height' and end with 'cm'
        'dimension_in': r'(Height.*?in)',  # dimensions in inches start with 'Height' and end with 'in'
        'year_created': r'in.*?(\d{4})',  # year is a 4-digit number before 'Edition'
        'date_sold': r'((?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4})',  # Matches 'Month YYYY'
        'auction_house': r'ago(.*?)(?=\[Lot)',  # auction house name is after 'ago' and before '[Lot'
        #'lot': r'\[Lot (.*?)\]',  # lot # dropped due to not useful
        'estimate_usd_low': r'(?:.*est\.\s*[\d,]+\s*u\s*[\d,]+\s*\w+\s*[\d,]+\s*\w+\s*est\.\s*)?([\d,]+)\s*u',  # low estimate in USD is the number (possibly with commas) after the second 'est. ' and before 'u'
        'estimate_usd_high': r'u\s*([\d,]+)\s*USD',  # high estimate in USD is a number (possibly with commas) after 'u ' and followed by 'USD'
        'hammer_price': r'USD([\d,\.]+)USD',
        'percent_estimate': r'USD\|\s*(\d+)% est',  # Matches 'X% est' after 'USD| '
    }

    for text in new_body:
        # Extract fields
        fields = {}
        for field, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                fields[field] = match.group(1)
            else:
                fields[field] = None
        fields = {}
        for field, pattern in patterns.items():
            if field == 'edition':
                fields[field] = 'yes' if 'edition' in text.lower() else 'no'
            else:
                match = re.search(pattern, text)
                if match:
                    fields[field] = match.group(1)
                else:
                    fields[field] = None

        # Extract title_medium from the remaining text
        match = re.search(r'(.*?)(?=Height)', text)
        if match:
            fields['title_medium'] = match.group(1).strip()
        else:
            fields['title_medium'] = None

        data.append(fields)

    df = pd.DataFrame(data)
    return df

def no_foreign_currency_regex(new_body):
    '''
    This function applies regular expressions to extract specific fields from the text data.
    
    Parameters:
    new_body (list): A list of text data to be processed.

    Returns:
    df (DataFrame): A pandas DataFrame containing the extracted fields as columns.
    '''
    data = []
    patterns = {
        'artist': r'^(Zhang Daqian|Andy Warhol|Banksy|Salvador Dali|Marc Chagell|Pablo Picasso|\
                        Rembrandt van Rijn|KAWS|Leonard Tsuguharu Foujita|Yayoi Kusama)',  # Matches any of the provided artist names at the start of the string
        'dimension_cm': r'(Height.*?cm)',  # Assumes dimensions in cm start with 'Height' and end with 'cm'
        'dimension_in': r'(Height.*?in)',  # Assumes dimensions in inches start with 'Height' and end with 'in'
        'year_created': r'in.*?(\d{4})',  # Assumes year is a 4-digit number before 'Edition'
        'date_sold': r'((?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4})',  # Matches 'Month YYYY'
        'auction_house': r'ago(.*?)(?=\[Lot)',  # Assumes auction house name is after 'ago' and before '[Lot'
        #'lot': r'\[Lot (.*?)\]',  # Matches 'Lot X' inside square brackets
        'estimate_usd_low': r'est\.\s*([\d,]+)\s*u',  # Assumes low estimate in USD is the second number (possibly with commas) after ']est. ' or 'currencyest. ' and before 'u'
        'estimate_usd_high': r'u\s*([\d,]+)\s*USD',  # Assumes high estimate in USD is a number (possibly with commas) after 'u ' and followed by 'USD'
        'hammer_price': r'USD([\d,\.]+)USD',
        'percent_estimate': r'USD\|\s*(\d+)% est',  # Matches 'X% est' after 'USD| '
    }

    for text in new_body:
        # Extract fields
        fields = {}
        for field, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                fields[field] = match.group(1)
            else:
                fields[field] = None
        fields = {}
        for field, pattern in patterns.items():
            if field == 'edition':
                fields[field] = 'yes' if 'edition' in text.lower() else 'no'
            else:
                match = re.search(pattern, text)
                if match:
                    fields[field] = match.group(1)
                else:
                    fields[field] = None

        # Extract title_medium from the remaining text
        match = re.search(r'(.*?)(?=Height)', text)
        if match:
            fields['title_medium'] = match.group(1).strip()
        else:
            fields['title_medium'] = None

        data.append(fields)

    df = pd.DataFrame(data)
    return df

def clean_prep_df(df):
    '''
    This function cleans the DataFrame by converting columns to the correct data types and creating new columns.
    Some cleaning and preparing include:
        1. Filling None values with np.NaN
        2. changings dtypes to numerical data
        3. changing other dtypes
        4. create dummy columns for format and medium to express the various types of art and its methodology of creation.
        5. retrieving month sold of art
        6. Preparing dimensions column and dropping cm (only in in)
        7. Impuding nulls of percent estimate with 0s to express the hammer_price was within the estimate
        8. Auction House dummy columns
        9. dummy columns for artists
        10. lowering letter cases
        11. Impuding values that needed (very little rows needed)
        12. transforming hammer_price logarthimically
    '''

    # changing nonetypes to NaN
    df = df.fillna(value=np.nan)

    # Cleaning monetary values
    df.estimate_usd_low = df.estimate_usd_low.str.replace(',', '').astype(float)
    df.estimate_usd_high = df.estimate_usd_high.str.replace(',', '').astype(float)
    df.hammer_price = df.hammer_price.str.replace(',', '').astype(float)

    # Zhang Daqian has missing values in his year_created column, this was not being expressed with regex
    df.loc[df['artist'] == 'Zhang Daqian', 'year_created'] = df.loc[df['artist'] == 'Zhang Daqian', 'year_created'].replace('2022', 'None')

    # Ensure 'title_medium' column is of type string
    df['title_medium'] = df['title_medium'].astype(str)

    # Create new columns for identifying the format at which the art is
    df['is_paper'] = df['title_medium'].str.contains('|'.join(['scroll', 'page', 'book', 'paper', 'papers', 'sheet', 'vellum']), case=False).astype(int)
    df['is_print'] = df['title_medium'].str.contains('|'.join(['silkscreen', 'screenprint', 'screenprints', 'lithograph',\
                                                       'photolithograph', 'stencil', 'lithographs', 'printed']), case=False).astype(int)
    df['is_sculpture'] = df['title_medium'].str.contains('|'.join(['bronze', 'figure', 'porcelain', 'silver', 'gold', 'ceramic',\
                                                           'earthenware', 'terre de', 'faience', 'clay', 'multiple', 'reinforced', 'plushes',\
                                                              'outfits']), case=False).astype(int)
    df['is_canvas'] = df['title_medium'].str.contains('|'.join(['canvas','canvasboard']), case=False).astype(int)
    df['is_other_format'] = df['title_medium'].str.contains('|'.join(['woodcut', 'etching', 'board', 'arches', 'drypoint', 'xylography', 'xylographies',\
                                                              'linoleum', 'panel', 'polaroid', 'portfolio', 'tapestry', 'mixed media', 'drawing',\
                                                                'fabric']), case=False).astype(int)

    # Create new column for identifying artwork medium
    df['is_ink'] = df['title_medium'].str.contains('|'.join(['pen','ink','brush']), case=False).astype(int)
    df['is_paint'] = df['title_medium'].str.contains('|'.join(['oil','paint','acrylic','gouache','watercolor','watercolour',\
                                                      'pastel','tempera','pastel','aquatint']), case=False).astype(int)
    df['is_pencil'] = df['title_medium'].str.contains('|'.join(['pencil','graphite','charcoal','crayon']), case=False).astype(int)
    df['is_pottery'] = df['title_medium'].str.contains('|'.join(['pottery','ceramic','earthenware','terracotta','clay']), case=False).astype(int)
    df['is_other_medium'] = df['title_medium'].str.contains('|'.join(['wool', 'mixed media', 'plastic', 'polyester', 'dior', 'fiberglass', 'mixed-media',\
                                                              'walnut', 'wood', 'vinyl', 'drawing']), case=False).astype(int)
                                                             
    # retrieving month sold -- all sales are from 2022 so no changes needed
    df['month_sold'] = pd.to_datetime(df['date_sold']).dt.month

    # creating columns for dimensions in inches & dropping cm column
    df.drop(columns=['dimension_cm'], inplace=True)
        # inches dimensions text clean up
    df['dimension_in'] = df['dimension_in'].str.split('cm.').str[1]
    df['height_in'] = df['dimension_in'].str.extract('Height (\d+\.?\d*)', expand=False).astype(float)
    df['width_in'] = df['dimension_in'].str.extract('Width (\d+\.?\d*)', expand=False).astype(float)

    # Percent estimate as float
    df['percent_estimate'] = df['percent_estimate'].astype(float)
    df['percent_estimate'].fillna(value=0, inplace=True)

    # Inputting some auction house dummy columns
    df['is_sothebys'] = 0
    df.loc[df['auction_house'].fillna('').str.contains('sotheby', case=False), 'is_sothebys'] = 1
    df['is_christies'] = 0
    df.loc[df['auction_house'].fillna('').str.contains('christie', case=False), 'is_christies'] = 1
    df['is_phillips'] = 0
    df.loc[df['auction_house'].fillna('').str.contains('phillip', case=False), 'is_phillips'] = 1
    df['is_bonhams'] = 0
    df.loc[df['auction_house'].fillna('').str.contains('bonham', case=False), 'is_bonhams'] = 1
    df['is_other_house'] = 0
    df.loc[df['auction_house'].fillna('').str.contains('sotheby|christie|phillip|bonham', case=False) == False, 'is_other_house'] = 1
    
    # Correcting NaN artist with starting piece of title_medium
    for index, row in df.iterrows():
        if pd.isnull(row['artist']):
            if isinstance(row['title_medium'], str):
                if row['title_medium'].startswith('Marc Chagall'):
                    df.loc[index, 'artist'] = 'Marc Chagall'
                elif row['title_medium'].startswith('Rembrandt van Rijn'):
                    df.loc[index, 'artist'] = 'Rembrandt van Rijn'

    # creating dummy columns for artist
    artist_dummies = pd.get_dummies(df['artist'],dtype=int)
    df = pd.concat([df,artist_dummies],axis=1)

    # dropping duplicates
    df = df.dropna(subset=['hammer_price'])
    df = df.drop_duplicates(subset=['title_medium'])

    # changing estimates that are under to negatives
    df = df.reset_index(drop=True)
    df.loc[df['hammer_price'] < df['estimate_usd_low'], 'percent_estimate'] = df.loc[df['hammer_price'] < df['estimate_usd_low'], 'percent_estimate'] * -1

    # imputing values that are missing from data
    df['year_created'] = df['year_created'].replace('None', np.nan).fillna(0).astype(float)
    list_index = [26, 60, 147, 227, 232, 293, 672, 680, 720, 834, 920, 968, 977, 978, 980]
    df.loc[list_index, 'year_created'] = np.NaN

    # Create a new column that contains the first name in title_medium
    df['artist'] = df['title_medium'].str.extract('(\w+)', expand=False)

    # lower names
    names = ['Zhang Daqian', 'Andy Warhol', 'Banksy', 'Salvador Dali','Marc Chagall','Pablo Picasso',
         'Rembrandt van Rijn','KAWS','Leonard Tsuguharu Foujita','Yayoi Kusama']
    names = [name.lower().strip() for name in names]

    # Replace the short artist names in the artist column with the full names
    df['artist'] = df['artist'].apply(lambda short_name: next((full_name for full_name in names if short_name.lower().strip() in full_name), short_name))

    # using median to impute missing values
    imputer = SimpleImputer(strategy='median')
    df.year_created = imputer.fit_transform(df.year_created.values.reshape(-1,1))
    df.height_in = imputer.fit_transform(df.height_in.values.reshape(-1,1))
    df.width_in = imputer.fit_transform(df.width_in.values.reshape(-1,1))

    # lowering all columns
    df.columns = df.columns.str.lower()

    # log transforming hammer_price
    df['log_hammer_price'] = np.log10(df['hammer_price'])
    return df

def splitter(df, stratify=None):
    '''
    Returns
    Train, Validate, Test from SKLearn
    Sizes are 60% Train, 20% Validate, 20% Test
    '''
    if stratify is None:
        train, temp = train_test_split(df, test_size=.4, random_state=4343)
        validate, test = train_test_split(temp, test_size=.5, random_state=4343)
    else:
        train, temp = train_test_split(df, test_size=.4, random_state=4343, stratify=df[stratify])
        validate, test = train_test_split(temp, test_size=.5, random_state=4343, stratify=temp[stratify])

    print(f'Dataframe: {df.shape}', '100%')
    print(f'Train: {train.shape}', '| ~60%')
    print(f'Validate: {validate.shape}', '| ~20%')
    print(f'Test: {test.shape}','| ~20%')

    return train, validate, test
