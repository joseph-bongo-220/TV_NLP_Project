# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import json
from app_config import get_config

config = get_config()

class Genius_TV_Scraper(object):
    """This object scrapes Genius for scipts to game of thrones and the office"""
    def __init__(self, show, seasons=None):
        """Initialize Seasons and URLs with error handling"""

        # base url should be placed in a config file
        base_url = config["app"]["base_url"]

        show = self.clean_show(show)

        if show == "The office":
            show = "The office us"

        self.show = show
        show_url = re.sub(" ", "-", show)
        show_url = show_url + "/"

        if show == "Game of thrones":
            # GOT extension should be put in config file
            extension = config["Game of thrones"]["extension"]
            GOT_season = config["Game of thrones"]["seasons"]
            if seasons is None:
                seasons = range(1,GOT_season+1)
            for i in seasons:
                if i not in range(1, 8):
                    raise ValueError("Episodes from Game of Thrones seasons 1-7 are available on Genius.")
        elif show == "The office us":
            # Office extension should be put in config file
            extension = config["The office us"]["extension"]
            TO_Season = config["The office us"]["seasons"]
            if seasons is None:
                seasons = range(1,TO_Season+1)
            for i in seasons:
                if i not in range(1,3):
                    raise ValueError("Episodes from The Office US seasons 1-2 are available on Genius.")
        else:
            raise ValueError("Scraper pulls data from Game of Thrones or The Office. Set show equal to 'Game of thrones' or 'The office us'")
        self.seasons = [i for i in seasons]
        self.urls = [base_url+show_url+extension.format(i) for i in seasons]
    
    def clean_show(self, show):
        """Clean the show name for processing"""
        first_letter = show[0].upper()
        rest_of_show = show[1:len(show)].lower()
        show = first_letter + rest_of_show
        return show

    #fix issues with list comprehensions
    def get_scripts(self, json=False):
        """Get the scripts and return them as dataframes or JSON"""
        temp_show = re.sub(" ", "-", self.show)
        show_df = pd.DataFrame()

        for url in self.urls:
            season_df = pd.DataFrame()
            response = requests.get(url, timeout=5)
            content = BeautifulSoup(response.content, "html.parser")
            block = content.find_all("a", {"class": "u-display_block"}, href=True)
            links = [b.get("href") for b in block]

            for link in links:
                # use regex  and re.sub to extract episode from URL (THE OFFICE)
                pattern = re.compile("(?<=https://genius.com/"+temp_show+"-)(.*)(?=-annotated)")
                match = re.search(pattern, link)
                episode = match.group(0)
                episode = re.sub("-", " ", episode)

                #get soups for links
                res = requests.get(link, timeout = 5)
                episode_content = BeautifulSoup(res.content, "html.parser")

                ep_script_list = [episode_content.find_all("p")[x].text for x in range(len(episode_content.find_all("p"))) if x!=len(episode_content.find_all("p"))-1 or x==0]
                ep_script = " ".join(ep_script_list)

                # Create regex pattern to parse out names of speakers (first capture group) and the quotes (second capture group)
                lines = re.findall(r"(?P<narration>(?<=^).*(?=\n)|(?<=\n).*(?=\n)|(?<=\n).*(?=$))", ep_script)
                lines=[l for l in lines if l!=""]
                
                names_n_quotes=[]
                names = []
                quotes =[]
                line_counter = 0
                for line in lines:
                    lines2 = re.findall(r"(?P<name>(?<=^)[A-Z]{1}.*?(?=:| \()|(?<=\n)[A-Z]{1}.*?(?=:| \()).*: (?P<quote>.*)", line)
                    names_n_quotes.append(lines2)
                    if names_n_quotes[line_counter] == []:
                        x = ""
                        y = ""
                    else:
                        x= names_n_quotes[line_counter][0][0]
                        y= names_n_quotes[line_counter][0][1]
                    x = re.sub(r"[ ]*\(.*\)", "", x)
                    names.append(x.upper().strip())
                    quotes.append(y)
                    line_counter+=1

                # create pandas dataframe to store name, quote, show, and episode
                ep_dict = {"Character_Name": names,
                            "Line": quotes,
                            "Narration": lines,
                            "Show": [self.show for s in range(0, len(lines))],
                            "Episode": [episode.lower() for e in range(0, len(lines))],
                            "URL": [link for l in range(0, len(lines))]}

                ep_df = pd.DataFrame(data=ep_dict)

                season_df = pd.concat([season_df, ep_df])

            show_df = pd.concat([show_df, season_df])
            show_df = show_df.reset_index(drop=True)

        if json == False:
            return(show_df)

        elif json == True:
            json_data = show_df.to_json(orient="index")
            json_data = re.sub(r"\\u2019", "’", json_data)
            return(json_data)

        else:
            raise TypeError("JSON variable must be of type boolean.")

def correct_characters(df, show, min_match=config["app"]["min_fuzzy_matching_ratio"], min_partial_match=config["app"]["min_partial_fuzzy_matching_ratio"]):
    character_dict_path = config[show]["character_config"]
    with open(character_dict_path) as f:
        character_dict = json.load(f)

    for key, value in character_dict.items():
        for name in value:
            if key == "":
                for i in list(df["Character_Name"].loc[df["Character_Name"]==name].index):
                    df["Narration"][i]=df["Line"][i]
                    df["Line"][i] = ""
            # impliment some fuzzy matching on the names
            df["Character_Name"].loc[df["Character_Name"]==name]=key
    
    return df