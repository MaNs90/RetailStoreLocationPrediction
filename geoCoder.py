# -*- coding: utf-8 -*-

from geopy.geocoders import Nominatim
from geopy.distance import vincenty
from difflib import SequenceMatcher, get_close_matches
from collections import Counter
import operator
import itertools
import googlemaps
import math
import numpy as np
import csv
import json
import os
import time
import re

# This class is used to search for nearby places using the latitude and longitude.
# Currently there are two ways: OSM Nominatum which is open source but limited to one request per second and does not contain all the places.
# The second way is Google Places API as it is much more accurate than OSM Nominatum and it contains 150000 free requests per day.

class geoCoder:
    def __init__(self):
        self.geolocator = Nominatim()
        
        self.gmaps = googlemaps.Client(key="AIzaSyBUmjs3xEVPJhAVZyc5JbJ0q8OhNfj2DoI")
        
        self.csv_file = []
        # The data structure used here is a list of dicts where each dict is a row in the CSV file with header as key and attribute as value
        with open("C:/Users/ahmed/OneDrive/Documents/Masters Courses/Dissertation/SQL Databases/OwnershipCoordinatesFinal2.csv",encoding='utf8') as sqldbCSV:
            csvread = csv.reader(sqldbCSV, delimiter=",")
            count = -1
            for line in csvread:
                #print(line)
                count = count + 1
                # Ignoring the header row
                
                if count == 0:
                    continue
                
                my_dict = {}
                #my_dict["ID"] = line[0]
                my_dict["street_name"] = line[0]
                my_dict["town_name"] = line[1]
                my_dict["postcode"] = line[2]
                my_dict["company_number"] = line[3].upper()
                my_dict["tenure"] = line[4]
                my_dict["date"] = line[5]
                my_dict["company_name"] = line[6]
                my_dict["title_number"] = line[7]
                my_dict["SIC"] = line[8]
                my_dict["latitude"] = line[9]
                my_dict["longitude"] = line[10]
                
                self.csv_file.append(my_dict)
        '''        
        self.official_data = {}        
        # Get all the needed CSVs and parse them
        for root, dirs, files in os.walk("C:/Users/ahmed/Downloads/BasicCompanyDataAsOneFile-2017-03-06/"):
            for file in files:
                if file.endswith(".csv"):
                    with open(root+file,encoding='utf8') as sqldbCSV:
                        csvread = csv.reader(sqldbCSV, delimiter=",")
                        count = -1
                        for line in csvread:
                            count = count + 1
                            if count == 0:
                                continue
                            
                            self.official_data[line[1]] = line[0]          
        '''
    
    def test_json(self, file):
        
        csv_dict = {}
        
        for dicts in self.csv_file:
            csv_dict[dicts["company_number"]] = dicts["company_name"]
        
        with open(file, "r" ,encoding='utf8') as jsontext:
            json_data = json.load(jsontext)
            
            count = 0
            for element in json_data:
                element["company_name"] = csv_dict[element["company_number"]]
                count +=1
                    
            print(count)
        
        with open('myJSONModification.json', 'w') as fout: 
            json.dump(json_data, fout, sort_keys=True, indent = 4)    
            
    def normalize_Companyname(self):
        # This will make all company names consistent with the unified company beta house website naming convention
        count = 0
        for i, dicts in enumerate(self.csv_file):
            if dicts["company_number"] in self.official_data:
                self.csv_file[i]["company_name"] = self.official_data[dicts["company_number"]] 
                count += 1

        
        
    def categorize_Google(self,file):
        
        with open("items.json","r", encoding = 'utf8') as websites:
            company_websites = json.load(websites)    
        #print(company_websites)
        websites_dict = {}
        for record in company_websites:
            websites_dict[record["name"]] = record["website"]
        
        #for k,v in websites_dict.items():
        #    print(k,v)     
        
        with open(file, "r" ,encoding='utf8') as jsontext:
            self.json_data = json.load(jsontext)
            
        types_dict = {} 
        for elements in self.json_data:
            name = elements["company_name"]
            #print(name)
            if name in websites_dict:    
                website = websites_dict[name]
                if website:  
                    if "www" in website:
                        website2 = "".join(re.findall("\.([A-Za-z0-9-]+)\.", website.lower()))     
                    else:
                        website2 = "".join(re.findall("/([A-Za-z0-9-]+)\.", website.lower()))    
            #print(name)
            if elements["google_data"] != "None":
                data = elements["google_data"]["results"]
                #print("data ", len(data), name)
                if "google_data_nextpage" in elements:
                    data2 = elements["google_data_nextpage"]
                    # print("data2 ",len(data2["results"]), name)
                    if "google_data_nextpage2" in elements:
                        data3 = elements["google_data_nextpage2"]
                    #    print("data3 ",len(data3["results"]), name)
                    else:
                        data3 = {"results":[]}    
                else:
                    data2 = {"results":[]}  
                    data3 = {"results":[]}  
                    
                data.extend(data2["results"])
                data.extend(data3["results"])
                #print("data final ",len(data), name)   
                
                if "amsric" in name.lower().split():
                    name2 = "StarbucksKFC"
                elif "magawell ltd" in name.lower().split():
                    name2 = "evans pharmacy"
                elif name.lower() == "a. levy & son limited":
                    name2 = "Blue Inc." 
                elif name == "DB MODHA LIMITED":
                    name2 = "Pandora"
                elif name == "MACKAYS STORES LIMITED":
                    name2 = "M&Co."
                elif name == "WALLACE BOOKS LIMITED":
                    name2 = "David's Bookshop"
                elif name == "DSG RETAIL LIMITED":
                    name2 = "Currys PC World"
                elif name == "GREAT LONDON SOUVENIRS LTD":
                    name2 = "A Little Present"
                elif name == "HALA (SOUTH) LIMITED":
                    name2 = "Dominos Pizza"                
                elif name == "COLONEL FOODS LIMITED":
                    name2 = "KFC"
                elif name == "WILCARE CO. LIMITED":
                    name2 = "Pharmacy"         
                elif name == "MEXICAN GRILL LTD":
                    name2 = "tortilla"        
                elif name == "APPT CORPORATION LIMITED":
                    name2 = "Mcdonalds"
                elif name == "LA PLAGE LIMITED":
                    name2 = "Vilebrequin"
                elif name == "MARSH (BOLTON) LIMITED":
                    name2 = "Lloyd's pharmacy"
                elif name == "THE FAMOUS ENTERPRISE COCKTAIL COMPANY LIMITED":
                    name2 = "SPAR Tolley"   
                elif name == "MOLEND LIMITED":
                    name2 = "Govani"
                elif name == "KENTUCKY FRIED CHICKEN (GREAT BRITAIN) LIMITED":
                    name2 = "KFC"      
                elif name == "SPORTSWIFT LIMITED":
                    name2 = "Card Factory"  
                elif name == "CDS (SUPERSTORES INTERNATIONAL) LIMITED":
                    name2 = "The range" 
                elif name == "GPS (GREAT BRITAIN) LIMITED":
                    name2 = "Gap Inc"    
                elif name == "BESTWAY NATIONAL CHEMISTS LIMITED":
                    name2 = "Well Pharmacy"
                elif name == "AZZURRI RESTAURANTS LIMITED":
                    name2 = "ItalianZizzisCoco di"  
                elif name == "MULTI-TILE LIMITED":
                    name2 = "Topps tiles"              
                else:
                    name2 = name
                # Loop through each result in the element and check if the name is similar to the company name, if so then add its types to the dictionary.
                flag = False 
                for x in data:
                    # Regex to remove any brackets in the company name and whats inside the brackets.
                    #print("name2 = ", name2)
                    x1 = re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",name2.lower())
                    x2 = re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower())
                    if SequenceMatcher(None,x1,x2).ratio() >=0.6:
                        #flag = True
                        if name in types_dict:
                            #print(x["name"])
                            #print(x["types"])
                            #print(len(types_dict[name]), name)
                            types_dict[name].extend(x["types"])
                        else:
                            types_dict[name] = x["types"]   
                    else:
                        # There are two companies who have not been returned by the scrapping because it is not in their database
                        if website:        
                                # Regex to remove any brackets in the company name and whats inside the brackets.
                                #print("name2 = ", name2)
                                x1 = re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",website2.lower())
                                #x2 = re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower())
                                if SequenceMatcher(None,x1,x2).ratio() >=0.6:
                                    if name in types_dict:
                                        #print(x["name"])
                                        #print(x["types"])
                                        #print(len(types_dict[name]), name)
                                        types_dict[name].extend(x["types"])
                                    else:
                                        types_dict[name] = x["types"]                            
                                
                
        print(len(types_dict.keys()))
                            
        # Remove the very common words and get the most common category of each company
        
        for dict_key in types_dict.keys():
            while "establishment" in types_dict[dict_key]: types_dict[dict_key].remove("establishment") 
            while "store" in types_dict[dict_key]: types_dict[dict_key].remove("store") 
            while "point_of_interest" in types_dict[dict_key]: types_dict[dict_key].remove("point_of_interest") 
            while "political" in types_dict[dict_key]: types_dict[dict_key].remove("political")
            while "food" in types_dict[dict_key]: types_dict[dict_key].remove("food")
            while "locality" in types_dict[dict_key]: types_dict[dict_key].remove("locality")
            while "finance" in types_dict[dict_key]: types_dict[dict_key].remove("finance")
            while "transit_station" in types_dict[dict_key]: types_dict[dict_key].remove("transit_station")
            while "accounting" in types_dict[dict_key]: types_dict[dict_key].remove("accounting")
            while "general_contractor" in types_dict[dict_key]: types_dict[dict_key].remove("general_contractor")
            #while "lodging" in types_dict[dict_key]: types_dict[dict_key].remove("lodging")
            # If after removing those three categories the list is empty, add a None to it to avoid a index out of bounds exception
            if len(types_dict[dict_key]) == 0:
                types_dict[dict_key] = ["None"]
            #print(types_dict[dict_key]) 
            data = Counter(types_dict[dict_key])
            #print(data)
            types_dict[dict_key]= data.most_common(1)[0][0]
            #print(data.most_common(1)[0][0])   
        with open('google_categories_mostcommon2.json', 'w') as fout: 
            json.dump(types_dict, fout) 
        
        for k,v in types_dict.items():
            # Changelog REMOVED FOOD FROM RESTAURANTS AND CAFES
            if v in ["cafe","meal_delivery","meal_takeaway","restaurant", "bar", "night_club", "lodging"]:
                types_dict[k] = "Restaurants and Cafes"
            elif v in ["convenience_store","grocery_or_supermarket", "gas_station", "bakery","liquor_store"]:      
                types_dict[k] = "Food and Grocery"
            elif v in ["department_store","shopping_mall"]:   
                types_dict[k] = "Department_Stores"
            elif v in ["clothing_store","jewelry_store","shoe_store"]: 
                types_dict[k] = "Clothing and Accessories"
            elif v in ["beauty_salon","dentist","doctor","hair_care","hospital","pharmacy","spa", "health"]:
                types_dict[k] = "Health and Cosmetics"
            elif v in ["electronics_store","hardware_store"]: 
                types_dict[k] = "Electricals and Electronics"
            elif v in ["bicycle_store","book_store","car_dealer","car_rental","car_repair","car_wash", "art_gallery", "pet_store", "florist"]:      
                types_dict[k] = "Miscellaneous"
            elif v in ["home_goods","furniture_store","home_goods_store", "garden centre"]:  
                types_dict[k] =  "Homeware and DIY"
                
        for elements in self.json_data:
            if elements["company_name"] not in types_dict:
                types_dict[elements["company_name"]] = "None"       
        
        
        with open('Google_Categories_Remade.json', 'w') as fout: 
            json.dump(types_dict, fout)      
        
   
        return types_dict
        #print(types_dict["RIVER ISLAND CLOTHING CO. LIMITED"])  
        #while "establishment" in types_dict["RIVER ISLAND CLOTHING CO. LIMITED"]: types_dict["RIVER ISLAND CLOTHING CO. LIMITED"].remove("establishment") 
        #while "store" in types_dict["RIVER ISLAND CLOTHING CO. LIMITED"]: types_dict["RIVER ISLAND CLOTHING CO. LIMITED"].remove("store") 
        #while "point_of_interest" in types_dict["RIVER ISLAND CLOTHING CO. LIMITED"]: types_dict["RIVER ISLAND CLOTHING CO. LIMITED"].remove("point_of_interest") 
        #print(types_dict["RIVER ISLAND CLOTHING CO. LIMITED"]) 
        #data = Counter(types_dict["RIVER ISLAND CLOTHING CO. LIMITED"])
        #print(data.most_common(1)[0][0])       
          

    def reverse_GeocodeOSM(self,latlon):
        location = self.geolocator.reverse(latlon)
        splitting = location.address.split(",")
        print(location.address)
        print(location)
        return splitting
    
    
    def crawl_google_extra(self):
        csv_file = []
        with open("C:/Users/ahmed/OneDrive/Documents/Masters Courses/Dissertation/python_code/new_Data3.csv",encoding='utf8') as sqldbCSV:
            csvread = csv.reader(sqldbCSV, delimiter=",")
            count = -1
            for line in csvread:
                count = count + 1
                # Ignoring the header row
                if count == 0:
                    continue
                
                my_dict = {}
                #my_dict["ID"] = line[0]
                my_dict["street_name"] = line[0]
                my_dict["town_name"] = line[1]
                my_dict["postcode"] = line[5]
                my_dict["company_number"] = line[6].upper()
                my_dict["tenure"] = line[7]
                my_dict["date"] = line[8]
                my_dict["company_name"] = line[9]
                my_dict["title_number"] = line[10]
                my_dict["SIC"] = line[11]
                my_dict["latitude"] = line[12]
                my_dict["longitude"] = line[13]
                
                csv_file.append(my_dict)
        
        crawled_data = []
        count = 0
        for dicts in csv_file:
            #if count < 50:
                crawled_dict = {}
                #lat = dicts["Coordinates"].split(",")[0].replace("(","")
                #lon = dicts["Coordinates"].split(",")[1].replace(")","")
                lat = dicts["latitude"]
                lon = dicts["longitude"]
                #print(lat,lon)
                coordinates = (float(lat),float(lon))
                nearby_places = self.gmaps.places_nearby(coordinates,radius = 50)
                crawled_dict["company_name"] = dicts["company_name"]
                crawled_dict["company_number"] = dicts["company_number"]
                crawled_dict["title_number"] = dicts["title_number"]
                #crawled_dict["Polygon"] = dicts["Polygon"]
                crawled_dict["Coordinates"] = coordinates
                if nearby_places["status"] == "OK":
                    crawled_dict["google_data"] = nearby_places
                    if "next_page_token" in nearby_places:
                        token = nearby_places["next_page_token"]
                        # This sleep is because the next page is delayed before its available in google so I have to sleep the thread for two seconds until its ready
                        time.sleep(2)
                        nearby_places2 = self.gmaps.places_nearby(location = coordinates, radius = 50, page_token = token)
                        crawled_dict["google_data_nextpage"] = nearby_places2
                        if "next_page_token" in nearby_places2:
                            token2 = nearby_places2["next_page_token"]
                            #This sleep is because the next page is delayed before its available in google so I have to sleep the thread for two seconds until its ready
                            time.sleep(2)
                            nearby_places3 = self.gmaps.places_nearby(location = coordinates, radius = 50, page_token = token2)
                            crawled_dict["google_data_nextpage2"] = nearby_places3
                            
                else:
                    crawled_dict["google_data"] = "None"
                
                #count = count + 1  
                crawled_data.append(crawled_dict)  
         
        with open('crawledData_New.json', 'w') as fout:    
            json.dump(crawled_data, fout, sort_keys=True)
            
            
    def crawl_google(self, file):
        with open(file, "r" ,encoding='utf8') as jsontext:
            offset_data = json.load(jsontext)
        #reverse_geocode_result = self.gmaps.reverse_geocode((50.2263488135232, -5.26820519402275))
        #print(reverse_geocode_result)
        crawled_data = []
        count = 0
        for dicts in offset_data:
            #if count < 50:
                crawled_dict = {}
                #lat = dicts["Coordinates"].split(",")[0].replace("(","")
                #lon = dicts["Coordinates"].split(",")[1].replace(")","")
                lat = dicts["latitude"]
                lon = dicts["longitude"]
                #print(lat,lon)
                coordinates = (float(lat),float(lon))
                nearby_places = self.gmaps.places_nearby(coordinates,radius = 20)
                crawled_dict["company_name"] = dicts["company_name"]
                crawled_dict["company_number"] = dicts["company_number"]
                crawled_dict["title_number"] = dicts["title_number"]
                #crawled_dict["Polygon"] = dicts["Polygon"]
                crawled_dict["Coordinates"] = coordinates
                if nearby_places["status"] == "OK":
                    crawled_dict["google_data"] = nearby_places
                    if "next_page_token" in nearby_places:
                        token = nearby_places["next_page_token"]
                        # This sleep is because the next page is delayed before its available in google so I have to sleep the thread for two seconds until its ready
                        time.sleep(2)
                        nearby_places2 = self.gmaps.places_nearby(location = coordinates, radius = 20, page_token = token)
                        crawled_dict["google_data_nextpage"] = nearby_places2
                        if "next_page_token" in nearby_places2:
                            token2 = nearby_places2["next_page_token"]
                            #This sleep is because the next page is delayed before its available in google so I have to sleep the thread for two seconds until its ready
                            time.sleep(2)
                            nearby_places3 = self.gmaps.places_nearby(location = coordinates, radius = 20, page_token = token2)
                            crawled_dict["google_data_nextpage2"] = nearby_places3
                            
                else:
                    crawled_dict["google_data"] = "None"
                
                #count = count + 1  
                crawled_data.append(crawled_dict)  
         
        with open('crawledData_Offset.json', 'w') as fout:    
            json.dump(crawled_data, fout, sort_keys=True)
                     
    
    def mergeJSON(self,file1,file2):
        # This function is to merge the initial main JSON file and the new JSON file of missing data crawled at 100m radius.
        
        with open(file1, "r" ,encoding='utf8') as jsontext:
            json_data = json.load(jsontext)
            
            with open(file2, "r" ,encoding='utf8') as jsontext2:
                json_data2 = json.load(jsontext2)
            count = 0    
            for elements2 in json_data2:
                title = elements2["title_number"]
                for index,elements in enumerate(json_data):
                    if title == elements["title_number"]:
                        count += 1
                        json_data[index] = elements2
                        print(count)
                        break
        
        with open('JSON_data_merged_test.json', 'w') as fout: 
            json.dump(json_data, fout, sort_keys=True, indent = 4)    
        print("Done")        
                
    def modify_JSONsize(self, file):
        
        with open(file, "r" ,encoding='utf8') as jsontext:
            self.json_data = json.load(jsontext)
           
            for elements in self.json_data:
                if elements["google_data"] != "None":
                    for index,dict in enumerate(elements["google_data"]["results"]):
                        if "geometry" in dict:
                            del elements["google_data"]["results"][index]["geometry"]
                        if "icon" in dict:
                            del elements["google_data"]["results"][index]["icon"]
                        if "id" in dict:
                            del elements["google_data"]["results"][index]["id"]
                        if "reference" in dict:
                            del elements["google_data"]["results"][index]["reference"]
                        if "photos" in dict:
                            del elements["google_data"]["results"][index]["photos"]      
                if "google_data_nextpage" in elements:
                    for index,dict in enumerate(elements["google_data_nextpage"]["results"]):
                        if "geometry" in dict:
                            del elements["google_data_nextpage"]["results"][index]["geometry"]
                        if "icon" in dict:
                            del elements["google_data_nextpage"]["results"][index]["icon"]
                        if "id" in dict:
                            del elements["google_data_nextpage"]["results"][index]["id"]
                        if "reference" in dict:
                            del elements["google_data_nextpage"]["results"][index]["reference"]
                        if "photos" in dict:
                            del elements["google_data_nextpage"]["results"][index]["photos"]      
                if "google_data_nextpage2" in elements:
                    for index,dict in enumerate(elements["google_data_nextpage2"]["results"]):
                        if "geometry" in dict:
                            del elements["google_data_nextpage2"]["results"][index]["geometry"]
                        if "icon" in dict:
                            del elements["google_data_nextpage2"]["results"][index]["icon"]
                        if "id" in dict:
                            del elements["google_data_nextpage2"]["results"][index]["id"]
                        if "reference" in dict:
                            del elements["google_data_nextpage2"]["results"][index]["reference"]
                        if "photos" in dict:
                            del elements["google_data_nextpage2"]["results"][index]["photos"]                           
                            
                    
        with open('crawledData_New2.json', 'w') as fout: 
            json.dump(self.json_data, fout, sort_keys=True, indent = 4)   
        
        print("Done")     
                
    def parse_JSON2(self, file):
        
        with open(file, "r" ,encoding='utf8') as jsontext:
            self.json_data = json.load(jsontext)
          
        with open("Mixed_Categories_Remade.json",encoding = 'utf8') as jsoncats:
            categories = json.load(jsoncats)     
        
        with open("items.json","r", encoding = 'utf8') as websites:
            company_websites = json.load(websites)    
        #print(company_websites)
        websites_dict = {}
        for record in company_websites:
            websites_dict[record["name"]] = record["website"]
        #print(categories["WHITE BIRD INVESTMENTS LIMITED"])          
        count = 0   
        names_dict = {} 
        for elements in self.json_data:
            name = elements["company_name"]
            name_for_website = name
            #print(name)
            if elements["google_data"] != "None":
                data = elements["google_data"]["results"]
                if "google_data_nextpage" in elements:
                    data2 = elements["google_data_nextpage"]
                    if "google_data_nextpage2" in elements:
                        data3 = elements["google_data_nextpage2"]
                    else:
                        data3 = {"results":[]}    
                else:
                    data2 = {"results":[]}   
                    data3 = {"results":[]}  
                    
                data.extend(data2["results"])
                data.extend(data3["results"])
                cats = []
                if categories[name] == "Restaurants and Cafes":
                    cats = ["cafe","meal_delivery","meal_takeaway","restaurant", "bar", "night_club", "lodging"] 
                elif categories[name] == "Food and Grocery":      
                    cats = ["convenience_store","grocery_or_supermarket", "gas_station", "bakery","liquor_store"]
                elif categories[name] == "Department_Stores":   
                    cats = ["department_store","shopping_mall","clothing_store", "store"]
                elif categories[name] == "Clothing and Accessories": 
                    cats = ["clothing_store","jewelry_store","shoe_store"]
                elif categories[name] == "Health and Cosmetics":
                    cats = ["beauty_salon","dentist","doctor","hair_care","hospital","pharmacy","spa", "health"]
                elif categories[name] == "Electricals and Electronics": 
                    cats = ["electronics_store","hardware_store"]
                elif categories[name] == "Miscellaneous":      
                    cats = ["bicycle_store","book_store","car_dealer","car_rental","car_repair", "garden centre","car_wash", "art_gallery", "pet_store", "florist", "gas_station"]
                elif categories[name] == "Homeware and DIY":  
                    cats = ["home_goods","furniture_store","home_goods_store", "garden centre"]      
                                
                
                if "amsric" in name.lower().split():
                    name = "StarbucksKFC"
                elif "magawell ltd" in name.lower().split():
                    name = "evans pharmacy"
                elif name.lower() == "a. levy & son limited":
                    name = "Blue Inc." 
                elif name == "DB MODHA LIMITED":
                    name = "Pandora"
                elif name == "MACKAYS STORES LIMITED":
                    name = "M&Co."
                elif name == "WALLACE BOOKS LIMITED":
                    name = "David's Bookshop"
                elif name == "DSG RETAIL LIMITED":
                    name = "Currys PC World"
                elif name == "GREAT LONDON SOUVENIRS LTD":
                    name = "A Little Present"
                elif name == "HALA (SOUTH) LIMITED":
                    name = "Dominos Pizza"                
                elif name == "COLONEL FOODS LIMITED":
                    name = "KFC"
                elif name == "WILCARE CO. LIMITED":
                    name = "Pharmacy"         
                elif name == "MEXICAN GRILL LTD":
                    name = "tortilla"        
                elif name == "APPT CORPORATION LIMITED":
                    name = "Mcdonalds"
                elif name == "LA PLAGE LIMITED":
                    name = "Vilebrequin"
                elif name == "MARSH (BOLTON) LIMITED":
                    name = "Lloyd's pharmacy"
                elif name == "THE FAMOUS ENTERPRISE COCKTAIL COMPANY LIMITED":
                    name = "SPAR Tolley"  
                elif name == "MOLEND LIMITED":
                    name = "Govani"
                elif name == "KENTUCKY FRIED CHICKEN (GREAT BRITAIN) LIMITED":
                    name = "KFC"      
                elif name == "SPORTSWIFT LIMITED":
                    name = "Card Factory"         
                elif name == "CDS (SUPERSTORES INTERNATIONAL) LIMITED":
                    name = "The range" 
                elif name == "GPS (GREAT BRITAIN) LIMITED":
                    name = "Gap Inc"    
                elif name == "BESTWAY NATIONAL CHEMISTS LIMITED":
                    name = "Well Pharmacy"
                elif name == "AZZURRI RESTAURANTS LIMITED":
                    name = "ItalianZizzisCoco di"  
                elif name == "MULTI-TILE LIMITED":
                    name = "Topps tiles" 
                elif name == "BESTWAY PANACEA HEALTHCARE LIMITED":
                    name = "Well Pharmacy"      
                elif name == "HHGL LIMITED":
                    name = "Homebase"
                elif name == "CAVERSHAM TRADING LIMITED":
                    name = "Bright house"       
                elif name == "T. J. MORRIS LIMITED":
                    name = "Home bargains"      
                elif name == "GORGEMEAD LIMITED":
                    name = "Cohens chemist"
                elif name == "YUM! III (UK) LIMITED":
                    name = "Pizza hut delivery"        
                elif name == "TFS STORES LIMITED":
                    name = "The Fragrance Shop"
                elif name == "RYMAN LIMITED":
                    name = "Ryman Stationery"  
                elif name == "EAT LIMITED":
                    name = "EAT. -"      
                elif name == "C-RETAIL LIMITED":
                    name = "Superdry" 
                elif name == "DAVE WHELAN SPORTS LIMITED":
                    name = "DW Sports Fitness"
                elif name == "MOTO HOSPITALITY LIMITED":
                    name = "Moto services"
                elif name == "H&M HENNES & MAURITZ UK LIMITED":
                    name = "H&M"          
                elif name == "GRABAL ALOK (UK) LIMITED":
                    name = "Store twenty one"         
                elif name == "MARTIN RETAIL GROUP LIMITED":
                    name = "MartinsMccolls"
                elif name == "03820011 LTD" or name == "10092014 LIMITED":
                    name = "Garden Store"    
                elif name == "106 (WREXHAM) LTD" or name == "110 (BURY) LTD":
                    name = "The chinese buffet"   
                elif name == "10K RACING LIMITED":
                    name = "Humber runner"     
                elif name == "1453 BAR & GRILL LTD":
                    name = "One beaufort"
                elif name == "149 (BARNARD CASTLE) LIMITED" or name == "149 (BRIDLINGTON) LIMITED":
                    name = "Fish & Chips"        
                elif name == "17TH CENTURY HEALTH FOOD LIMITED":
                    name = "Castle chemist"
                elif name == "1906 LTD":
                    name = "Hanbury chinese"    
                elif name == "1994 INC LTD":
                    name = "Supreme Newyork"
                elif name == "2014 COFFEE HOUSE LTD":
                    name = "Truth"
                elif name == "21STCENTURYRETRO LIMITED":
                    name = "Blue Vintage"
                elif name == "23.5 DEGREES LIMITED":
                    name = "Starbucks coffee"
                elif name == "28-50 LIMITED":
                    name = "28-50 Wine"                        
                elif name == "3 INVOLVED LIMITED":
                    name = "Anchor lane"
                elif name == "3D&CO LIMITED":
                    name = "outwoods pharmacy"
                elif name == "40/46 THE SIDE NEWCASTLE LIMITED":
                    name = "prima " 
                elif name == "WM MORRISON SUPERMARKETS P L C":
                    name = "WM Morrisons "
                elif name == "MALLETT & SON  LIMITED":
                    name = "mallettshomehardware"               
                    
                if name.lower() == "PUNCH PARTNERSHIPS (PML) LIMITED".lower() or name.lower() == "MITCHELLS & BUTLERS RETAIL LIMITED".lower() or name.lower() == "MARSTON'S PUBS LIMITED".lower() or name == "ADMIRAL TAVERNS LIMITED" or name == "SPIRIT PUB COMPANY (MANAGED) LIMITED" or name == "ORCHID PUBS & DINING LIMITED" or name == "THE RESTAURANT GROUP (UK) LIMITED" or name == "DICKINSON BROTHERS LIMITED" or name == "WHITBREAD GROUP PLC" or name == "TJX UK" or name == "HALL & WOODHOUSE LIMITED" or name == "GREENE KING BREWING AND RETAILING LIMITED" or name == "CO-OPERATIVE GROUP LIMITED" or name == "NEXT HOLDINGS LIMITED":    
                    matches = [x["name"].lower() for x in data if any(cat in x["types"] for cat in cats)]
                else:
                    matches = []
                    # TEST FOR TOMMOROW, TRY TO REMOVE THE CATEGORIES AND SEE THE INCREASE AFTER THE REGEX         
                    x1 = re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",name.lower())
                    #x2 = re.sub("\(.*\)|limited|ltd","",x["name"].lower())
                    close1 = get_close_matches(x1,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data], n=5, cutoff=0.8)
                    close2 = get_close_matches(x1,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data], n=5, cutoff=0.6)
                    close3 = get_close_matches(x1,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data], n=5, cutoff=0.5)
                    if file == "JSON_data_merged.json":
                        # This first if is to say that if the similarity is very high between the names then no need for checking type
                        if len(close1) > 0:
                            matches = close1 
                        elif len(close2) > 0: 
                            matches = close2  
                        elif len(close3) > 0:
                            matches = close3          
                        elif len(get_close_matches(x1,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data if any(cat in x["types"] for cat in cats)], n=5, cutoff=0.4)) > 0:                                
                            matches = get_close_matches(x1,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data if any(cat in x["types"] for cat in cats)], n=5, cutoff=0.4)
                            #print("Found a 0.4 first match for it",x1, " : ", matches)
                        #else:
                        #    matches = [x["name"].lower() for x in data if any(cat in x["types"] for cat in cats)]  
                        else:
                            if name_for_website in websites_dict:
                                #print(name)    
                                website = websites_dict[name_for_website]
                                if website:  
                                    if "www" in website:
                                        website2 = "".join(re.findall("\.([A-Za-z0-9-]+)\.", website.lower()))     
                                    else:
                                        website2 = "".join(re.findall("/([A-Za-z0-9-]+)\.", website.lower()))    
                                    #print("I am using the website name ",website2, "for ", name_for_website)
                                    
                                    if len(get_close_matches(website2,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data], n=5, cutoff=0.8)) > 0:
                                        matches = get_close_matches(website2,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data], n=5, cutoff=0.8) 
                                        #print("Found a 0.8 match for it",website2, " : ", matches)
                                    elif len(get_close_matches(website2,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data], n=5, cutoff=0.6)) > 0: 
                                        matches = get_close_matches(website2,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data], n=5, cutoff=0.6)  
                                        #print("Found a 0.6 match for it",website2, " : ", matches)  
                                    elif len(get_close_matches(website2,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data], n=5, cutoff=0.5)) > 0: 
                                        matches = get_close_matches(website2,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data], n=5, cutoff=0.5)  
                                        #print("Found a 0.6 match for it",website2, " : ", matches)    
                                    elif len(get_close_matches(website2,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data if any(cat in x["types"] for cat in cats)], n=5, cutoff=0.4)) > 0:                                
                                        matches = get_close_matches(website2,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data if any(cat in x["types"] for cat in cats)], n=5, cutoff=0.4)
                                        #print("Found a 0.4 match for it",website2, " : ", matches)
                                    #else:
                                    #    matches = [x["name"].lower() for x in data if any(cat in x["types"] for cat in cats)]
                                            
                                #else:
                                #    matches = [x["name"].lower() for x in data if any(cat in x["types"] for cat in cats)]
                            #else:            
                            #    matches = [x["name"].lower() for x in data if any(cat in x["types"] for cat in cats)]    
                    else:
                        # This first if is to say that if the similarity is very high between the names then no need for checking type
                        if len(get_close_matches(x1,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data], n=5, cutoff=0.6)) > 0:
                            matches = get_close_matches(x1,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data], n=5, cutoff=0.6) 
                        elif len(get_close_matches(x1,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data if any(cat in x["types"] for cat in cats)], n=5, cutoff=0.4)) > 0:                                
                            matches = get_close_matches(x1,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data if any(cat in x["types"] for cat in cats)], n=5, cutoff=0.4)
                        else:
                            matches = [x["name"].lower() for x in data if any(cat in x["types"] for cat in cats)]
                            #print("Found a type match for it",x1, " : ", matches)
                            
                        if name_for_website in websites_dict:
                                #print(name)    
                                website = websites_dict[name_for_website]
                                if website:  
                                    if "www" in website:
                                        website2 = "".join(re.findall("\.([A-Za-z0-9-]+)\.", website.lower()))     
                                    else:
                                        website2 = "".join(re.findall("/([A-Za-z0-9-]+)\.", website.lower()))    
                                    #print("I am using the website name ",website2, "for ", name_for_website)
                                    if len(get_close_matches(website2,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data], n=5, cutoff=0.6)) > 0: 
                                        matches = get_close_matches(website2,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data], n=5, cutoff=0.6)  
                                        #print("Found a 0.6 match for it",website2, " : ", matches)  
                                    elif len(get_close_matches(website2,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data], n=5, cutoff=0.5)) > 0: 
                                        matches = get_close_matches(website2,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data], n=5, cutoff=0.5)  
                                        #print("Found a 0.6 match for it",website2, " : ", matches)    
                                    elif len(get_close_matches(website2,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data if any(cat in x["types"] for cat in cats)], n=5, cutoff=0.4)) > 0:                                
                                        matches = get_close_matches(website2,[re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",x["name"].lower()) for x in data if any(cat in x["types"] for cat in cats)], n=5, cutoff=0.4)
                                        #print("Found a 0.4 match for it",website2, " : ", matches)
                        
                if len(matches) != 0: 
                    #coordinates = ''.join(str(elements["Coordinates"]))
                    #while(coordinates in names_dict):
                    #    elements["Coordinates"][0] = np.nextafter(elements["Coordinates"][0],1)
                    #    elements["Coordinates"][1] = np.nextafter(elements["Coordinates"][1],1)
                    #    coordinates = ''.join(str(elements["Coordinates"]))
                    count = count + 1
                    names_dict[elements["title_number"]] = (matches,elements["company_name"])    
                    
                
        print(count)
        print(len(self.json_data)) 
        counts = Counter([v[1] for k,v in names_dict.items()])
        sorted_counts = sorted(counts.items(), key = operator.itemgetter(1))
        if file == "JSON_data_merged.json":
            with open('countsss_new2.txt', 'w') as fout: 
                for i in sorted_counts:
                    #print(i)
                    fout.write(str(i) + '\n')
                 
            with open('hobescalob_new2.json', 'w') as fout: 
                # CHECK IT!!!! INDENT AND SEPERATOR WHAT DO THEY REPRESENT!!!
                json.dump(names_dict, fout)
        
        else:
            with open('countsss2_new2.txt', 'w') as fout: 
                for i in sorted_counts:
                    #print(i)
                    fout.write(str(i) + '\n')
                 
            with open('hobescalob2_new2.json', 'w') as fout: 
                # CHECK IT!!!! INDENT AND SEPERATOR WHAT DO THEY REPRESENT!!!
                json.dump(names_dict, fout)
                    
                
        return names_dict      
         
    
    def calcMissing(self, found):
        count = 0
        missing_Data = []
        print(len(self.csv_file))
        titles = [dicts["title_number"] for dicts in found]
        for index,dicts in enumerate(self.csv_file):
            #lat = dicts["Coordinates"].split(",")[0].replace("(","[")
            #lon = dicts["Coordinates"].split(",")[1].replace(")","]")
            # Did that to get rid of the scientific notation of the longitudes
            #latitude = float(dicts["latitude"])
            #longitude = float(dicts["longitude"])
            #latitude = str(latitude)
            #longitude = str(longitude)
            #coordinates = "["+ latitude + ", " + longitude + "]"
            #print(coordinates)
            #coordinates = ','.join([lat,lon])
            if dicts["title_number"] not in titles:
                missing_Data.append(self.csv_file[index])
                count = count + 1
                
        print(count)            
        print(len(missing_Data))
        with open('missingData_Final2.json', 'w') as fout: 
            json.dump(missing_Data, fout)
        '''
        count2 = 0  
        count3 = 0  
        for elements in self.json_data:
            count3 += 1
            coordinates = ''.join(str(elements["Coordinates"]))
            if coordinates in found:
                count2 += 1
                
        print(count2)
        print(count3)
        '''        
        '''    
        # Converting the coordinates of the missing data into offset coordinates    
        for index,dicts in enumerate(missing_Data):
            lat = dicts["Coordinates"].split(",")[0].replace("(","")
            lon = dicts["Coordinates"].split(",")[1].replace(")","")
            latlon = self.calcOffset(float(lat),float(lon))
            missing_Data[index]["Coordinates"] = latlon  
            
        with open('missingData_offset.json', 'w') as fout: 
            json.dump(missing_Data, fout) 
        '''    
        return missing_Data    
            
    def calcOffset(self,lat,lon):

        #Earth radius, sphere
        R=6378137
        
        #offsets in meters
        dn = 100
        de = 100
        
        #Coordinate offsets in radians
        dLat = dn/R
        dLon = de/(R*math.cos(math.pi*lat/180))
        
        #OffsetPosition, decimal degrees
        latO = lat + dLat * 180/math.pi
        lonO = lon + dLon * 180/math.pi  
        return str((latO,lonO))            
        
        
    def calculateDistance(self, listofdicts):
        found_Data = []
        for index,dicts in enumerate(self.csv_file):
            lat = dicts["Coordinates"].split(",")[0].replace("(","[")
            lon = dicts["Coordinates"].split(",")[1].replace(")","]")
            coordinates = ','.join([lat,lon])
            if coordinates in listofdicts:
                found_Data.append(self.csv_file[index])
                
        count = 0
        for index,dicts in enumerate(found_Data):
            
            if count == 20:
                break
            lat = dicts["Coordinates"].split(",")[0].replace("(","")
            lon = dicts["Coordinates"].split(",")[1].replace(")","")
            coordinates = (float(lat),float(lon))
            for index2,dicts2 in enumerate(found_Data):
                lat2 = dicts2["Coordinates"].split(",")[0].replace("(","")
                lon2 = dicts2["Coordinates"].split(",")[1].replace(")","")
                coordinates2 = (float(lat2),float(lon2))
                distance = vincenty(coordinates,coordinates2).meters
                if distance < 400:
                    print(coordinates," ",coordinates2," ",distance, " " ,dicts["company_name"], " ", dicts2["company_name"])
            count +=1
            
    def completeData(self,found1,found2):
        complete_dict = {**found1, **found2}
        self.final_Data = []
        count = 0
        for index,dicts in enumerate(self.csv_file):
            #lat = dicts["Coordinates"].split(",")[0].replace("(","[")
            #lon = dicts["Coordinates"].split(",")[1].replace(")","]")
            #coordinates = ','.join([lat,lon])
            if dicts["title_number"] in complete_dict:
                self.final_Data.append(self.csv_file[index])
                count = count + 1
                
        print(count)            
        
        print(len(complete_dict))
        
        with open('finalData3.json', 'w') as fout: 
            json.dump(self.final_Data, fout)
            
        return self.final_Data
                    
    def reverse_geocodeGoogle(self,latlon):
        return [components["long_name"] for geocodes in self.x.gmaps.reverse_geocode(latlon) for components in geocodes["address_components"] if "postal_code" in components["types"]]
    
    def generate_missingPostcodes(self):
        '''
        with open("finalData.json", "r" ,encoding='utf8') as jsontext:
            final_Data = json.load(jsontext)
        
            
        missingPostcodes_dict = {}
        for dicts in final_Data:
            if dicts["postcode"] == "":
                lat = dicts["Coordinates"].split(",")[0].replace("(","")
                lon = dicts["Coordinates"].split(",")[1].replace(")","")
                coordinates = (float(lat),float(lon))
                missingPostcodes_dict[dicts["Coordinates"]] = self.reverse_geocodeGoogle(coordinates)
                #print(dicts["Coordinates"], missingPostcodes_dict[dicts["Coordinates"]])
        
        #with open("missing_Postcodes.json","w") as fout:
        #    json.dump(missingPostcodes_dict, fout) 
        '''
        
        csv_file = []
        # The data structure used here is a list of dicts where each dict is a row in the CSV file with header as key and attribute as value
        with open("missingPostcodes2.csv",encoding='utf8') as sqldbCSV:
            csvread = csv.reader(sqldbCSV, delimiter=",")
            count = -1
            for line in csvread:
                #print(line)
                count = count + 1
                # Ignoring the header row
                
                if count == 0 or count==1:
                    continue
                
                my_dict = {}
                #my_dict["ID"] = line[0]
                my_dict["Coordinates"] = line[12]
                
                csv_file.append(my_dict)
        
            
        missingPostcodes_dict = {}
        for dicts in csv_file:
            
            lat = dicts["Coordinates"].split(",")[0].replace("(","")
            lon = dicts["Coordinates"].split(",")[1].replace(")","")
            coordinates = (float(lat),float(lon))
            missingPostcodes_dict[dicts["Coordinates"]] = self.reverse_geocodeGoogle(coordinates)
            #print(dicts["Coordinates"], missingPostcodes_dict[dicts["Coordinates"]])
        
        
        with open("missing_Postcodes2.json","w") as fout:
            json.dump(missingPostcodes_dict, fout)     
        
    def check_missingPostcodes(self):
        
        postcodes = []
        with open("C:/Users/ahmed/Downloads/PCD11_OA11_LSOA11_MSOA11_LAD11_EW_LU_aligned_v2/PCD11_OA11_LSOA11_MSOA11_LAD11_EW_LU_aligned_v2.csv",encoding='utf8') as sqldbCSV:
            csvread = csv.reader(sqldbCSV, delimiter=",")
            count = -1
            for line in csvread:
                print(count)
                count = count + 1
                # Ignoring the header row
                if count == 0 or not line:
                    continue
                
                code = line[0].replace(" ","")
                postcodes.append(code)
                
        with open("missing_Postcodes2.json", "r" ,encoding='utf8') as jsontext:
            missing_Data = json.load(jsontext)
            
        count = 0
        missing_Data_Modified = {}
        for k,v in missing_Data.items():
            for post in v:
                if post.replace(" ","") in postcodes:
                    count += 1
                    missing_Data_Modified[k] = post.replace(" ","")
                    #print("Yes")
                    break
        print(count) 
        
        with open("missing_Data_Modified2.json","w") as fout:
            json.dump(missing_Data_Modified, fout)  
        
    def add_missingPostcodes(self):
                         
        with open("missing_Data_Modified.json", "r" ,encoding='utf8') as jsontext:
            missing_postcodes = json.load(jsontext)        
                     
        with open("missing_Data_Modified2.json", "r" ,encoding='utf8') as jsontext:
            missing_postcodes2 = json.load(jsontext)
            
        for index,dicts in enumerate(self.final_Data):
            if dicts["postcode"] == "":
                if dicts["Coordinates"] in missing_postcodes:
                    self.final_Data[index]["postcode"] = missing_postcodes[dicts["Coordinates"]]
                    
        for index,dicts in enumerate(self.final_Data):
            if dicts["Coordinates"] in missing_postcodes2:
                self.final_Data[index]["postcode"] = missing_postcodes2[dicts["Coordinates"]]            
                
        with open('finalData2.json', 'w') as fout: 
            json.dump(self.final_Data, fout) 
        
        f = csv.writer(open("finalData_New2.csv", "w", encoding = 'utf8', newline=''))     
        f.writerow(["street_name", "town_name", "postcode", "company_number", "ownership_key", "tenure", "date", "company_name", "title_number", "SIC", "Polygon", "Category", "Coordinates"])          
        for dicts in self.final_Data:
            #lat = dicts["Coordinates"].split(",")[0].replace("(","")
            #lon = dicts["Coordinates"].split(",")[1].replace(")","")
            f.writerow([dicts["street_name"], dicts["town_name"], dicts["postcode"], dicts["company_number"], dicts["ownership_key"], dicts["tenure"], dicts["date"], dicts["company_name"], dicts["title_number"], dicts["SIC"], dicts["Polygon"], dicts["Category"], dicts["Coordinates"]])
                    
    def missingData_FromstreetNames(self):
        missingcsv_file = []
        # The data structure used here is a list of dicts where each dict is a row in the CSV file with header as key and attribute as value
        with open("missingPostcodes_Final.csv",encoding='utf8') as sqldbCSV:
            csvread = csv.reader(sqldbCSV, delimiter=",")
            count = -1
            for line in csvread:
                #print(line)
                count = count + 1
                # Ignoring the header row
                
                if count == 0 or count==1:
                    continue
                
                my_dict = {}
                #my_dict["ID"] = line[0]
                my_dict["street_name"] = line[0]
                my_dict["town_name"] = line[1]
                my_dict["postcode"] = line[2]
                my_dict["company_number"] = line[3]
                my_dict["ownership_key"] = line[4]
                my_dict["tenure"] = line[5]
                my_dict["date"] = line[6]
                my_dict["company_name"] = line[7]
                my_dict["title_number"] = line[8]
                my_dict["SIC"] = line[9]
                my_dict["Polygon"] = line[10]
                my_dict["Category"] = line[11]
                my_dict["Coordinates"] = line[12]
                
                
                
                missingcsv_file.append(my_dict)  
                
                
        finalcsv_file = []
        # The data structure used here is a list of dicts where each dict is a row in the CSV file with header as key and attribute as value
        with open("final_Data_with_OA.csv",encoding='utf8') as sqldbCSV:
            csvread = csv.reader(sqldbCSV, delimiter=",")
            count = -1
            for line in csvread:
                #print(line)
                count = count + 1
                # Ignoring the header row
                
                if count == 0 or count==1:
                    continue
                
                my_dict = {}
                #my_dict["ID"] = line[0]
                my_dict["street_name"] = line[0]
                my_dict["town_name"] = line[1]
                my_dict["postcode"] = line[2]
                my_dict["company_number"] = line[3]
                my_dict["ownership_key"] = line[4]
                my_dict["tenure"] = line[5]
                my_dict["date"] = line[6]
                my_dict["company_name"] = line[7]
                my_dict["title_number"] = line[8]
                my_dict["SIC"] = line[9]
                my_dict["Polygon"] = line[10]
                my_dict["Category"] = line[11]
                my_dict["Coordinates"] = line[12]
                my_dict["OA"] = line[13]
                my_dict["LSOA"] = line[14]
                my_dict["MSOA"] = line[15]
                
                
                finalcsv_file.append(my_dict)  
            count = 0    
            missing_postcodes2 = {}
            for index,dicts in enumerate(missingcsv_file):
                for dicts2 in finalcsv_file:
                    if dicts["street_name"] == dicts2["street_name"] and dicts["town_name"] == dicts2["town_name"]:
                        missing_postcodes2[dicts["Coordinates"]] = dicts2["postcode"] 
                        missingcsv_file[index]["postcode"] = dicts2["postcode"]
                        count = count + 1
            print(count)
            count = 0
            for index,dicts in enumerate(self.final_Data):
                if dicts["Coordinates"] in missing_postcodes2:
                    self.final_Data[index]["postcode"] = missing_postcodes2[dicts["Coordinates"]]
                    count = count + 1
            print(count)
            
            with open('finalData3.json', 'w') as fout: 
                json.dump(self.final_Data, fout) 
            '''    
            f = csv.writer(open("finalData_New3.csv", "w", encoding = 'utf8', newline=''))     
            f.writerow(["street_name", "town_name", "postcode", "company_number", "ownership_key", "tenure", "date", "company_name", "title_number", "SIC", "Polygon", "Category", "Coordinates"])          
            for dicts in self.final_Data:
                #lat = dicts["Coordinates"].split(",")[0].replace("(","")
                #lon = dicts["Coordinates"].split(",")[1].replace(")","")
                f.writerow([dicts["street_name"], dicts["town_name"], dicts["postcode"], dicts["company_number"], dicts["ownership_key"], dicts["tenure"], dicts["date"], dicts["company_name"], dicts["title_number"], dicts["SIC"], dicts["Polygon"], dicts["Category"], dicts["Coordinates"]])
            '''            
    
    def most_common(self,L):
        # get an iterable of (item, iterable) pairs
        SL = sorted((x, i) for i, x in enumerate(L))
        # print 'SL:', SL
        groups = itertools.groupby(SL, key=operator.itemgetter(0))
        # auxiliary function to get "quality" for an item
        def _auxfun(g):
            item, iterable = g
            count = 0
            min_index = len(L)
            for _, where in iterable:
                count += 1
                min_index = min(min_index, where)
                # print 'item %r, count %r, minind %r' % (item, count, min_index)
            return count, -min_index
        # pick the highest-count/earliest item
        return max(groups, key=_auxfun)[0]                               
                
    def mergeCategories(self, data):
        
        print(len(data))
        with open("Mixed_Categories_Remade.json",encoding = 'utf8') as jsoncats:
            categories = json.load(jsoncats)
            
        for index,dicts in enumerate(data):
            name = dicts["company_name"]
            if name in categories:
                data[index]["category"] = categories[name]
            
            
        print(len(data))
        
        f = csv.writer(open("finalDataV2_New_Final3.csv", "w", encoding = 'utf8', newline=''))     
        f.writerow(["street_name", "town_name", "postcode", "company_number", "tenure", "date", "company_name", "title_number", "SIC", "category", "latitude", "longitude"])          
        for dicts in data:    
            f.writerow([dicts["street_name"], dicts["town_name"], dicts["postcode"], dicts["company_number"], dicts["tenure"], dicts["date"], dicts["company_name"], dicts["title_number"], dicts["SIC"], dicts["category"], dicts["latitude"], dicts["longitude"]])
                 
    def hobescalob(self,file1,file2):
        
        with open(file1, "r" ,encoding='utf8') as jsontext:
            Main_data = json.load(jsontext)
            
        with open(file2, "r" ,encoding='utf8') as jsontext:
            New_data = json.load(jsontext)
            
        titles_new = set([dicts["title_number"] for dicts in New_data])
        titles_main = set([dicts["title_number"] for dicts in Main_data])
        count = 0
        duplicates = []
        for title in titles_new:
            if title in titles_main:
                print(title)
                count +=1 
                duplicates.append(title)       
        print(count)   
        print(len(duplicates))
        for index, dicts in enumerate(New_data):
            if dicts["title_number"] in duplicates:
                del New_data[index]      
            else:
                Main_data.append(dicts)       
        
        with open('crawledData_New2.json', 'w') as fout: 
            json.dump(New_data, fout, sort_keys=True, indent = 4)
        
        with open('JSON_data_merged.json', "w") as fout:
            json.dump(Main_data, fout, sort_keys=True, indent = 4)    
            
        
                              
x = geoCoder()
#x.hobescalob("JSON_data_merged.json", "crawledData_New2.json")
#x.crawl_google_extra()
#x.modify_JSONsize("crawledData_New.json")
#x.mergeJSON("crawledData_Modified.json", "crawledData_Offset.json")
#x.normalize_Companyname()
#x.test_json("crawledData_Modified.json")
#########x.categorize_Google("JSON_data_merged.json")
#x.crawl_google()
names_dict = x.parse_JSON2("JSON_data_merged.json")
#x.parse_JSON2("crawledData_Offset.json")
##missing_data = x.calcMissing(names_dict)
#x.crawl_google("missingData2.json")
names_dict2 = x.parse_JSON2("crawledData_Offset.json")
#missing_data = x.calcMissing(names_dict2)
complete_data = x.completeData(names_dict, names_dict2)
x.mergeCategories(complete_data)

missing_data = x.calcMissing(complete_data)
missing_companies = [dicts["company_name"] for dicts in missing_data]
counts = Counter(missing_companies)
print(sorted(counts.items(), key = operator.itemgetter(1), reverse = True))
print(x.most_common(missing_companies))

