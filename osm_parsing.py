# -*- coding: utf-8 -*-

import json
import re
import csv
from collections import Counter
import Levenshtein


class osm_parsing:
    def __init__(self):
        # Import the OSM Data which I got from overpass turbo and it contains all shops and amenities in the UK registered in OSM
        with open("osm_data_extended_final.json", "r" ,encoding='utf8') as jsontext:
            self.osm_data = json.load(jsontext)
            
        # The below code was used to clean the osm_data file to make it smaller and for faster indexing    
        '''    
            for elements in self.osm_data["elements"]:
                if "id" in elements:
                    del elements["id"]
                if "lat" in elements:
                    del elements["lat"]
                if "lon" in elements:
                    del elements["lon"]
                if "tags" in elements:
                    if "addr:city" in elements["tags"]:
                        del elements["tags"]["addr:city"]    
                    if "addr:housenumber" in elements["tags"]:
                        del elements["tags"]["addr:housenumber"] 
                    if "addr:postcode" in elements["tags"]:
                        del elements["tags"]["addr:postcode"] 
                    if "addr:street" in elements["tags"]:
                        del elements["tags"]["addr:street"] 
                    if "opening_hours" in elements["tags"]:
                        del elements["tags"]["opening_hours"] 
                    if "opening_hours:url" in elements["tags"]:
                        del elements["tags"]["opening_hours:url"]                     
                            
        
        with open('osm_data_extended_final.json', 'w') as fout: 
            json.dump(self.osm_data, fout, sort_keys=True, indent = 1)   
        '''        
        #print(self.osm_data["elements"][0]["tags"]["addr:postcode"])    
        
        # Import the CSV file which contains all the Retail, fastfood, and restaurant companies based on their SIC Code. This list is obtained from SQLDB
        self.csv_file = []
        # The data structure used here is a list of dicts where each dict is a row in the CSV file with header as key and attribute as value
        with open("C:/Users/ahmed/OneDrive/Documents/Masters Courses/Dissertation/SQL Databases/OwnershipCoordinatesFinal2.csv",encoding='utf8') as sqldbCSV:
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
                
        print(len(self.csv_file))    
          
        '''
        # Importing the top 50 retail companies by the number of properties they bought over the past 2 years. Got it from SQLDB
        self.topRetails = []
        with open("C:/Users/ahmed/OneDrive/Documents/Masters Courses/Dissertation/SQL Databases/Top50_Retails_Bycount.csv",encoding='utf8') as top50:
            csvread = csv.reader(top50, delimiter=",")
            count = -1
            for line in csvread:
                count = count + 1
                if count == 0:
                    continue
                self.topRetails.append(line[1])
        '''
    # This function returns the OSM data
    def get_OSMData(self):
        return self.osm_data
        
    # Function to get the companies data by quering the BetaCompanyHouse company list which is the official names of the companies
    def get_CompaniesData2(self):
        # This function is used to normalize the company names into a concise format which is used by BetaCompanyHouse and Office of National Statistics
        companiesData = []
        official_data = {}
        companiesDict = {}
        '''
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
                            
                            official_data[line[1]] = line[0]
        
        print(len(official_data)) 
        '''
        # Get only the unique company numbers no duplicates from the csv_file
        numbers = set([dicts["company_number"] for dicts in self.csv_file])  
        # Check if this company is present in official_data which is the BetaCompanyHouse list. if so then add it to the companiesData using the number as key and company name as value              
        for dicts in self.csv_file:
            companiesDict[dicts["company_number"]] = dicts["company_name"]
        
        #print([(key,values) for key, values in companiesDict.items() if len(set(values)) > 1])     
        #print(len(companiesDict.keys()))    
        
        for number in numbers:
            companiesData.append(companiesDict[number]) 
        
        print(len(companiesData)) 
        
        '''
        new_dict = {}
        for k, v in companiesDict.items():
            new_dict.setdefault(v, []).append(k)  
        print([(key,values) for key, values in new_dict.items() if len(values) > 1])     
        '''  
        return companiesData      
    
    
    # This function returns the topRetails companies          
    def get_TopCompanies(self):
        return self.topRetails
     
    # Depreciated function to compare the postcodes in the OSM data with the post codes in the csv file to know which ones are actual shops and which were bought for investment           
    def compare(self):
        count = 0
        my_postcodes = []
        for postcodes in self.osm_data["elements"]:
            if "addr:postcode" in postcodes["tags"]:
                my_postcodes.append((postcodes["tags"]['addr:postcode']).replace(" ", ""))
        for dicts in self.csv_file:
            if dicts["postcode"] in my_postcodes:
                #print("I found {} with a title number of {}".format(dicts["company_name"],dicts["title_number"]))
                count  = count +1
        
        print(count)    
    
    # This function is used to parse all the shop names in the OSM data whether they are in "name", "operator", or "brand" and puts them in shop_names then in a dict for faster lookup        
    def parsing_OSM(self):
        shops_names = []
        types = ["butcher_&_convinience","bar", "general_stores","bbq","frozen_food","computer_games","tiles","household_goods","clothing","Moss Bros.","catalogue_store", "fast_food;restaurant","electrical_supplies","cafe;restaurant","general_store","specialist_food","coffeeshop","housewares","pub; restaurant","electronics_&_electrical","grocer","chemists","Home Store","Sports_Store","videogame","homewares","electrics","opticians","health_and_beauty","footwear","healthfood","carpets","Handbags and Leather Goods","tile_shop","butcher;greengrocer","furniture;garden_centre","discount_store","fancy-dress","cafe", "furnishing","fast_food", "jewellery","mobile_phone_accessory","restaurant;pub;bar;fast_food","computer_parts","outdoor; doityourself; garden_centre", "household_appliances" ,"household,discount", "Wine bar and deli", "fast_food;cafe","former_jewelry", "shopping_centre","chinese_takeaway","heathfood", "electronics;electrical;radiotechnics", "handbags", "cheesemonger", "bedding","frozen_foods","electrical;white_goods;repair","Discount_Store","curtain;interior_decoration","homeware","Indoor_Market","fruiterer","greetings cards","Childrenswear","brazilian_store","lingerie", "groceries","technology", "convenience;newsagent","newsagent;toy", "windows;doors","deli;cafe","wholefood","auto", "canteen", "wine;cheese","convenience;alcohol","luxury_Toiletries", "decorating", "electronical","factory_outlet","beauty;hairdresser","electrical_appliance", "fixme", "discounter", "hair_accessories","greeting cards", "fishing;clothing", "Bedding,_net_curtains_&_curtain_shop", "map;gift","nightclub", "seafood,butcher,grocer", "floors","interior_decoration","tile", "fish","ceramics","game", "convenience;Polish","cookies","hometextile", "tobacconist", "cheese;wine","hair_products","pub", "restaurant", "ice_cream", "food_court", "alcohol","bakery","beverages", "fishmonger", "butcher","coffee","confectionery", "deli", "dairy", "farm", "greengrocer", "pastry", "seafood", "spices", "tea", "general", "kiosk", "mall", "supermarket", "convenience", "grocery", "second_hand", "variety_store", "discount", "charity", "baby_goods", "accessories", "bag","wedding" , "boutique", "clothes", "fabric", "fasion", "jewlery", "jewelry", "leather", "shoes", "tailor", "watches", "department_store", "catalogue", "beauty", "chemist", "cosmetics", "erotic", "hairdresser", "health_food", "herbalist", "medical_supplies", "perfumery", "optician", "nutrition_supplies", "pharmacy", "doityourself", "electrical", "energy", "fireplace","florist", "garden_centre", "hardware", "houseware", "security", "trade", "anitques", "bed", "candles", "carpet", "curtains", "furniture", "decoration", "kitchen", "household", "computer", "electronics", "newsagent", "hunting", "hifi", "mobile_phones", "radiotechnics", "vacuum_cleaner", "art", "collector", "craft", "games", "music", "cameras", "video", "video_games", "bicycle", "car", "car_repair", "car_parts", "fuel", "sports", "outdoor", "anime", "books", "gift", "lottery", "stationary", "ticket", "car_rental"]
        notypes = []
        for nodes in self.osm_data["elements"]:
            if "name" in nodes["tags"] and "shop" in nodes["tags"]:
                if nodes["tags"]["shop"] in types:
                    shops_names.append(((nodes["tags"]["name"]).lower(),nodes["tags"]["shop"]))
                else:
                    notypes.append(nodes["tags"]["shop"])    
            
            elif "name" in nodes["tags"] and "amenity" in nodes["tags"]:
                if nodes["tags"]["amenity"] in types:
                    shops_names.append(((nodes["tags"]["name"]).lower(),nodes["tags"]["amenity"]))
                else:
                    notypes.append(nodes["tags"]["amenity"])
            
            elif "operator" in nodes["tags"] and "shop" in nodes["tags"]:
                if nodes["tags"]["shop"] in types:
                    shops_names.append(((nodes["tags"]["operator"]).lower(),nodes["tags"]["shop"]))
                else:
                    notypes.append(nodes["tags"]["shop"])
            
            elif "operator" in nodes["tags"] and "amenity" in nodes["tags"]:
                if nodes["tags"]["amenity"] in types:
                    shops_names.append(((nodes["tags"]["operator"]).lower(),nodes["tags"]["amenity"])) 
                else:
                    notypes.append(nodes["tags"]["amenity"])       

            elif "brand" in nodes["tags"] and "shop" in nodes["tags"]:
                if nodes["tags"]["shop"] in types:
                    shops_names.append(((nodes["tags"]["brand"]).lower(),nodes["tags"]["shop"]))
                else:
                    notypes.append(nodes["tags"]["shop"])    
            
            elif "brand" in nodes["tags"] and "amenity" in nodes["tags"]:
                if nodes["tags"]["amenity"] in types:
                    shops_names.append(((nodes["tags"]["brand"]).lower(),nodes["tags"]["amenity"])) 
                else:
                    notypes.append(nodes["tags"]["amenity"])     
        
        self.dct = {}
        for name, type in shops_names:
            if name in self.dct:
                self.dct[name].extend([type])
            else:
                self.dct[name] = [type] 
                                
        #self.dct = dict(shops_names) 
        
        #self.dct = OrderedDict(sorted(self.dct.items(), key=lambda t: t[0]))   
        #for k in self.dct.keys():
        #    data = Counter(self.dct[k])
        #    self.dct[k]= [data.most_common(1)[0][0]]   
            #print(k, self.dct[k]) 

        
        print(len(self.dct.keys()))
    # Function to categorize each retail store in the dataset parameter which is usually the output from get_companiesData2      
    def categorize_OSM3(self, dataset):
        for k in self.dct.keys():
            #data = Counter(self.dct[k])
            #self.dct[k]= [data.most_common(1)[0][0]]   
            print(k, self.dct[k]) 
        
        with open("items.json","r", encoding = 'utf8') as websites:
            company_websites = json.load(websites)    
        #print(company_websites)
        websites_dict = {}
        for record in company_websites:
            websites_dict[record["name"]] = record["website"]
            
        count = 0
        types_dict = {}
        # Loop through every retail store in the dataset (which is the output from get_companiesData2)
        for retail in ["DEBENHAMS RETAIL"]:
            print("Categorizing ", retail)
            # This line is because of a memory leak I had with the RESTAURANT companies for some reason they all pointed to the same list and it kept getting bigger
            types_dict[retail] = []
            #x1 = re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",name2.lower())
            #retail = retail.lower()
            retail2 = re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings|plc","",retail.lower())
            website2 = ""
            if retail in websites_dict:
                #print(name)    
                website = websites_dict[retail]
                if website:  
                    if "www" in website:
                        website2 = "".join(re.findall("\.([A-Za-z0-9-]+)\.", website.lower()))     
                    else:
                        website2 = "".join(re.findall("/([A-Za-z0-9-]+)\.", website.lower()))    
                        #print(website2)
            # Build the regex
            #matcher = re.compile(retail.lower())
            # Loop through each element in the shops dataset from parseOSM() and look for my retail store name
            for names in self.dct:
                names2 = re.sub("\(.*\)|limited|ltd|u0.*\s|HOLDINGS|holdings","",names.lower())
                #print(names)
                # If it matches then append the company name to a temp list
                if Levenshtein.ratio(retail2,names2) >= 0.8:
                    #print(retail," :: " ,names," :: " ,Levenshtein.ratio(retail2,names2))
                    if retail in types_dict:
                        #print(x["name"])
                        #print(x["types"])
                        #print(len(types_dict[name]), name)
                        types_dict[retail].extend(self.dct[names])
                    else:
                        types_dict[retail] = self.dct[names] 
                
                elif Levenshtein.ratio(retail2,names2) >= 0.7:
                    #print(retail," :: " ,names," :: " ,Levenshtein.ratio(retail2,names2))
                    if retail in types_dict:
                        #print(x["name"])
                        #print(x["types"])
                        #print(len(types_dict[name]), name)
                        types_dict[retail].extend(self.dct[names])
                    else:
                        types_dict[retail] = self.dct[names]           
                     
                    #Found.append(retail)
                    #temp_category.append(self.dct[names])
                    
                else:
                    if website2:
                        if Levenshtein.ratio(website2,names2) >= 0.8:
                            #print(website2," :: " ,names," :: " ,Levenshtein.ratio(website2,names2))
                            if retail in types_dict:
                                #print(x["name"])
                                #print(x["types"])
                                #print(len(types_dict[name]), name)
                                types_dict[retail].extend(self.dct[names])
                            else:
                                types_dict[retail] = self.dct[names]
                       
                        elif Levenshtein.ratio(website2,names2) >= 0.7:
                        #print(retail," :: " ,names," :: " ,Levenshtein.ratio(retail2,names2))
                            if retail in types_dict:
                                #print(x["name"])
                                #print(x["types"])
                                #print(len(types_dict[name]), name)
                                types_dict[retail].extend(self.dct[names])
                            else:
                                types_dict[retail] = self.dct[names]           
            
            try:
                print(len(types_dict[retail]))
                print(types_dict[retail])
                data = Counter(types_dict[retail])   
                print(data)
                types_dict[retail]= data.most_common(1)[0][0]  
            except KeyError:
                print("No categories for ",retail)
            except IndexError:
                print("No categories for ",retail)
                            
            #for k,v in types_dict.items():
            #    print(k, v)     
            #try:                    
            #    print(types_dict[retail])
            #except KeyError:
            #    print("No categories for ",retail)                                       
            #count +=1
            #if count == 3600:
            #    break            
            # If I found something then add it to the categories list
            #categories.append(temp_category)
        #with open('OSM_TestDetailed.json', 'w') as fout: 
        #    json.dump(types_dict, fout)         
        #for dict_key in types_dict:
            #print(dict_key)
            #data = Counter(types_dict[dict_key])
            #print(data)
            #types_dict[dict_key]= data.most_common(1)[0][0]      
        print(len(types_dict.keys()))
        print("Finished Categorizing OSM")   
        #with open('OSM_Test.json', 'w') as fout: 
        #    json.dump(types_dict, fout) 
            
        for k,v in types_dict.items():
            
            if v in ["bar", "bbq","fast_food;restaurant", "cafe;restaurant", "coffeeshop", "coffee","deli;cafe", "pub", "restaurant", "ice_cream", "food_court", "nightclub", "pub; restaurant", "cafe", "fast_food", "restaurant;pub;bar;fast_food", "Wine bar and deli", "fast_food;cafe", "chinese_takeaway", ]:
                types_dict[k] = "Restaurants and Cafes"
            elif v in ["butcher_&_convinience", "frozen_food", "canteen", "confectionery", "deli", "dairy", "farm", "greengrocer", "pastry", "seafood", "spices", "tea", "general", "kiosk", "mall", "supermarket", "convenience", "grocery", "alcohol","bakery","beverages", "fishmonger", "butcher", "fuel", "tobacconist", "cheese;wine", "cookies", "convenience;Polish", "convenience;alcohol", "fish", "seafood,butcher,grocer", "factory_outlet", "wholefood", "wine;cheese","specialist_food", "grocer", "butcher;greengrocer", "cheesemonger", "frozen_foods", "Indoor_Market", "fruiterer", "brazilian_store", "groceries", "convenience;newsagent", ]:      
                types_dict[k] = "Food and Grocery"
            elif v in ["department_store", "catalogue", "catalogue_store", "general_stores", "general_store",]:   
                types_dict[k] = "Department_Stores"
            elif v in ["clothing","Moss Bros.", "sports_store", "footwear", "Handbags and Leather Goods",  "fishing;clothing", "fancy-dress", "jewellery", "former_jewelry", "shopping_centre", "handbags", "Childrenswear", "lingerie", "baby_goods", "accessories", "bag","wedding" , "sports", "boutique", "clothes", "fabric", "fasion", "jewlery", "jewelry", "leather", "shoes", "tailor", "watches", "luxury_Toiletries", "hair_products", "beauty;hairdresser", "hair_accessories"]: 
                types_dict[k] = "Clothing and Accessories"
            elif v in ["chemists", "opticians", "health_and_beauty", "healthfood", "beauty", "chemist", "cosmetics", "erotic", "hairdresser", "health_food", "herbalist", "medical_supplies", "perfumery", "optician", "nutrition_supplies", "pharmacy", "heathfood"]:
                types_dict[k] = "Health and Cosmetics"
            elif v in ["computer_games", "electrical_supplies", "electronics_&_electrical", "electrical", "electronical", "game", "hardware", "fixme", "electrical_appliance","videogame", "computer", "electronics", "electrics", "hifi", "mobile_phones", "radiotechnics", "vacuum_cleaner", "games", "music", "cameras", "video", "video_games", "mobile_phone_accessory", "computer_parts", "electronics;electrical;radiotechnics", "electrical;white_goods;repair", "technology", ]: 
                types_dict[k] = "Electricals and Electronics"
            elif v in ["greetings cards", "newsagent;toy", "auto", "greeting cards", "map;gift", "second_hand", "trade", "anitques", "energy", "security", "fireplace","florist", "garden_centre", "newsagent", "hunting", "art", "collector", "craft", "bicycle", "car", "car_repair", "car_parts", "outdoor", "anime", "books", "gift", "lottery", "stationary", "ticket", "car_rental"]:      
                types_dict[k] = "Miscellaneous"
            elif v in ["tiles", "household_goods", "housewares", "windows;doors", "doityourself", "variety_store", "bed", "candles", "carpet", "curtains", "furniture", "decoration", "kitchen", "household", "discount", "charity", "discounter", "houseware", "hometextile", "ceramics", "decorating", "floors","interior_decoration","tile", "Bedding,_net_curtains_&_curtain_shop", "Home Store", "bedding", "homewares", "carpets", "tile_shop", "furniture;garden_centre", "discount_store", "furnishing", "outdoor; doityourself; garden_centre",  "household_appliances" ,"household,discount","Discount_Store","curtain;interior_decoration","homeware"]:  
                types_dict[k] =  "Homeware and DIY" 
            else:
                types_dict[k] =  "None" 
            if k == "JOHN LEWIS PLC":
                types_dict[k] = "Department_Stores"    
            if k == "DEBENHAMS RETAIL PLC":
                types_dict[k] = "Department_Stores"        
        '''        
        for retail in dataset:
            if retails not in types_dict:
                types_dict[retails] = "None"
        '''
        #with open('OSM_Categories_Remade.json', 'w') as fout: 
        #    json.dump(types_dict, fout) 
        
        # Checks how many companies had no categories out of the total companies     
        #print((len(types_dict.keys())/len(dataset))*100,"% Not Found")     
        print(len([x for x in types_dict.keys() if x != "None"])/len(dataset)*100,"% Found")
                       
            
        ''' 
        # Saves the tuple (Company, Category) into a text file
        text_file =  open("Output_OSM4_WithNewCategorization2.txt", "w") 
        for t in categorizations:
            text_file.write(str(t))
            text_file.write('\n')    
        
        return categorizations
        '''
    
    
    
parser = osm_parsing()
companies = parser.get_CompaniesData2()
#companies2 = parser.get_TopCompanies()
parser.parsing_OSM()
categories = parser.categorize_OSM3(sorted(companies))

 





