# -*- coding: utf-8 -*-

import json
import re
import csv
import os


class SIC_Categories():
    def __init__(self):
        # Loops through all the SIC Databases I created and parses them here in a dict of list of dicts where each dict is a file and the list inside it is the dicts of its columns
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
    
           
    def categorize2(self):
        # Categorizes the companies based on the file they are in, as I already put the companies with specific SIC codes in different files
        # However this function makes the categorizations compatible with the OSM categorizations so that I can merge them together eventually
        categories = {}
        
        for dicts in self.csv_file:
            if dicts["SIC"] in ["47190"]:
                categories[dicts["company_name"]] = "Department_Stores"
            elif dicts["SIC"] in ["47110", "47210", "47220", "47230", "47240", "47250", "47260", "47290", "47810"]:
                categories[dicts["company_name"]] = "Food and Grocery"    
            elif dicts["SIC"] in ["47710", "47720", "47721", "47722", "47770", "47780", "47789", "47820"]:
                categories[dicts["company_name"]] = "Clothing and Accessories" 
            elif dicts["SIC"] in ["47730", "47740", "47741", "47749", "47750", "47782"]:
                categories[dicts["company_name"]] = "Health and Cosmetics"
            elif dicts["SIC"] in ["47410", "47420", "47421", "47429", "47430", "47540", "47591", "47630"]:
                categories[dicts["company_name"]] = "Electricals and Electronics"
            elif dicts["SIC"] in ["47510", "47520", "47530", "47590", "47599"]:
                categories[dicts["company_name"]] = "Homeware and DIY"
            elif dicts["SIC"] in ["47300", "47610", "47620", "47640", "47650", "47760", "47781", "47791", "47799", "47890", "47790", "47910", "47990"]:
                categories[dicts["company_name"]] = "Miscellaneous"  
            elif dicts["SIC"] in ["56101", "56102", "56103"]:
                categories[dicts["company_name"]] = "Restaurants and Cafes"
        
        with open('SIC_Categories_Remade.json', 'w') as fout: 
            json.dump(categories, fout)        
                    
        
        '''
        for file,listt in self.dict_files.items():
            if file == "SIC47110_companies.csv":
                for dicts in listt:
                    categories.append((dicts["company_name"],dicts["company_number"],"General_Store/Supermarket"))
            elif file == "SIC47210_47240_companies.csv" :
                for dicts in listt:
                    categories.append((dicts["company_name"],dicts["company_number"],"Food and Drink Stores"))
            elif file == "SIC47250_47260_companies.csv" :
                for dicts in listt:
                    categories.append((dicts["company_name"],dicts["company_number"],"Food and Drink Stores"))
            elif file == "SIC47190_companies.csv" :
                for dicts in listt:
                    categories.append((dicts["company_name"],dicts["company_number"],"Department_Stores"))
            elif file == "SIC47510_47720_companies.csv" :
                for dicts in listt:
                    categories.append((dicts["company_name"],dicts["company_number"],"Clothes and Accessories"))
            elif file == "SIC47430_47630_companies.csv" :
                for dicts in listt:
                    categories.append((dicts["company_name"],dicts["company_number"],"Household Goods"))
            elif file == "SIC47730_47750_companies.csv" :
                for dicts in listt:
                    categories.append((dicts["company_name"],dicts["company_number"],"Pharmacies and Cosmetics"))
            elif file == "SIC47410_47790_companies.csv" :
                for dicts in listt:
                    if dicts["SIC"] == "47770":
                        categories.append((dicts["company_name"],dicts["company_number"],"Clothes and Accessories"))
                    elif dicts["SIC"] == "47530":
                        categories.append((dicts["company_name"],dicts["company_number"],"Household Goods"))    
                    else:    
                        categories.append((dicts["company_name"],dicts["company_number"],"Electronics/Vehicles/Outdoor/Sports"))
            elif file == "SIC56101_56103_companies.csv" :
                for dicts in listt:
                    categories.append((dicts["company_name"],dicts["company_number"],"Restaurants and Cafes"))   
        '''
        
        '''
        # Getting the official BetaCompanyData to make the company names official
        official_data = {}
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
        '''
        # Gathers the company number, company name, and categorization in a list of tuples and makes sure that there is no duplicate company number
        '''
        categories_modified = []                    
        for name, number, cat in categories:
            if number in official_data and number not in [x[1] for x in categories_modified]:
                categories_modified.append((official_data[number], number, cat))
        '''
        '''                                                
        categories_modified = []                    
        for name, number, cat in categories:
            if number not in [x[1] for x in categories_modified]:
                categories_modified.append((name, number, cat))
        
                                                                                                        
        for i,j,k in categories_modified:
            print(i,j,k)     
        
        # Writes the tuples in a new file
        text_file =  open("Output_SIC_Modified.txt", "w") 
        for t in categories_modified:
            text_file.write(str(t))
            text_file.write('\n')
        
        return categories_modified
    
        '''
                                         
parser = SIC_Categories()
parser.categorize2()                    