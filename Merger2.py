# -*- coding: utf-8 -*-

#from geoCoder import geoCoder
import json

class Merger2():
    def __init__(self):
        self.OSM_Cats = {}
        self.SIC_Cats = {}
        self.Google_Cats = {}
        self.All_Cats = {}
        types = ["Restaurants and Cafes", "Food and Grocery", "Department_Stores", "Clothing and Accessories", "Health and Cosmetics", "Electricals and Electronics", "Miscellaneous", "Homeware and DIY"]
        '''
        with open("Output_Merged_Cats.txt", "r") as filestream:
            for line in filestream:
                print(line)
                line = line.rstrip('\n')
                print(line)
                currentline = line.split(" == ")
                print(currentline)
                self.OSM_Cats[currentline[0]] = currentline[1]
                self.SIC_Cats[currentline[0]] = currentline[2] 
        ''' 
        with open("OSM_Categories_Remade.json", "r" ,encoding='utf8') as jsontext:
            self.OSM_Cats = json.load(jsontext)
        
        print(len(self.OSM_Cats))
        
        with open("SIC_Categories_Remade.json", "r" ,encoding='utf8') as jsontext:
            self.SIC_Cats = json.load(jsontext)
            
        print(len(self.OSM_Cats))
            
        with open("Google_Categories_Remade.json", "r" ,encoding='utf8') as jsontext:
            self.Google_Cats = json.load(jsontext) 
            
        print(len(self.OSM_Cats))   
         
        for k,v in self.Google_Cats.items():
            self.All_Cats[k] = [self.OSM_Cats[k], self.SIC_Cats[k], v]
                   
                       
        #coder = geoCoder()
        #coder.normalize_Companyname()
        #self.Google_Cats = coder.categorize_Google("JSON_data_merged.json")  
        
    
    def mergeCats(self):
        count = 0
        merged_cats = {}
        types = ["Restaurants and Cafes", "Food and Grocery", "Department_Stores", "Clothing and Accessories", "Health and Cosmetics", "Electricals and Electronics", "Miscellaneous", "Homeware and DIY"]
        for k,v in self.All_Cats.items():
            if v[0] == "None" and v[2] == "None":
                merged_cats[k] = v[1]
            elif v[0] == "None" and v[2] != "None":
                if v[1] == v[2]:
                    merged_cats[k] = v[1]
                else:
                    #print("Clash: ", k, v)
                    if v[2] in types:
                        merged_cats[k] = v[2]
                    else:
                        merged_cats[k] = v[1]
                            
                    #count+=1
                    
            elif v[0] != "None" and v[2] == "None":    
                if v[0] == v[1]:
                    merged_cats[k] = v[1]
                else:
                    #print("Clash: ", k, v)
                    #count +=1
                    merged_cats[k] = v[1]
            
            elif v[0] == v[1] and v[1] == v[2]:
                merged_cats[k] = v[1]   
            elif v[0] == v[1]:
                merged_cats[k] = v[1]
            elif v[1] == v[2]:
                merged_cats[k] = v[1]
            elif v[0] == v[2]:
                merged_cats[k] = v[2]
            else:
                # All three disagree
                merged_cats[k] = v[1]
                #print("Clash: ", k, v)
                #count +=1
            if k == "PUNCH PARTNERSHIPS (PML) LIMITED":
                merged_cats[k] =  v[2]     
            if k == "GREENE KING BREWING AND RETAILING LIMITED":
                merged_cats[k] =  v[2]
            if k == "PETS AT HOME LTD":
                merged_cats[k] =  v[2]
            if k == "B & M RETAIL LIMITED":
                merged_cats[k] =  v[1]    
                    
                           
        with open('Mixed_Categories_Remade.json', 'w') as fout: 
            json.dump(merged_cats, fout)        
        
        '''
        for k,v in self.Google_Cats.items():
            if v == "None":
                self.Google_Cats[k] = self.SIC_Cats[k]
        
        for k,v in self.Google_Cats.items():
            if self.Google_Cats[k] == self.OSM_Cats[k] and self.Google_Cats[k] == self.SIC_Cats[k]:
                merged_cats[k] = v
            elif self.Google_Cats[k] == self.OSM_Cats[k] and self.Google_Cats[k] != self.SIC_Cats[k]:
                merged_cats[k] = v
            elif self.Google_Cats[k] != self.OSM_Cats[k] and self.Google_Cats[k] == self.SIC_Cats[k]:
                merged_cats[k] = v
            elif self.OSM_Cats[k] == self.SIC_Cats[k] and self.OSM_Cats[k] != self.Google_Cats[k]:
                merged_cats[k] = self.OSM_Cats[k]
            else:
                # If all disagrees, SIC Wins
                print(k,self.OSM_Cats[k],self.SIC_Cats[k],self.Google_Cats[k])
                count +=1
                merged_cats[k] = self.SIC_Cats[k]
            if k == "GAME RETAIL LIMITED":
                merged_cats[k] = self.Google_Cats[k]    
            if k == "PUNCH PARTNERSHIPS (PML) LIMITED":
                merged_cats[k] =  self.Google_Cats[k] 
            if k == "EURO GARAGES LIMITED":
                merged_cats[k] = self.OSM_Cats[k] 
            if k == "TJX UK":
                merged_cats[k] = self.Google_Cats[k]
            if k == "SPECSAVERS OPTICAL SUPERSTORES LIMITED":
                merged_cats[k] = self.Google_Cats[k] 
            if k == "ST ALBANS OPERATING COMPANY LIMITED":
                merged_cats[k] = self.Google_Cats[k]
            if k == "BOOTS UK LIMITED":
                merged_cats[k] = self.Google_Cats[k]  
            if k == "BOOTS OPTICIANS PROFESSIONAL SERVICES LIMITED":
                merged_cats[k] = self.Google_Cats[k]  
            if k == "MARTIN MCCOLL LIMITED":
                merged_cats[k] = self.Google_Cats[k]
            if k == "SUPERDRUG STORES PLC":
                merged_cats[k] = self.SIC_Cats[k]  
            if k == "VOISINS LIMITED":
                merged_cats[k] = "Department_Stores"  
            if k == "SPORTSDIRECT.COM RETAIL LIMITED":
                merged_cats[k] = self.Google_Cats[k]  
            if k == "HOMEBASE LIMITED":
                merged_cats[k] = self.Google_Cats[k]          
                                    
                       
        '''               
        print(count)                          
        #with open('mixed_categories2.json', 'w') as fout: 
        #    json.dump(merged_cats, fout)      
            

x = Merger2()
x.mergeCats()            
            
                    
                      
        