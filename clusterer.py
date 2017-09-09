# -*- coding: utf-8 -*-

import subprocess
import csv
import json
import time
import numpy as np
import re
import pandas as pd
from ast import literal_eval
import matplotlib
import matplotlib.cm as cm
from math import radians, cos, sin, asin, sqrt
import os
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster.optics_ import OPTICS
import copy
from collections import defaultdict
#%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import DistanceMetric
import seaborn as sns
import hdbscan
from orangecontrib.associate.fpgrowth import *
#from geopy.distance import vincenty
#from sklearn import metrics
from scipy.spatial import distance
#from scipy.spatial.distance import pdist
from operator import itemgetter
from collections import Counter
from datetime import datetime as dt
from sklearn import metrics
from optics import *
#from itertools import chain




class clusterer:
    def __init__(self):
        
        self.dir = os.getcwd()
        print(self.dir)
        csv_file = []
        # The data structure used here is a list of dicts where each dict is a row in the CSV file with header as key and attribute as value
        with open("finalData_FinalVersion.csv",encoding='utf8') as sqldbCSV:
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
                my_dict["district_name"] = line[2]
                my_dict["county_name"] = line[3]
                my_dict["region_name"] = line[4]
                my_dict["postcode"] = line[5]
                my_dict["company_number"] = line[6]
                my_dict["tenure"] = line[7]
                my_dict["date"] = line[8]
                my_dict["company_name"] = line[9]
                my_dict["title_number"] = line[10]
                my_dict["SIC"] = line[11]
                my_dict["category"] = line[12]
                my_dict["latitude"] = line[13]
                my_dict["longitude"] = line[14]
                
                
                
                csv_file.append(my_dict)        
        '''
        f = csv.writer(open("final_Data_with_Range.csv", "w", encoding = 'utf8', newline=''))     
        f.writerow(["street_name", "town_name", "district_name", "county_name", "region_name", "postcode", "company_number", "tenure", "date", "company_name", "title_number", "SIC", "category", "latitude", "longitude"])       
        for dicts in csv_file:
            #lat = float(dicts["latitude"])
            #lon = float(dicts["longitude"])
            if len(dicts["date"]) != 0:
                z = dicts["date"].split()[0]
                z = z.replace("-","/")
                x = dt.strptime(z,"%Y/%m/%d")
            else:
                x = dt.strptime("0001/01/01","%Y/%m/%d")
            start = dt.strptime('01/01/2000',"%d/%m/%Y")
            end = dt.strptime('30/12/2013',"%d/%m/%Y")
            if x >= start and x <= end:
                f.writerow([dicts["street_name"], dicts["town_name"], dicts["district_name"], dicts["county_name"], dicts["region_name"], dicts["postcode"], dicts["company_number"], dicts["tenure"], dicts["date"], dicts["company_name"], dicts["title_number"], dicts["SIC"], dicts["category"], dicts["latitude"], dicts["longitude"]])
        '''              
    
    
    def my_haversine(self,u,v):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        lat1 = u[0]
        lon1 = u[1]
        lat2 = v[0]
        lon2 = v[1]
        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles
        return c * r    
            
    def DBSCAN_Clustering(self, file):   
        
        csv_file = []
        # The data structure used here is a list of dicts where each dict is a row in the CSV file with header as key and attribute as value
        with open(file,encoding='utf8') as sqldbCSV:
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
                my_dict["district_name"] = line[2]
                my_dict["county_name"] = line[3]
                my_dict["region_name"] = line[4]
                my_dict["postcode"] = line[5]
                my_dict["company_number"] = line[6]
                my_dict["tenure"] = line[7]
                my_dict["date"] = line[8]
                my_dict["company_name"] = line[9]
                my_dict["title_number"] = line[10]
                my_dict["SIC"] = line[11]
                my_dict["category"] = line[12]
                my_dict["latitude"] = line[13]
                my_dict["longitude"] = line[14]
                
                
                
                csv_file.append(my_dict)   
        
        coordinates = np.empty([len(csv_file),2],dtype=np.float64)
        for index, dicts in enumerate(csv_file):
            #lat = dicts["Coordinates"].split(",")[0].replace("(","")
            #lon = dicts["Coordinates"].split(",")[1].replace(")","")
            coordinates[index] = [float(dicts["latitude"]),float(dicts["longitude"])]
               
        print(coordinates.dtype)
        #vincenty(coordinates,coordinates2).meters
        print(coordinates) 
        print(coordinates[0])  
        
        #X = pdist(coordinates, lambda X, Y: vincenty(X,Y).meters)   
        #print(X) 
        kms_per_radian = 6371.0088
        epsilon = 0.8
        #db = DBSCAN(eps=5/kms_per_radian, min_samples=10, metric=lambda X, Y: vincenty(X,Y).meters, n_jobs = -1).fit(coordinates) 
        db = DBSCAN(eps=epsilon/kms_per_radian, min_samples=4, algorithm='ball_tree', metric='haversine', n_jobs = -1).fit(np.radians(coordinates)) 
        #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        #core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        print(labels)
        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        #clusters = pd.Series([coordinates[labels == n] for n in range(n_clusters)], index = range(n_clusters))
        
        print('Estimated number of clusters: %d' % n_clusters)
        #haversine = DistanceMetric.get_metric("haversine")
        
        clusters_dict = {}
        for n in range(n_clusters):
            clusters_dict[n] = coordinates[labels == n].tolist()
        
        print(len(clusters_dict))
        coordinates_to_clusters = {}
        for k,v in clusters_dict.items():
            for coordinate in v:
                x = tuple(coordinate)
                coordinates_to_clusters[x] = k
        
        print(len(coordinates_to_clusters))        
        
        #test = distance.cdist(coordinates, coordinates, self.my_haversine)
        #print(len(test))
                
        silhouettes = []
        #dist = DistanceMetric.get_metric('haversine')
        '''
        for x in range(10):
            labels2 = []
            print("Starting a round of sihouettes")
            sub_coordinates = coordinates[np.random.choice(coordinates.shape[0], 10000, replace=False),:] 
            for coor in sub_coordinates:
                if tuple(coor) in coordinates_to_clusters:
                    labels2.append(coordinates_to_clusters[tuple(coor)])
                else:
                    labels2.append(-1)       
            print(sub_coordinates[0], labels2[0])
            print(sub_coordinates[1], labels2[1])
            print(sub_coordinates[2], labels2[2])
            print(sub_coordinates[3], labels2[3])
            print(sub_coordinates[4], labels2[4])       
            #labels2 = [coordinates_to_clusters[tuple(coor)] for coor in sub_coordinates if tuple(coor) in coordinates_to_clusters]
            #precomp = dist.pairwise(sub_coordinates)  
            precomp = distance.cdist(sub_coordinates, sub_coordinates, self.my_haversine)   
            print("Created distance matrix, computing silhouette for it")  
            silhouettes.append(metrics.silhouette_score(precomp, labels2, metric="precomputed"))
            print(silhouettes)
            
        print("Silhouette Score is : ", np.mean(silhouettes))
        #print(clusters)
        '''
        '''
        for x in range(10):
            silhouettes.append(metrics.silhouette_score(coordinates, labels, metric=self.my_haversine, sample_size=10000))
            print(silhouettes)
            
        print("Silhouette Score is : ", np.mean(silhouettes))
        '''    
                
        #clusters_j = clusters.to_json(orient = 'records', force_ascii = False)
        with open('clusterings2.json', 'w') as fout: 
            json.dump(clusters_dict, fout)  
        
        for index, dicts in enumerate(csv_file):
            #lat = dicts["Coordinates"].split(",")[0].replace("(","")
            #lon = dicts["Coordinates"].split(",")[1].replace(")","")
            coordinate = (float(dicts["latitude"]),float(dicts["longitude"]))
            if coordinate in coordinates_to_clusters:
                csv_file[index]["cluster"] = coordinates_to_clusters[coordinate] 
        
        f = csv.writer(open("final_Data_with_Range_Cluster.csv", "w", encoding = 'utf8', newline=''))     
        f.writerow(["street_name", "town_name", "district_name", "county_name", "region_name", "postcode", "company_number", "tenure", "date", "company_name", "title_number", "SIC", "category", "latitude", "longitude", "cluster"])       
        for dicts in csv_file:
            if "cluster" in dicts:
                f.writerow([dicts["street_name"], dicts["town_name"], dicts["district_name"], dicts["county_name"], dicts["region_name"], dicts["postcode"], dicts["company_number"], dicts["tenure"], dicts["date"], dicts["company_name"], dicts["title_number"], dicts["SIC"], dicts["category"], dicts["latitude"], dicts["longitude"], dicts["cluster"]])
                        
        '''
        f = csv.writer(open("final_Data_for_QGIS.csv", "w", encoding = 'utf8', newline=''))     
        f.writerow(["street_name", "town_name", "district_name", "county_name", "region_name", "postcode", "company_number", "tenure", "date", "company_name", "title_number", "SIC", "category", "latitude", "longitude", "cluster"])       
        for dicts in csv_file:
            if "cluster" in dicts:
                f.writerow([dicts["street_name"], dicts["town_name"], dicts["district_name"], dicts["county_name"], dicts["region_name"], dicts["postcode"], dicts["company_number"], dicts["tenure"], dicts["date"], dicts["company_name"], dicts["title_number"], dicts["SIC"], dicts["category"], dicts["latitude"], dicts["longitude"], dicts["cluster"]])
        '''
        '''           
        #print(clusters_j)
        colors = list("bgrcmyk")
        i = 0
        markers = list(matplotlib.lines.Line2D.markers.keys())
        df = pd.read_csv("final_Data_with_OA2_QGIS.csv")
        fig, ax = plt.subplots(figsize=[10, 6])
        for k,v in clusters_dict.items():
            temp = np.empty([len(v),2],dtype=np.float64)
            color = colors[i % len(colors)]
            markerr = markers[i % len(markers)]
            for index, coords in enumerate(v):
                #print(coords)
                temp[index] = coords 
            df_scatter = ax.scatter(temp[:,1], temp[:,0], alpha=0.9, s=3, c = color, marker = markerr)
            i +=1  
                
                
        ax.set_title('Full data set with epsilon = {}'.format(epsilon))
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        #ax.legend([df_scatter, rs_scatter], ['Full set', 'Reduced set'], loc='upper right')
        plt.savefig('clustered.pdf', format='pdf', dpi=1200)
        plt.show()
        '''
    def HDBSCAN_Clustering(self, file):
        
        #sns.set_context('poster')
        #sns.set_style('white')
        #sns.set_color_codes()
        #plot_kwds = {'alpha' : 0.5, 's' : 60, 'linewidths':0}
        csv_file = []
        # The data structure used here is a list of dicts where each dict is a row in the CSV file with header as key and attribute as value
        with open(file,encoding='utf8') as sqldbCSV:
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
                my_dict["district_name"] = line[2]
                my_dict["county_name"] = line[3]
                my_dict["region_name"] = line[4]
                my_dict["postcode"] = line[5]
                my_dict["company_number"] = line[6]
                my_dict["tenure"] = line[7]
                my_dict["date"] = line[8]
                my_dict["company_name"] = line[9]
                my_dict["title_number"] = line[10]
                my_dict["SIC"] = line[11]
                my_dict["category"] = line[12]
                my_dict["latitude"] = line[13]
                my_dict["longitude"] = line[14]
                
                
                
                csv_file.append(my_dict)   
        
        coordinates = np.empty([len(csv_file),2],dtype=np.float64)
        for index, dicts in enumerate(csv_file):
            #lat = dicts["Coordinates"].split(",")[0].replace("(","")
            #lon = dicts["Coordinates"].split(",")[1].replace(")","")
            coordinates[index] = [float(dicts["latitude"]),float(dicts["longitude"])]
        

        #plt.scatter(*coordinates.T, s=50, linewidth=0, c='b', alpha=0.25)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=20,min_samples=4, metric= 'haversine').fit(np.radians(coordinates))    
        #core_samples_mask = np.zeros_like(clusterer.labels_, dtype=bool)
        #core_samples_mask[clusterer.core_sample_indices_] = True
        labels = clusterer.labels_
        print(labels)
        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        #clusters = pd.Series([coordinates[labels == n] for n in range(n_clusters)], index = range(n_clusters))
        
        print('Estimated number of clusters: %d' % n_clusters)
        #print(clusters)
        Noise = len([x for x in labels if x == -1])
        print("Noise = ", Noise)
        
        clusters_dict = {}
        for n in range(n_clusters):
            clusters_dict[n] = coordinates[labels == n].tolist()
            
        coordinates_to_clusters = {}
        for k,v in clusters_dict.items():
            for coordinate in v:
                x = tuple(coordinate)
                coordinates_to_clusters[x] = k
                
        #clusters_j = clusters.to_json(orient = 'records', force_ascii = False)
        with open('clusterings_hdbscan.json', 'w') as fout: 
            json.dump(clusters_dict, fout)  
        
        for index, dicts in enumerate(csv_file):
            #lat = dicts["Coordinates"].split(",")[0].replace("(","")
            #lon = dicts["Coordinates"].split(",")[1].replace(")","")
            coordinate = (float(dicts["latitude"]),float(dicts["longitude"]))
            if coordinate in coordinates_to_clusters:
                csv_file[index]["cluster"] = coordinates_to_clusters[coordinate]    
        
        
        f = csv.writer(open("final_Data_with_Range_Cluster_hdbscan.csv", "w", encoding = 'utf8', newline=''))     
        f.writerow(["street_name", "town_name", "district_name", "county_name", "region_name", "postcode", "company_number", "tenure", "date", "company_name", "title_number", "SIC", "category", "latitude", "longitude", "cluster"])       
        for dicts in csv_file:
            if "cluster" in dicts:
                f.writerow([dicts["street_name"], dicts["town_name"], dicts["district_name"], dicts["county_name"], dicts["region_name"], dicts["postcode"], dicts["company_number"], dicts["tenure"], dicts["date"], dicts["company_name"], dicts["title_number"], dicts["SIC"], dicts["category"], dicts["latitude"], dicts["longitude"], dicts["cluster"]])
        
        '''    
        labels = clusterer.labels_
        palette = sns.color_palette('deep', np.unique(labels).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
        plt.scatter(coordinates.T[0], coordinates.T[1], c=colors, **plot_kwds)
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        #plt.savefig('clustered.pdf', format='pdf', dpi=1200)
        plt.show()
        '''
    
    def OPTICS_Clustering(self, file):
        csv_file = []
        with open(file,encoding='utf8') as sqldbCSV:
            csvread = csv.reader(sqldbCSV, delimiter=",")
            count = -1
            for line in csvread:
                #print(line)
                count = count + 1
                # Ignoring the header row
                
                if count == 0:
                    continue
                #if count == 10000:
                #    break
                my_dict = {}
                #my_dict["ID"] = line[0]
                my_dict["street_name"] = line[0]
                my_dict["town_name"] = line[1]
                my_dict["district_name"] = line[2]
                my_dict["county_name"] = line[3]
                my_dict["region_name"] = line[4]
                my_dict["postcode"] = line[5]
                my_dict["company_number"] = line[6]
                my_dict["tenure"] = line[7]
                my_dict["date"] = line[8]
                my_dict["company_name"] = line[9]
                my_dict["title_number"] = line[10]
                my_dict["SIC"] = line[11]
                my_dict["category"] = line[12]
                my_dict["latitude"] = line[13]
                my_dict["longitude"] = line[14]
                
                
                
                csv_file.append(my_dict)   
        
        coordinates = np.empty([len(csv_file),2],dtype=np.float64)
        for index, dicts in enumerate(csv_file):
            #lat = dicts["Coordinates"].split(",")[0].replace("(","")
            #lon = dicts["Coordinates"].split(",")[1].replace(")","")
            coordinates[index] = [float(dicts["latitude"]),float(dicts["longitude"])]
            count += 1
        print(len(csv_file))    
        print(coordinates.dtype)
        #vincenty(coordinates,coordinates2).meters
        print(coordinates)  
        # max_bound should be in kilometres because my_haversine multiplied the number by 6731 hence converting it to kilometers
        clust = OPTICS(max_bound=2.0, min_samples=4, algorithm = "ball_tree", metric=self.my_haversine).fit(coordinates)
        #5.0*6.0/6371.0088
        labels = clust.labels_
        print("labels length = " , len(clust.labels_))
        n_clusters = len(set(clust.labels_)) - int(-1 in clust.labels_)
        print(n_clusters)
        print("Noise = ", len([x for x in clust.labels_ if x == -1]))
        
        clusters_dict = {}
        for n in range(n_clusters):
            clusters_dict[n] = coordinates[labels == n].tolist()
            
        coordinates_to_clusters = {}
        for k,v in clusters_dict.items():
            for coordinate in v:
                x = tuple(coordinate)
                coordinates_to_clusters[x] = k
                
        #clusters_j = clusters.to_json(orient = 'records', force_ascii = False)
        with open('clusterings_optics.json', 'w') as fout: 
            json.dump(clusters_dict, fout)  
        
        for index, dicts in enumerate(csv_file):
            #lat = dicts["Coordinates"].split(",")[0].replace("(","")
            #lon = dicts["Coordinates"].split(",")[1].replace(")","")
            coordinate = (float(dicts["latitude"]),float(dicts["longitude"]))
            if coordinate in coordinates_to_clusters:
                csv_file[index]["cluster"] = coordinates_to_clusters[coordinate]    
        
        
        f = csv.writer(open("final_Data_with_Range_Cluster_optics.csv", "w", encoding = 'utf8', newline=''))     
        f.writerow(["street_name", "town_name", "district_name", "county_name", "region_name", "postcode", "company_number", "tenure", "date", "company_name", "title_number", "SIC", "category", "latitude", "longitude", "cluster"])       
        for dicts in csv_file:
            if "cluster" in dicts:
                f.writerow([dicts["street_name"], dicts["town_name"], dicts["district_name"], dicts["county_name"], dicts["region_name"], dicts["postcode"], dicts["company_number"], dicts["tenure"], dicts["date"], dicts["company_name"], dicts["title_number"], dicts["SIC"], dicts["category"], dicts["latitude"], dicts["longitude"], dicts["cluster"]])
        

    def leaderfollowerModel(self,file, clustering):
        
        csv_file = []
        # The data structure used here is a list of dicts where each dict is a row in the CSV file with header as key and attribute as value
        with open(file,encoding='utf8') as sqldbCSV:
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
                my_dict["district_name"] = line[2]
                my_dict["county_name"] = line[3]
                my_dict["region_name"] = line[4]
                my_dict["postcode"] = line[5]
                my_dict["company_number"] = line[6]
                my_dict["tenure"] = line[7]
                my_dict["date"] = line[8]
                my_dict["company_name"] = line[9]
                my_dict["title_number"] = line[10]
                my_dict["SIC"] = line[11]
                my_dict["category"] = line[12]
                my_dict["latitude"] = line[13]
                my_dict["longitude"] = line[14]
                my_dict["cluster"] = line[15]
                
                
                
                csv_file.append(my_dict)
        
        #with open('clusterings2.json', "r" ,encoding='utf8') as jsontext:
        #    clusters_dict = json.load(jsontext)
        
        clusters_dict2 = {}
        for dicts in csv_file:
            if dicts["cluster"] not in clusters_dict2:
                clusters_dict2[dicts["cluster"]] = [dicts["title_number"]]
            else:
                clusters_dict2[dicts["cluster"]].extend([dicts["title_number"]])    
           
        titles_dict = {}
        self.category_dict = {}
        counties= []
        for dicts in csv_file:
            titles_dict[dicts["title_number"]] = (dicts["company_name"],dicts["date"], dicts["category"], dicts["county_name"], dicts["region_name"], dicts["cluster"])
            self.category_dict[dicts["company_name"]] = dicts["category"]
            counties.append(dicts["county_name"])
        
        counties = set(counties)
        leaders_count = defaultdict(int)
        all_leaders = []    
        total_leaders = []
        for mycounty in counties:    
            # County = West Midlands
            leaders_dict = {}
            first_arrival_1 = []
            first_arrival_2 = []
            first_arrival_3 = []
            first_arrival_4 = []
            first_arrival_5 = []
            first_arrival_6 = []
            first_arrival_7 = []
            first_arrival_8 = []
            count = 0
            for clusters, titles in clusters_dict2.items():
                thecounty = [titles_dict[title][3] for title in titles if titles_dict[title][3] == mycounty]
                if thecounty:
                    cat_1 = []
                    cat_2 = []
                    cat_3 = []
                    cat_4 = []
                    cat_5 = []
                    cat_6 = []
                    cat_7 = []
                    cat_8 = []
                    for title in titles:
                        name, date, cat, county, region, cluster = titles_dict[title]
                        if name != "PUNCH PARTNERSHIPS (PML) LIMITED" and name != "MITCHELLS & BUTLERS RETAIL LIMITED" and name != "MARSTON'S PUBS LIMITED" and name != "ADMIRAL TAVERNS LIMITED" and name != "SPIRIT PUB COMPANY (MANAGED) LIMITED":
                            
                            if cat == "Restaurants and Cafes":
                                    cat_1.append((name,date,cat, county, region, cluster)) 
                            elif cat == "Food and Grocery":      
                                    cat_2.append((name,date,cat, county, region, cluster))
                            elif cat == "Department_Stores":   
                                    cat_3.append((name,date,cat, county, region, cluster))
                            elif cat == "Clothing and Accessories": 
                                    cat_4.append((name,date,cat, county, region, cluster))
                            elif cat == "Health and Cosmetics":
                                    cat_5.append((name,date,cat, county, region, cluster))
                            elif cat == "Electricals and Electronics": 
                                    cat_6.append((name,date,cat, county, region, cluster))
                            elif cat == "Miscellaneous":      
                                    cat_7.append((name,date,cat, county, region, cluster))
                            elif cat == "Homeware and DIY":  
                                    cat_8.append((name,date,cat, county, region, cluster))
                    
                    if len(cat_1) != 0: 
                        #print(cat_1)           
                        first_arrival_1.append(min(cat_1, key = itemgetter(1)))
                        leaders_count[min(cat_1, key = itemgetter(1))[0]] += 1
                    if len(cat_2) != 0:
                        #print(cat_2)     
                        first_arrival_2.append(min(cat_2, key = itemgetter(1)))
                        leaders_count[min(cat_2, key = itemgetter(1))[0]] += 1
                    if len(cat_3) != 0: 
                        #print(cat_3)
                        first_arrival_3.append(min(cat_3, key = itemgetter(1)))
                        leaders_count[min(cat_3, key = itemgetter(1))[0]] += 1
                    if len(cat_4) != 0: 
                        #print(cat_4)
                        first_arrival_4.append(min(cat_4, key = itemgetter(1)))
                        leaders_count[min(cat_4, key = itemgetter(1))[0]] += 1
                    if len(cat_5) != 0: 
                        #print(cat_5)
                        first_arrival_5.append(min(cat_5, key = itemgetter(1)))
                        leaders_count[min(cat_5, key = itemgetter(1))[0]] += 1
                    if len(cat_6) != 0: 
                        #print(cat_6)
                        first_arrival_6.append(min(cat_6, key = itemgetter(1)))
                        leaders_count[min(cat_6, key = itemgetter(1))[0]] += 1
                    if len(cat_7) != 0: 
                        #print(cat_7)
                        first_arrival_7.append(min(cat_7, key = itemgetter(1)))
                        leaders_count[min(cat_7, key = itemgetter(1))[0]] += 1
                    if len(cat_8) != 0: 
                        #print(cat_8)
                        first_arrival_8.append(min(cat_8, key = itemgetter(1)))
                        leaders_count[min(cat_8, key = itemgetter(1))[0]] += 1    
                
            common_1 = Counter(first[0] for first in first_arrival_1) 
            #print(common_1)
            common_2 = Counter(first[0] for first in first_arrival_2)
            #print(common_2)
            common_3 = Counter(first[0] for first in first_arrival_3)
            #print(common_3)
            common_4 = Counter(first[0] for first in first_arrival_4)
            #print(common_4)
            common_5 = Counter(first[0] for first in first_arrival_5)
            #print(common_5)
            common_6 = Counter(first[0] for first in first_arrival_6)
            #print(common_6)
            common_7 = Counter(first[0] for first in first_arrival_7)
            #print(common_7)
            common_8 = Counter(first[0] for first in first_arrival_8)
            #print(common_8)  
            
            if common_1: total_leaders.append(common_1.most_common(1)[0][0])
            if common_2: total_leaders.append(common_2.most_common(1)[0][0])
            if common_3: total_leaders.append(common_3.most_common(1)[0][0])
            if common_4: total_leaders.append(common_4.most_common(1)[0][0])
            if common_5: total_leaders.append(common_5.most_common(1)[0][0])
            if common_6: total_leaders.append(common_6.most_common(1)[0][0])
            if common_7: total_leaders.append(common_7.most_common(1)[0][0])
            if common_8: total_leaders.append(common_8.most_common(1)[0][0])
              
            if common_1: leaders_dict["Restaurants and Cafes"] = common_1.most_common(1)[0][0]
            if common_2: leaders_dict["Food and Grocery"] = common_2.most_common(1)[0][0]
            if common_3: leaders_dict["Department_Stores"] = common_3.most_common(1)[0][0]
            if common_4: leaders_dict["Clothing and Accessories"] = common_4.most_common(1)[0][0]
            if common_5: leaders_dict["Health and Cosmetics"] = common_5.most_common(1)[0][0]
            if common_6: leaders_dict["Electricals and Electronics"] = common_6.most_common(1)[0][0]
            if common_7: leaders_dict["Miscellaneous"] = common_7.most_common(1)[0][0]
            if common_8: leaders_dict["Homeware and DIY"] = common_8.most_common(1)[0][0]
            

            all_leaders.append((leaders_dict,mycounty))
        
        #for k,v in leaders_count.items():
        #    print(k,v)    
                
        #with open('D.json', 'w') as fout: 
        #    print("Dumping data to leaders")
        #    json.dump(all_leaders, fout)
        
        
        leaderCounts = {}
        leaderCounts["Restaurants and Cafes"] = Counter(y[0]['Restaurants and Cafes'] for y in all_leaders if 'Restaurants and Cafes' in y[0])
        leaderCounts["Food and Grocery"] = Counter(y[0]['Food and Grocery'] for y in all_leaders if 'Food and Grocery' in y[0])
        leaderCounts["Department_Stores"] = Counter(y[0]['Department_Stores'] for y in all_leaders if 'Department_Stores' in y[0])
        leaderCounts["Clothing and Accessories"] = Counter(y[0]['Clothing and Accessories'] for y in all_leaders if 'Clothing and Accessories' in y[0])
        leaderCounts["Health and Cosmetics"] = Counter(y[0]['Health and Cosmetics'] for y in all_leaders if 'Health and Cosmetics' in y[0])
        leaderCounts["Electricals and Electronics"] = Counter(y[0]['Electricals and Electronics'] for y in all_leaders if 'Electricals and Electronics' in y[0])
        leaderCounts["Miscellaneous"] = Counter(y[0]['Miscellaneous'] for y in all_leaders if 'Miscellaneous' in y[0])
        leaderCounts["Homeware and DIY"] = Counter(y[0]['Homeware and DIY'] for y in all_leaders if 'Homeware and DIY' in y[0])
        
        if clustering == "dbscan":
            with open('DBSCAN_leaders.json', 'w') as fout: 
                print("Dumping data to main leaders")
                json.dump(leaderCounts, fout)
        elif clustering == "optics":
            with open('OPTICS_leaders.json', 'w') as fout: 
                print("Dumping data to main leaders")
                json.dump(leaderCounts, fout)
        elif clustering == "hdbscan":
            with open('HDBSCAN_leaders.json', 'w') as fout: 
                print("Dumping data to main leaders")
                json.dump(leaderCounts, fout)
                    
        total_leaders = set(total_leaders)
        print(len(total_leaders))       
        clusters_dict3 = {}
        self.all_companies = []
        for k,v in clusters_dict2.items():
            companies = []
            for title in v:
                companies.append(titles_dict[title][0])
                self.all_companies.append(titles_dict[title][0])
            clusters_dict3[k] = list(set(companies))
        
        with open('clusters_companies.json', 'w') as fout: 
            print("Dumping data for cluster companies")
            json.dump(clusters_dict3, fout)
        
        
        #associations = {}  
        #associations2 = {}
        occurence_counts = defaultdict(int)
        self.all_companies = set(self.all_companies)  
        print(len(self.all_companies))
        
        for company in self.all_companies:
            #associations[company] = []
            for cluster in clusters_dict3.values():
                if company in cluster:
                    #cluster2 = [x for x in cluster if x != company and self.category_dict[company] == self.category_dict[x]]
                    #associations[company].extend(cluster2)
                    occurence_counts[company] += 1
            #associations2[str((company,occurence_counts[company]))] = associations[company]
        
        '''            
        for k,v in associations2.items():
            data = Counter(v)
            #associations[k]= data
            associations2[k]= data.most_common(10)
            
                    
        with open('clusters_associations.json', 'w') as fout: 
            print("Dumping associations for cluster companies")
            json.dump(associations2, fout)
        '''
        if clustering == "dbscan":
            self.relative_leadernumber_dbscan = {}    
            for company in self.all_companies:
                if company in total_leaders:
                    self.relative_leadernumber_dbscan[company] = leaders_count[company]/occurence_counts[company]
            with open("relative_leadernumber_dbscan.json", "w") as fout:
                json.dump(self.relative_leadernumber_dbscan, fout)        
        elif clustering == "optics":
            self.relative_leadernumber_optics = {}    
            for company in self.all_companies:
                if company in total_leaders:
                    self.relative_leadernumber_optics[company] = leaders_count[company]/occurence_counts[company]
            with open("relative_leadernumber_optics.json", "w") as fout:
                json.dump(self.relative_leadernumber_optics, fout)        
        elif clustering == "hdbscan":
            self.relative_leadernumber_hdbscan = {}    
            for company in self.all_companies:
                if company in total_leaders:
                    self.relative_leadernumber_hdbscan[company] = leaders_count[company]/occurence_counts[company]
            with open("relative_leadernumber_hdbscan.json", "w") as fout:
                json.dump(self.relative_leadernumber_hdbscan, fout)        

        print("Done")        
        
        return clusters_dict3
    
    def leaderfollowerModel_test(self,file):
        csv_file = []
        # The data structure used here is a list of dicts where each dict is a row in the CSV file with header as key and attribute as value
        with open(file,encoding='utf8') as sqldbCSV:
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
                my_dict["district_name"] = line[2]
                my_dict["county_name"] = line[3]
                my_dict["region_name"] = line[4]
                my_dict["postcode"] = line[5]
                my_dict["company_number"] = line[6]
                my_dict["tenure"] = line[7]
                my_dict["date"] = line[8]
                my_dict["company_name"] = line[9]
                my_dict["title_number"] = line[10]
                my_dict["SIC"] = line[11]
                my_dict["category"] = line[12]
                my_dict["latitude"] = line[13]
                my_dict["longitude"] = line[14]
                my_dict["cluster"] = line[15]
                
                
                
                csv_file.append(my_dict)
        
        #with open('clusterings2.json', "r" ,encoding='utf8') as jsontext:
        #    clusters_dict = json.load(jsontext)
        
        clusters_dict2 = {}
        for dicts in csv_file:
            if dicts["cluster"] not in clusters_dict2:
                clusters_dict2[dicts["cluster"]] = [dicts["title_number"]]
            else:
                clusters_dict2[dicts["cluster"]].extend([dicts["title_number"]])    
        
        titles_dict = {}   
        self.category_dict = {}
        for dicts in csv_file:
            titles_dict[dicts["title_number"]] = (dicts["company_name"],dicts["date"], dicts["category"], dicts["county_name"], dicts["region_name"], dicts["cluster"])
            self.category_dict[dicts["company_name"]] = dicts["category"]
            
        clusters_dict3 = {}
        self.all_companies = []
        for k,v in clusters_dict2.items():
            companies = []
            for title in v:
                companies.append(titles_dict[title][0])
                self.all_companies.append(titles_dict[title][0])
            clusters_dict3[k] = list(set(companies))

        self.all_companies = set(self.all_companies)  
        print("There are {} unique companies in this dataset".format(len(self.all_companies)))    
        return clusters_dict3
        
    def generateFiles(self, dataset):
        
        # Food and Grocery:
        text_file =  open("Food_and_Grocery.txt", "w") 
        count2 = 0
        for t in dataset.values():
            count = 0
            for company in t:
                if self.category_dict[company] == "Food and Grocery":
                    text_file.write(company + ',')
                    count +=1
            if count >=1:        
                text_file.write('\n')
                count2 += 1
        text_file.close()
        print("Count is :", count2)
        '''
        text_file =  open("Food_and_Grocery.txt", "w") 
        for t in dataset.values():
            count = 0
            count2 = 0
            #if len([company for company in t if self.category_dict[company] == "Food and Grocery"]) > 1:
            for company in t:
                if self.category_dict[company] == "Food and Grocery":
                    count2+=1
            if count2 > 1:
                for company in t:
                    if self.category_dict[company] == "Food and Grocery":        
                        text_file.write(company + ',')
                        count +=1
                if count >=1:        
                    text_file.write('\n')      
        
        text_file.close()   
        ''' 
        # Restaurants and Cafes:
        text_file =  open("Restaurants_and_Cafes.txt", "w") 
        count2 = 0
        for t in dataset.values():
            count = 0
            for company in t:
                if self.category_dict[company] == "Restaurants and Cafes":
                    text_file.write(company + ',')
                    count +=1
            if count >=1:        
                text_file.write('\n')
                count2 += 1
                
        text_file.close()   
        print("Count is :", count2)
             
        # Department_Stores:
        text_file =  open("Department_Stores.txt", "w") 
        count2 = 0
        for t in dataset.values():
            count = 0
            for company in t:
                if self.category_dict[company] == "Department_Stores":
                    text_file.write(company + ',')
                    count +=1
            if count >=1:        
                text_file.write('\n') 
                count2 += 1
                
        text_file.close()     
        print("Count is :", count2)
           
        # Clothing and Accessories:
        text_file =  open("Clothing_and_Accessories.txt", "w") 
        count2 = 0
        for t in dataset.values():
            count = 0
            for company in t:
                if self.category_dict[company] == "Clothing and Accessories":
                    text_file.write(company + ',')
                    count +=1
            if count >=1:        
                text_file.write('\n')   
                count2 += 1                 
                
        text_file.close()    
        print("Count is :", count2)
            
        # Health and Cosmetics:
        text_file =  open("Health_and_Cosmetics.txt", "w") 
        count2 = 0
        for t in dataset.values():
            count = 0
            for company in t:
                if self.category_dict[company] == "Health and Cosmetics":
                    text_file.write(company + ',')
                    count +=1
            if count >=1:        
                text_file.write('\n')   
                count2 += 1  
                
        text_file.close()     
        print("Count is :", count2)
           
        # Electricals and Electronics:
        text_file =  open("Electricals_and_Electronics.txt", "w") 
        count2 = 0
        for t in dataset.values():
            count = 0
            for company in t:
                if self.category_dict[company] == "Electricals and Electronics":
                    text_file.write(company + ',')
                    count +=1
            if count >=1:        
                text_file.write('\n')  
                count2 += 1  
        text_file.close()     
        print("Count is :", count2)
           
        # Miscellaneous:
        text_file =  open("Miscellaneous.txt", "w") 
        count2 = 0
        for t in dataset.values():
            count = 0
            for company in t:
                if self.category_dict[company] == "Miscellaneous":
                    text_file.write(company + ',')
                    count +=1
            if count >=1:        
                text_file.write('\n')  
                count2 += 1
        text_file.close()   
        print("Count is :", count2)
             
        # Homeware and DIY:
        text_file =  open("Homeware_and_DIY.txt", "w") 
        count2 = 0
        for t in dataset.values():
            count = 0
            for company in t:
                if self.category_dict[company] == "Homeware and DIY":
                    text_file.write(company + ',')
                    count +=1
            if count >=1:        
                text_file.write('\n')  
                count2 += 1
        text_file.close()    
        print("Count is :", count2)           
                        
    
    def association_mining(self, dataset):
            
        companies_to_number = {}
        number_to_companies = {}
        for index,data in enumerate(self.all_companies):
            companies_to_number[data] = index
            number_to_companies[index] = data
        
            
        with open('companies_map.json', 'w') as fout: 
            json.dump(companies_to_number, fout)           
           
                 
        print("Producing itemsets...")        
        final_dataset = []        
        for v in dataset.values():
            temp = []
            #print(v)
            for company in v:
                temp.append(companies_to_number[company])
            final_dataset.append(temp)
            #print(temp)
        
        text_file =  open("dataset.txt", "w") 
        for t in final_dataset:
            for company in t:
                text_file.write(str(company) + ',')
            text_file.write('\n')  
        text_file.close() 
           
        text_file =  open("dataset2.txt", "w") 
        for t in dataset.values():
            for company in t:
                text_file.write(company + ',')
            text_file.write('\n')          
        text_file.close()        
        print("Finished...")  
        '''        
        itemsets = frequent_itemsets(final_dataset,30)  
        print("passed frequent itemsets")
        itemsets_dict = dict(itemsets)
        print(len(itemsets_dict))
        print("passed frequent itemsets dicts")
        #itemsets = list(itemsets)
        new_itemsets = []
        for k,v in itemsets_dict.items():
            # This is to convert the frozenset into a list
            my_list = list(k)
            temp = []
            for number in my_list:
                company = number_to_companies[number]
                temp.append(company)
            if len(temp) > 1:    
                new_itemsets.append((temp,v))
        
        print("Writing itemsets...")          
        text_file =  open("itemsets.txt", "w") 
        for t in new_itemsets:
            text_file.write(str(t))
            text_file.write('\n')    
                
        print("Finished...")         
        
        
        rules = association_rules(itemsets_dict, 0.6) 
        '''
        # This part is not needed due to the fact that it is incorporated in more_stats
        '''
        rules_list = list(rules)  
        
        new_rules = []
        for x in rules_list:
            # This is to convert the frozenset into a list
            my_list1 = list(x[0])
            my_list2 = list(x[1])
            temp1 = []
            temp2 = []
            for number in my_list1:
                company = number_to_companies[number]
                temp1.append(company)
            for number in my_list2:
                company = number_to_companies[number]
                temp2.append(company)
                    
            new_rules.append((temp1,temp2,x[2],x[3]))
        
        print("Writing association rules...")          
        text_file =  open("rules.txt", "w") 
        for t in new_rules:
            text_file.write(str(t))
            text_file.write('\n')    
                
        print("Finished...")  
        '''
        
        '''
        more_stats = list(rules_stats(rules, itemsets_dict, len(final_dataset)))
        new_stats = []
        for x in more_stats:
            # This is to convert the frozenset into a list
            my_list1 = list(x[0])
            my_list2 = list(x[1])
            temp1 = []
            temp2 = []
            for number in my_list1:
                company = number_to_companies[number]
                temp1.append(company)
            for number in my_list2:
                company = number_to_companies[number]
                temp2.append(company)
            #LIFT has to be greater than 1    
            if x[6] > 1:        
                new_stats.append((temp1,temp2,x[2],x[3],x[4],x[5],x[6],x[7]))
        
        print("Writing association rules...")          
        text_file =  open("rules.txt", "w") 
        for t in new_stats:
            text_file.write(str(t))
            text_file.write('\n')    
                
        print("Finished...")  
        '''
    def hdbscan_test(self, file):
        
        csv_file = []
        # The data structure used here is a list of dicts where each dict is a row in the CSV file with header as key and attribute as value
        with open(file,encoding='utf8') as sqldbCSV:
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
                my_dict["district_name"] = line[2]
                my_dict["county_name"] = line[3]
                my_dict["region_name"] = line[4]
                my_dict["postcode"] = line[5]
                my_dict["company_number"] = line[6]
                my_dict["tenure"] = line[7]
                my_dict["date"] = line[8]
                my_dict["company_name"] = line[9]
                my_dict["title_number"] = line[10]
                my_dict["SIC"] = line[11]
                my_dict["category"] = line[12]
                my_dict["latitude"] = line[13]
                my_dict["longitude"] = line[14]
                
                
                
                csv_file.append(my_dict)   
        
        coordinates = np.empty([len(csv_file),2],dtype=np.float64)
        for index, dicts in enumerate(csv_file):
            #lat = dicts["Coordinates"].split(",")[0].replace("(","")
            #lon = dicts["Coordinates"].split(",")[1].replace(")","")
            coordinates[index] = [float(dicts["latitude"]),float(dicts["longitude"])]
        
        min_cluster_size = [2,4,6,8,10,12,14,16,18,20]
        min_samples = [4]
        all_results = []
        final_silhouettes= []
        for minsamples in min_samples:
            for minclusters in min_cluster_size:
                print("Starting with min_cluster_size = {} and min_samples = {}".format(minclusters, minsamples))
                clusterer = hdbscan.HDBSCAN(min_cluster_size=minclusters,min_samples=minsamples, metric= 'haversine', memory='/tmp').fit(np.radians(coordinates))    
                #
                labels = clusterer.labels_
                #print(labels)
                # Number of clusters in labels, ignoring noise if present.
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                #clusters = pd.Series([coordinates[labels == n] for n in range(n_clusters)], index = range(n_clusters))
                
                print('Estimated number of clusters: %d' % n_clusters)
                #print(clusters)
                Noise = len([x for x in labels if x == -1])
                print("Noise = ", Noise)
                
                clusters_dict = {}
                for n in range(n_clusters):
                    clusters_dict[n] = coordinates[labels == n].tolist()
                    
                coordinates_to_clusters = {}
                for k,v in clusters_dict.items():
                    for coordinate in v:
                        x = tuple(coordinate)
                        coordinates_to_clusters[x] = k
                
                csv_file2 = copy.deepcopy(csv_file)
                for index, dicts in enumerate(csv_file2):
                    #lat = dicts["Coordinates"].split(",")[0].replace("(","")
                    #lon = dicts["Coordinates"].split(",")[1].replace(")","")
                    coordinate = (float(dicts["latitude"]),float(dicts["longitude"]))
                    if coordinate in coordinates_to_clusters:
                        csv_file2[index]["cluster"] = coordinates_to_clusters[coordinate]
                
                with open("final_Data_with_Range_Cluster_hdbscan.csv", "w", encoding = 'utf8', newline='') as csvfile:                
                    f = csv.writer(csvfile)     
                    f.writerow(["street_name", "town_name", "district_name", "county_name", "region_name", "postcode", "company_number", "tenure", "date", "company_name", "title_number", "SIC", "category", "latitude", "longitude", "cluster"])       
                    for dicts in csv_file2:
                        if "cluster" in dicts:
                            f.writerow([dicts["street_name"], dicts["town_name"], dicts["district_name"], dicts["county_name"], dicts["region_name"], dicts["postcode"], dicts["company_number"], dicts["tenure"], dicts["date"], dicts["company_name"], dicts["title_number"], dicts["SIC"], dicts["category"], dicts["latitude"], dicts["longitude"], dicts["cluster"]])
                    
                
                dataset = self.leaderfollowerModel_test("final_Data_with_Range_Cluster_hdbscan.csv")  
                
                print("Generating the dataset for R and generating the sub-datasets...")  
                text_file =  open("dataset2.txt", "w") 
                for t in dataset.values():
                    for company in t:
                        text_file.write(company + ',')
                    text_file.write('\n') 
                text_file.close()
                 
                self.generateFiles(dataset)
                print("Running the R code")
                retcode = subprocess.call(["C:/Program Files/R/R-3.3.3/bin/x64/Rscript.exe", "C:/Users/ahmed/OneDrive/Documents/R/script_for_python.R"])
                R_output = {"All_rules":[],"restaurant":[],"food":[],"department":[],"health":[],"clothes":[],"electrics":[],"misc":[],"home":[]}
                with open("rule_results.csv",encoding='utf8') as sqldbCSV:
                    
                    csvread = csv.reader(sqldbCSV, delimiter=",")
                    count = -1
                    for line in csvread:
                        #print(line)
                        count = count + 1
                        # Ignoring the header row
                        if count == 0:
                            continue
                        
                        R_output["All_rules"].append(line[0])
                        R_output["restaurant"].append(line[1])
                        R_output["food"].append(line[2])
                        R_output["department"].append(line[3])
                        R_output["health"].append(line[4])
                        R_output["clothes"].append(line[5])
                        R_output["electrics"].append(line[6])
                        R_output["misc"].append(line[7])
                        R_output["home"].append(line[8])
                
                print("R output for the rules generated by min_cluster_size = {} and min_samples = {} are".format(minclusters, minsamples))
                for k,v in R_output.items():
                    print(k , v)        
                
                my_silhouette = metrics.silhouette_score(coordinates, labels, metric=self.my_haversine, sample_size=20000)
                '''
                silhouettes = []
                for i in range(10):
                    silhouette = metrics.silhouette_score(coordinates, labels, metric=self.my_haversine, sample_size=10000)
                    silhouettes.append(silhouette)
                    
                final_silhouettes.append(np.mean(silhouettes))
                all_results.append(("(epsilon = {},sample = {}, clusters = {}, noise = {}, silhouettes = {})".format(epsilon, minsamples, n_clusters, Noise, np.mean(silhouettes)),R_output)) 
                print(final_silhouettes)
                '''
                final_silhouettes.append(my_silhouette)
                all_results.append(("(cluster = {},sample = {}, clusters = {}, noise = {}, silhouettes = {})".format(minclusters, minsamples, n_clusters, Noise, my_silhouette),R_output)) 
                print(final_silhouettes)
                
                #all_results.append(("(cluster = {},sample = {}, clusters = {})".format(minclusters, minsamples, n_clusters),R_output))    
        final_dict = dict(all_results)
        
        with open('final_results_hdbscan.json', 'w') as fout: 
            print("Dumping final results")
            json.dump(final_dict, fout)
        
       
        with open("final_results_hdbscan.json", "r" ,encoding='utf8') as jsontext:
            final_data = json.load(jsontext)
        for k,v in final_data.items():
            if len([x for x in v.values() if x != ["0", "0"]]) > 7:
                print(k)    
        
    def dbscan_test(self, file):
        
        csv_file = []
        # The data structure used here is a list of dicts where each dict is a row in the CSV file with header as key and attribute as value
        with open(file,encoding='utf8') as sqldbCSV:
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
                my_dict["district_name"] = line[2]
                my_dict["county_name"] = line[3]
                my_dict["region_name"] = line[4]
                my_dict["postcode"] = line[5]
                my_dict["company_number"] = line[6]
                my_dict["tenure"] = line[7]
                my_dict["date"] = line[8]
                my_dict["company_name"] = line[9]
                my_dict["title_number"] = line[10]
                my_dict["SIC"] = line[11]
                my_dict["category"] = line[12]
                my_dict["latitude"] = line[13]
                my_dict["longitude"] = line[14]
                
                
                
                csv_file.append(my_dict)   
        
        coordinates = np.empty([len(csv_file),2],dtype=np.float64)
        for index, dicts in enumerate(csv_file):
            #lat = dicts["Coordinates"].split(",")[0].replace("(","")
            #lon = dicts["Coordinates"].split(",")[1].replace(")","")
            coordinates[index] = [float(dicts["latitude"]),float(dicts["longitude"])]
        
        epsilons = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        min_samples = [4]
        all_results = []
        kms_per_radian = 6371.0088
        final_silhouettes = []
        #epsilon = 0.4
        for minsamples in min_samples:
            for epsilon in epsilons:
                print("Starting with epsilon = {} and min_samples = {}".format(epsilon, minsamples))
                #db = DBSCAN(eps=5/kms_per_radian, min_samples=10, metric=lambda X, Y: vincenty(X,Y).meters, n_jobs = -1).fit(coordinates) 
                db = DBSCAN(eps=epsilon/kms_per_radian, min_samples=minsamples, algorithm='ball_tree', metric='haversine', n_jobs = -1).fit(np.radians(coordinates)) 
                labels = db.labels_
                #print(labels)
                # Number of clusters in labels, ignoring noise if present.
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                #clusters = pd.Series([coordinates[labels == n] for n in range(n_clusters)], index = range(n_clusters))
                
                print('Estimated number of clusters: %d' % n_clusters)
                #print(clusters)
                Noise = len([x for x in labels if x == -1])
                print("Noise = ", Noise)
                
                
                clusters_dict = {}
                for n in range(n_clusters):
                    clusters_dict[n] = coordinates[labels == n].tolist()
                                    
                coordinates_to_clusters = {}
                for k,v in clusters_dict.items():
                    for coordinate in v:
                        x = tuple(coordinate)
                        coordinates_to_clusters[x] = k
                        
                
                # DEEP COPY BECAUSE CSV FILE IS A LIST OF DICTS IF I COPY THE LIST, THEN THE LIST AND ITS COPY BOTH REFERENCE THE SAME DICTS
                csv_file_new = copy.deepcopy(csv_file)
                
                for index, dicts in enumerate(csv_file_new):
                    #lat = dicts["Coordinates"].split(",")[0].replace("(","")
                    #lon = dicts["Coordinates"].split(",")[1].replace(")","")
                    coordinate = (float(dicts["latitude"]),float(dicts["longitude"]))
                    if coordinate in coordinates_to_clusters:
                        csv_file_new[index]["cluster"] = coordinates_to_clusters[coordinate]
                
                with open("final_Data_with_Range_Cluster.csv", "w", encoding = 'utf8', newline='') as hobba:                
                    f = csv.writer(hobba)     
                    f.writerow(["street_name", "town_name", "district_name", "county_name", "region_name", "postcode", "company_number", "tenure", "date", "company_name", "title_number", "SIC", "category", "latitude", "longitude", "cluster"])       
                    for dicts in csv_file_new:
                        if "cluster" in dicts:
                            f.writerow([dicts["street_name"], dicts["town_name"], dicts["district_name"], dicts["county_name"], dicts["region_name"], dicts["postcode"], dicts["company_number"], dicts["tenure"], dicts["date"], dicts["company_name"], dicts["title_number"], dicts["SIC"], dicts["category"], dicts["latitude"], dicts["longitude"], dicts["cluster"]])
                    
                dataset = self.leaderfollowerModel_test("final_Data_with_Range_Cluster.csv")  
                
                print("Generating the dataset for R and generating the sub-datasets...")  
                count = 0
                text_file =  open("dataset2.txt", "w") 
                for t in dataset.values():
                    for company in t:
                        text_file.write(company + ',')
                    text_file.write('\n') 
                    count +=1
                text_file.close()
                 
                self.generateFiles(dataset)
                print("Running the R code")
                retcode = subprocess.call(["C:/Program Files/R/R-3.3.3/bin/x64/Rscript.exe", "C:/Users/ahmed/OneDrive/Documents/R/script_for_python.R"])
                R_output = {"All_rules":[],"restaurant":[],"food":[],"department":[],"health":[],"clothes":[],"electrics":[],"misc":[],"home":[]}
                with open("rule_results.csv",encoding='utf8') as sqldbCSV:
                    
                    csvread = csv.reader(sqldbCSV, delimiter=",")
                    count = -1
                    for line in csvread:
                        #print(line)
                        count = count + 1
                        # Ignoring the header row
                        if count == 0:
                            continue
                        
                        R_output["All_rules"].append(line[0])
                        R_output["restaurant"].append(line[1])
                        R_output["food"].append(line[2])
                        R_output["department"].append(line[3])
                        R_output["health"].append(line[5])
                        R_output["clothes"].append(line[4])
                        R_output["electrics"].append(line[6])
                        R_output["misc"].append(line[7])
                        R_output["home"].append(line[8])
                
                print("R output for the rules generated by epsilon = {} and min_samples = {} are".format(epsilon, minsamples))
                for k,v in R_output.items():
                    print(k , v)      
                         
                my_silhouette = metrics.silhouette_score(coordinates, labels, metric=self.my_haversine, sample_size=20000)
                '''
                silhouettes = []
                for i in range(10):
                    silhouette = metrics.silhouette_score(coordinates, labels, metric=self.my_haversine, sample_size=10000)
                    silhouettes.append(silhouette)
                    
                final_silhouettes.append(np.mean(silhouettes))
                all_results.append(("(epsilon = {},sample = {}, clusters = {}, noise = {}, silhouettes = {})".format(epsilon, minsamples, n_clusters, Noise, np.mean(silhouettes)),R_output)) 
                print(final_silhouettes)
                '''
                final_silhouettes.append(my_silhouette)
                all_results.append(("(epsilon = {},sample = {}, clusters = {}, noise = {}, silhouettes = {})".format(epsilon, minsamples, n_clusters, Noise, my_silhouette),R_output)) 
                print(final_silhouettes)
                #print("Silhouette Score is : ", np.mean(silhouettes))
        final_dict = dict(all_results)
        
        
        with open('final_results_dbscan.json', 'w') as fout: 
            print("Dumping final results")
            json.dump(final_dict, fout)
        
       
        with open("final_results_dbscan.json", "r" ,encoding='utf8') as jsontext:
            final_data = json.load(jsontext)
        for k,v in final_data.items():
            if len([x for x in v.values() if x != ["0", "0"]]) > 7:
                print(k)
        
    def OPTICS_test(self, file):
        csv_file = []
        # The data structure used here is a list of dicts where each dict is a row in the CSV file with header as key and attribute as value
        with open(file,encoding='utf8') as sqldbCSV:
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
                my_dict["district_name"] = line[2]
                my_dict["county_name"] = line[3]
                my_dict["region_name"] = line[4]
                my_dict["postcode"] = line[5]
                my_dict["company_number"] = line[6]
                my_dict["tenure"] = line[7]
                my_dict["date"] = line[8]
                my_dict["company_name"] = line[9]
                my_dict["title_number"] = line[10]
                my_dict["SIC"] = line[11]
                my_dict["category"] = line[12]
                my_dict["latitude"] = line[13]
                my_dict["longitude"] = line[14]
                
                
                
                csv_file.append(my_dict)   
        
        coordinates = np.empty([len(csv_file),2],dtype=np.float64)
        for index, dicts in enumerate(csv_file):
            #lat = dicts["Coordinates"].split(",")[0].replace("(","")
            #lon = dicts["Coordinates"].split(",")[1].replace(")","")
            coordinates[index] = [float(dicts["latitude"]),float(dicts["longitude"])]
            
        bounds = [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
        min_samples = [4]
        all_results = []
        final_silhouettes = []
        #epsilon = 0.4
        for minsamples in min_samples:
            for bound in bounds:
                print("Starting with epsilon = {} and min_samples = {}".format(bound, minsamples))
                #db = DBSCAN(eps=5/kms_per_radian, min_samples=10, metric=lambda X, Y: vincenty(X,Y).meters, n_jobs = -1).fit(coordinates) 
                clust = OPTICS(max_bound=bound, min_samples = minsamples, algorithm = "ball_tree", metric=self.my_haversine).fit(coordinates)
                #5.0*6.0/6371.0088
                labels = clust.labels_
                # Number of clusters in labels, ignoring noise if present.
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                #clusters = pd.Series([coordinates[labels == n] for n in range(n_clusters)], index = range(n_clusters))
                
                print('Estimated number of clusters: %d' % n_clusters)
                #print(clusters)
                Noise = len([x for x in labels if x == -1])
                print("Noise = ", Noise)
                
                
                clusters_dict = {}
                for n in range(n_clusters):
                    clusters_dict[n] = coordinates[labels == n].tolist()
                                    
                coordinates_to_clusters = {}
                for k,v in clusters_dict.items():
                    for coordinate in v:
                        x = tuple(coordinate)
                        coordinates_to_clusters[x] = k
                        
                
                # DEEP COPY BECAUSE CSV FILE IS A LIST OF DICTS IF I COPY THE LIST, THEN THE LIST AND ITS COPY BOTH REFERENCE THE SAME DICTS
                csv_file_new = copy.deepcopy(csv_file)
                
                for index, dicts in enumerate(csv_file_new):
                    #lat = dicts["Coordinates"].split(",")[0].replace("(","")
                    #lon = dicts["Coordinates"].split(",")[1].replace(")","")
                    coordinate = (float(dicts["latitude"]),float(dicts["longitude"]))
                    if coordinate in coordinates_to_clusters:
                        csv_file_new[index]["cluster"] = coordinates_to_clusters[coordinate]
                
                with open("final_Data_with_Range_Cluster_optics.csv", "w", encoding = 'utf8', newline='') as hobba:                
                    f = csv.writer(hobba)     
                    f.writerow(["street_name", "town_name", "district_name", "county_name", "region_name", "postcode", "company_number", "tenure", "date", "company_name", "title_number", "SIC", "category", "latitude", "longitude", "cluster"])       
                    for dicts in csv_file_new:
                        if "cluster" in dicts:
                            f.writerow([dicts["street_name"], dicts["town_name"], dicts["district_name"], dicts["county_name"], dicts["region_name"], dicts["postcode"], dicts["company_number"], dicts["tenure"], dicts["date"], dicts["company_name"], dicts["title_number"], dicts["SIC"], dicts["category"], dicts["latitude"], dicts["longitude"], dicts["cluster"]])
                    
                dataset = self.leaderfollowerModel_test("final_Data_with_Range_Cluster_optics.csv")  
                
                print("Generating the dataset for R and generating the sub-datasets...")  
                count = 0
                text_file =  open("dataset2.txt", "w") 
                for t in dataset.values():
                    for company in t:
                        text_file.write(company + ',')
                    text_file.write('\n') 
                    count +=1
                text_file.close()
                 
                self.generateFiles(dataset)
                print("Running the R code")
                retcode = subprocess.call(["C:/Program Files/R/R-3.3.3/bin/x64/Rscript.exe", "C:/Users/ahmed/OneDrive/Documents/R/script_for_python.R"])
                R_output = {"All_rules":[],"restaurant":[],"food":[],"department":[],"health":[],"clothes":[],"electrics":[],"misc":[],"home":[]}
                with open("rule_results.csv",encoding='utf8') as sqldbCSV:
                    
                    csvread = csv.reader(sqldbCSV, delimiter=",")
                    count = -1
                    for line in csvread:
                        #print(line)
                        count = count + 1
                        # Ignoring the header row
                        if count == 0:
                            continue
                        
                        R_output["All_rules"].append(line[0])
                        R_output["restaurant"].append(line[1])
                        R_output["food"].append(line[2])
                        R_output["department"].append(line[3])
                        R_output["health"].append(line[5])
                        R_output["clothes"].append(line[4])
                        R_output["electrics"].append(line[6])
                        R_output["misc"].append(line[7])
                        R_output["home"].append(line[8])
                
                print("R output for the rules generated by epsilon = {} and min_samples = {} are".format(bound, minsamples))
                for k,v in R_output.items():
                    print(k , v)      
                         
                my_silhouette = metrics.silhouette_score(coordinates, labels, metric=self.my_haversine, sample_size=20000)
                '''
                silhouettes = []
                for i in range(2):
                    silhouette = metrics.silhouette_score(coordinates, labels, metric=self.my_haversine, sample_size=10000)
                    silhouettes.append(silhouette)
                    
                final_silhouettes.append(np.mean(silhouettes))
                all_results.append(("(epsilon = {},sample = {}, clusters = {}, noise = {}, silhouettes = {})".format(bound, minsamples, n_clusters, Noise, np.mean(silhouettes)),R_output)) 
                print(final_silhouettes)
                '''
                
                final_silhouettes.append(my_silhouette)
                all_results.append(("(epsilon = {},sample = {}, clusters = {}, noise = {}, silhouettes = {})".format(bound, minsamples, n_clusters, Noise, my_silhouette),R_output)) 
                print(final_silhouettes)
                #print("Silhouette Score is : ", np.mean(silhouettes))
                
        final_dict = dict(all_results)
        
        
        with open('final_results_optics.json', 'w') as fout: 
            print("Dumping final results")
            json.dump(final_dict, fout)
        
       
        with open("final_results_optics.json", "r" ,encoding='utf8') as jsontext:
            final_data = json.load(jsontext)
        for k,v in final_data.items():
            if len([x for x in v.values() if x != ["0", "0"]]) > 7:
                print(k)
       
    def spatio_temporal_rules_majority_vote(self):
        
        '''
        #used for testing!!!
        with open("relative_leadernumber_dbscan.json","r", encoding = 'utf8') as websites:
            self.relative_leadernumber_dbscan = json.load(websites)    
        with open("relative_leadernumber_optics.json","r", encoding = 'utf8') as websites:
            self.relative_leadernumber_optics = json.load(websites)    
        with open("relative_leadernumber_hdbscan.json","r", encoding = 'utf8') as websites:
            self.relative_leadernumber_hdbscan = json.load(websites) 
        '''          
        self.relative_leadernumber_majority = {}
        
        majority = list(self.relative_leadernumber_dbscan.keys()) + list(self.relative_leadernumber_optics.keys()) + list(self.relative_leadernumber_hdbscan.keys())
        majority_companies = [item for item, count in Counter(majority).items() if count > 1]
        
        for comp in majority_companies:
            if comp in self.relative_leadernumber_dbscan and comp in self.relative_leadernumber_optics and comp in self.relative_leadernumber_hdbscan:
                self.relative_leadernumber_majority[comp] = np.mean([self.relative_leadernumber_dbscan[comp],self.relative_leadernumber_optics[comp],self.relative_leadernumber_hdbscan[comp]])
            elif comp in self.relative_leadernumber_dbscan and comp in self.relative_leadernumber_optics:
                self.relative_leadernumber_majority[comp] = np.mean([self.relative_leadernumber_dbscan[comp],self.relative_leadernumber_optics[comp]])
            elif comp in self.relative_leadernumber_dbscan and comp in self.relative_leadernumber_hdbscan:
                self.relative_leadernumber_majority[comp] = np.mean([self.relative_leadernumber_dbscan[comp],self.relative_leadernumber_hdbscan[comp]])   
            elif comp in self.relative_leadernumber_optics and comp in self.relative_leadernumber_hdbscan:
                self.relative_leadernumber_majority[comp] = np.mean([self.relative_leadernumber_optics[comp],self.relative_leadernumber_hdbscan[comp]]) 
                    
        with open("relative_leadernumber_majority.json", "w") as fout:
            json.dump(self.relative_leadernumber_majority, fout) 
        clothes_1 = {}
        clothes_2 = {}
        clothes_3 = {}
        department_1 = {}
        department_2 = {}
        department_3 = {}
        electrics_1 = {}
        electrics_2 = {}
        electrics_3 = {}
        food_1 = {}
        food_2 = {}
        food_3 = {}
        health_1 = {}
        health_2 = {}
        health_3 = {}
        home_1 = {}
        home_2 = {}
        home_3 = {}
        misc_1 = {}
        misc_2 = {}
        misc_3 = {}
        restaurant_1 = {}
        restaurant_2 = {}
        restaurant_3 = {}
        total_1 = {}
        total_2 = {}
        total_3 = {}
        '''
        common_clothes = []
        common_department = []
        common_electrics = []
        common_food = []
        common_health = []
        common_home = []
        common_misc = []
        common_restaurant = []
        common_total = []
        majority_clothes = []
        majority_department = []
        majority_electrics = []
        majority_food = []
        majority_health = []
        majority_home = []
        majority_misc = []
        majority_restaurant = []
        majority_total = []
        '''
        for filename in os.listdir("C:/Users/ahmed/OneDrive/Documents/Masters Courses/Dissertation/python_code/dbscan_rules"):
            print(filename)
            file = os.path.join("C:/Users/ahmed/OneDrive/Documents/Masters Courses/Dissertation/python_code/dbscan_rules", filename)
            with open(file, "r" ,encoding='utf8') as filestream:
                count = -1
                for line in filestream:
                    count += 1
                    if count == 0:
                        continue
                    splitline = re.findall('\"(:?.+?)\"',line)
                    LHS = splitline[1].replace("{","")
                    LHS = LHS.replace("}","")
                    RHS = splitline[2].replace("{","")
                    RHS = RHS.replace("}","")
                    RHS = (RHS,self.relative_leadernumber_majority[RHS]) if RHS in self.relative_leadernumber_majority else (RHS,0)
                    splitline = re.findall('\d+\.?\d*',line)
                    #print(splitline)
                    support = splitline[1]
                    confidence = splitline[2]
                    lift = splitline[3]
                    if "," in LHS:
                        LHS = LHS.split(",")
                        for index,comp in enumerate(LHS):
                            if comp in self.relative_leadernumber_majority:
                                LHS[index] = (LHS[index], self.relative_leadernumber_majority[comp])
                            else:
                                LHS[index] = (LHS[index], 0)           
                    else:
                        LHS = [LHS]
                        if LHS[0] in self.relative_leadernumber_majority:
                            LHS[0] = (LHS[0], self.relative_leadernumber_majority[LHS[0]])      
                        else:
                            LHS[0] = (LHS[0], 0)   
                        
                    
                    if any(x[0] in self.relative_leadernumber_majority for x in LHS):   
                        if(all(x[1] > RHS[1] for x in LHS)):
                            LHS.sort(key = lambda x:x[1], reverse=True)
                            if filename == "R_Rules_clothes.txt": 
                                clothes_1[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_department.txt":
                                department_1[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_electrics.txt":
                                electrics_1[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_food.txt":
                                food_1[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_health.txt":
                                health_1[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_home.txt":
                                home_1[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_misc.txt":
                                misc_1[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_restaurant.txt":
                                restaurant_1[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_pruned.txt":
                                total_1[(str(LHS),str(RHS))] = (support,confidence,lift)                            
                        
                            
        for filename in os.listdir("C:/Users/ahmed/OneDrive/Documents/Masters Courses/Dissertation/python_code/optics_rules"):   
            print(filename)     
            file = os.path.join("C:/Users/ahmed/OneDrive/Documents/Masters Courses/Dissertation/python_code/optics_rules", filename)
            with open(file, "r" ,encoding='utf8') as filestream:
                count = -1
                for line in filestream:
                    count += 1
                    if count == 0:
                        continue
                    splitline = re.findall('\"(:?.+?)\"',line)
                    LHS = splitline[1].replace("{","")
                    LHS = LHS.replace("}","")
                    RHS = splitline[2].replace("{","")
                    RHS = RHS.replace("}","")
                    RHS = (RHS,self.relative_leadernumber_majority[RHS]) if RHS in self.relative_leadernumber_majority else (RHS,0)
                    splitline = re.findall('\d+\.?\d*',line)
                    #print(splitline)
                    support = splitline[1]
                    confidence = splitline[2]
                    lift = splitline[3]
                    if "," in LHS:
                        LHS = LHS.split(",")
                        for index,comp in enumerate(LHS):
                            if comp in self.relative_leadernumber_majority:
                                LHS[index] = (LHS[index], self.relative_leadernumber_majority[comp])
                            else:
                                LHS[index] = (LHS[index], 0)           
                    else:
                        LHS = [LHS]
                        if LHS[0] in self.relative_leadernumber_majority:
                            LHS[0] = (LHS[0], self.relative_leadernumber_majority[LHS[0]])      
                        else:
                            LHS[0] = (LHS[0], 0)   
                    
                    if any(x[0] in self.relative_leadernumber_majority for x in LHS):
                        if(all(x[1] > RHS[1] for x in LHS)):
                            LHS.sort(key = lambda x:x[1], reverse=True)
                            if filename == "R_Rules_clothes.txt":    
                                clothes_2[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_department.txt":
                                department_2[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_electrics.txt":
                                electrics_2[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_food.txt":
                                food_2[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_health.txt":
                                health_2[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_home.txt":
                                home_2[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_misc.txt":
                                misc_2[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_restaurant.txt":
                                restaurant_2[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_pruned.txt":
                                total_2[(str(LHS),str(RHS))] = (support,confidence,lift)      
        
        for filename in os.listdir("C:/Users/ahmed/OneDrive/Documents/Masters Courses/Dissertation/python_code/hdbscan_rules"):     
            print(filename)       
            file = os.path.join("C:/Users/ahmed/OneDrive/Documents/Masters Courses/Dissertation/python_code/hdbscan_rules", filename)
            with open(file, "r" ,encoding='utf8') as filestream:
                count = -1
                for line in filestream:
                    count += 1
                    if count == 0:
                        continue
                    splitline = re.findall('\"(:?.+?)\"',line)
                    LHS = splitline[1].replace("{","")
                    LHS = LHS.replace("}","")
                    RHS = splitline[2].replace("{","")
                    RHS = RHS.replace("}","")
                    RHS = (RHS,self.relative_leadernumber_majority[RHS]) if RHS in self.relative_leadernumber_majority else (RHS,0)
                    splitline = re.findall('\d+\.?\d*',line)
                    #print(splitline)
                    support = splitline[1]
                    confidence = splitline[2]
                    lift = splitline[3]
                    if "," in LHS:
                        LHS = LHS.split(",")
                        for index,comp in enumerate(LHS):
                            if comp in self.relative_leadernumber_majority:
                                LHS[index] = (LHS[index], self.relative_leadernumber_majority[comp])
                            else:
                                LHS[index] = (LHS[index], 0)           
                    else:
                        LHS = [LHS]
                        if LHS[0] in self.relative_leadernumber_majority:
                            LHS[0] = (LHS[0], self.relative_leadernumber_majority[LHS[0]])      
                        else:
                            LHS[0] = (LHS[0], 0)   
                                
                    
                    if any(x[0] in self.relative_leadernumber_majority for x in LHS):  
                        if(all(x[1] > RHS[1] for x in LHS)):  
                            LHS.sort(key = lambda x:x[1], reverse=True)
                            if filename == "R_Rules_clothes.txt":    
                                clothes_3[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_department.txt":
                                department_3[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_electrics.txt":
                                electrics_3[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_food.txt":
                                food_3[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_health.txt":
                                health_3[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_home.txt":
                                home_3[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_misc.txt":
                                misc_3[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_restaurant.txt":
                                restaurant_3[(str(LHS),str(RHS))] = (support,confidence,lift)
                            elif filename == "R_Rules_pruned.txt":
                                total_3[(str(LHS),str(RHS))] = (support,confidence,lift)  
        
        common_clothes = list(clothes_1.keys()) + list(clothes_2.keys()) + list(clothes_3.keys())
        common_department = list(department_1.keys()) + list(department_2.keys()) + list(department_3.keys())
        common_electrics = list(electrics_1.keys()) + list(electrics_2.keys()) + list(electrics_3.keys())
        common_food = list(food_1.keys()) + list(food_2.keys()) + list(food_3.keys())
        common_health = list(health_1.keys()) + list(health_2.keys()) + list(health_3.keys())
        common_home = list(home_1.keys()) + list(home_2.keys()) + list(home_3.keys())
        common_misc = list(misc_1.keys()) + list(misc_2.keys()) + list(misc_3.keys())
        common_restaurant = list(restaurant_1.keys()) + list(restaurant_2.keys()) + list(restaurant_3.keys())
        common_total = list(total_1.keys()) + list(total_2.keys()) + list(total_3.keys())
        
        
        majority_clothes = [item for item, count in Counter(common_clothes).items() if count > 1]
        majority_department = [item for item, count in Counter(common_department).items() if count > 1]
        majority_electrics = [item for item, count in Counter(common_electrics).items() if count > 1]
        majority_food = [item for item, count in Counter(common_food).items() if count > 1]
        majority_health = [item for item, count in Counter(common_health).items() if count > 1]
        majority_home = [item for item, count in Counter(common_home).items() if count > 1]
        majority_misc = [item for item, count in Counter(common_misc).items() if count > 1]
        majority_restaurant = [item for item, count in Counter(common_restaurant).items() if count > 1]
        temp = majority_clothes + majority_department + majority_electrics + majority_food + majority_health + majority_home + majority_misc + majority_restaurant
        majority_total = [item for item, count in Counter(common_total).items() if count > 1 and item not in temp]
        majority_clothes.sort(key= lambda x:len(x[0]), reverse=False)
        majority_department.sort(key= lambda x:len(x[0]), reverse=False)
        majority_electrics.sort(key= lambda x:len(x[0]), reverse=False)
        majority_food.sort(key= lambda x:len(x[0]), reverse=False)
        majority_health.sort(key= lambda x:len(x[0]), reverse=False)
        majority_home.sort(key= lambda x:len(x[0]), reverse=False)
        majority_misc.sort(key= lambda x:len(x[0]), reverse=False)
        majority_restaurant.sort(key= lambda x:len(x[0]), reverse=False)
        majority_total.sort(key= lambda x:len(x[0]), reverse=False)
        print("Common clothes rules : ", len(common_clothes), " and majority is : ", len(majority_clothes))  
        print("Common department rules : ", len(common_department), " and majority is : ", len(majority_department))
        print("Common electrics rules : ", len(common_electrics), " and majority is : ", len(majority_electrics))
        print("Common food rules : ", len(common_food), " and majority is : ", len(majority_food))
        print("Common health rules : ", len(common_health), " and majority is : ", len(majority_health))
        print("Common home rules : ", len(common_home), " and majority is : ", len(majority_home))
        print("Common misc rules : ", len(common_misc), " and majority is : ", len(majority_misc))
        print("Common restaurant rules : ", len(common_restaurant), " and majority is : ", len(majority_restaurant))
        print("Common total rules : ", len(common_total), " and majority is : ", len(majority_total)) 
        
        text_file =  open("majority_rules_clothes.txt", "w") 
        for rule in majority_clothes:
            LHS = [x[0] for x in literal_eval(rule[0])]
            RHS = literal_eval(rule[1])[0]
            if rule in clothes_1 and rule in clothes_2:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " DBSCAN: " + str(clothes_1[rule]) + " OPTICS: " + str(clothes_2[rule]))
            elif rule in clothes_1 and rule in clothes_3:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " DBSCAN: " + str(clothes_1[rule]) + " HDBSCAN: " + str(clothes_3[rule]))
            else:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " OPTICS: " + str(clothes_2[rule]) + " HDBSCAN: " + str(clothes_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        f = csv.writer(open("majority_rules_clothes.csv", "w", encoding = 'utf8', newline=''))     
        f.writerow(["Antecedent", "Consequent", "Support", "Confidence", "Lift"])       
        for rule in majority_clothes:
            LHS = [x[0] for x in literal_eval(rule[0])]
            RHS = literal_eval(rule[1])[0]
            support = clothes_1[rule][0] if rule in clothes_1 else clothes_2[rule][0]
            confidence = clothes_1[rule][1] if rule in clothes_1 else clothes_2[rule][1]
            lift = clothes_1[rule][2] if rule in clothes_1 else clothes_2[rule][2]
            f.writerow([str(LHS).replace(",","AND"), RHS, support, confidence, lift])
        
        text_file =  open("majority_rules_department.txt", "w") 
        for rule in majority_department:
            LHS = [x[0] for x in literal_eval(rule[0])]
            RHS = literal_eval(rule[1])[0]
            if rule in department_1 and rule in department_2:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " DBSCAN: " + str(department_1[rule]) + " OPTICS: " + str(department_2[rule]))
            elif rule in department_1 and rule in department_3:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " DBSCAN: " + str(department_1[rule]) + " HDBSCAN: " + str(department_3[rule]))
            else:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " OPTICS: " + str(department_2[rule]) + " HDBSCAN: " + str(department_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        f = csv.writer(open("majority_rules_department.csv", "w", encoding = 'utf8', newline=''))     
        f.writerow(["Antecedent", "Consequent", "Support", "Confidence", "Lift"])       
        for rule in majority_department:
            LHS = [x[0] for x in literal_eval(rule[0])]
            RHS = literal_eval(rule[1])[0]
            support = department_1[rule][0] if rule in department_1 else department_2[rule][0]
            confidence = department_1[rule][1] if rule in clothes_1 else department_2[rule][1]
            lift = department_1[rule][2] if rule in department_1 else department_2[rule][2]
            f.writerow([str(LHS).replace(",","AND"), RHS, support, confidence, lift])
            
        text_file =  open("majority_rules_electrics.txt", "w") 
        for rule in majority_electrics:
            LHS = [x[0] for x in literal_eval(rule[0])]
            RHS = literal_eval(rule[1])[0]            
            if rule in electrics_1 and rule in electrics_2:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " DBSCAN: " + str(electrics_1[rule]) + " OPTICS: " + str(electrics_2[rule]))
            elif rule in electrics_1 and rule in electrics_3:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " DBSCAN: " + str(electrics_1[rule]) + " HDBSCAN: " + str(electrics_3[rule]))
            else:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " OPTICS: " + str(electrics_2[rule]) + " HDBSCAN: " + str(electrics_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        f = csv.writer(open("majority_rules_electrics.csv", "w", encoding = 'utf8', newline=''))     
        f.writerow(["Antecedent", "Consequent", "Support", "Confidence", "Lift"])       
        for rule in majority_electrics:
            LHS = [x[0] for x in literal_eval(rule[0])]
            RHS = literal_eval(rule[1])[0]
            support = electrics_1[rule][0] if rule in electrics_1 else electrics_2[rule][0]
            confidence = electrics_1[rule][1] if rule in electrics_1 else electrics_2[rule][1]
            lift = electrics_1[rule][2] if rule in electrics_1 else electrics_2[rule][2]
            f.writerow([str(LHS).replace(",","AND"), RHS, support, confidence, lift])
            
        text_file =  open("majority_rules_food.txt", "w") 
        for rule in majority_food:
            LHS = [x[0] for x in literal_eval(rule[0])]
            RHS = literal_eval(rule[1])[0]            
            if rule in food_1 and rule in food_2:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " DBSCAN: " + str(food_1[rule]) + " OPTICS: " + str(food_2[rule]))
            elif rule in food_1 and rule in food_3:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " DBSCAN: " + str(food_1[rule]) + " HDBSCAN: " + str(food_3[rule]))
            else:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " OPTICS: " + str(food_2[rule]) + " HDBSCAN: " + str(food_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        f = csv.writer(open("majority_rules_food.csv", "w", encoding = 'utf8', newline=''))     
        f.writerow(["Antecedent", "Consequent", "Support", "Confidence", "Lift"])       
        for rule in majority_food:
            LHS = [x[0] for x in literal_eval(rule[0])]
            RHS = literal_eval(rule[1])[0]
            support = food_1[rule][0] if rule in food_1 else food_2[rule][0]
            confidence = food_1[rule][1] if rule in food_1 else food_2[rule][1]
            lift = food_1[rule][2] if rule in food_1 else food_2[rule][2]
            f.writerow([str(LHS).replace(",","AND"), RHS, support, confidence, lift])
            
        text_file =  open("majority_rules_health.txt", "w") 
        for rule in majority_health:
            LHS = [x[0] for x in literal_eval(rule[0])]
            RHS = literal_eval(rule[1])[0]            
            if rule in health_1 and rule in health_2:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " DBSCAN: " + str(health_1[rule]) + " OPTICS: " + str(health_2[rule]))
            elif rule in health_1 and rule in health_3:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " DBSCAN: " + str(health_1[rule]) + " HDBSCAN: " + str(health_3[rule]))
            else:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " OPTICS: " + str(health_2[rule]) + " HDBSCAN: " + str(health_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        f = csv.writer(open("majority_rules_health.csv", "w", encoding = 'utf8', newline=''))     
        f.writerow(["Antecedent", "Consequent", "Support", "Confidence", "Lift"])       
        for rule in majority_health:
            LHS = [x[0] for x in literal_eval(rule[0])]
            RHS = literal_eval(rule[1])[0]
            support = health_1[rule][0] if rule in health_1 else health_2[rule][0]
            confidence = health_1[rule][1] if rule in health_1 else health_2[rule][1]
            lift = health_1[rule][2] if rule in health_1 else health_2[rule][2]
            f.writerow([str(LHS).replace(",","AND"), RHS, support, confidence, lift])
            
        text_file =  open("majority_rules_home.txt", "w") 
        for rule in majority_home:
            LHS = [x[0] for x in literal_eval(rule[0])]
            RHS = literal_eval(rule[1])[0]            
            if rule in home_1 and rule in home_2:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " DBSCAN: " + str(home_1[rule]) + " OPTICS: " + str(home_2[rule]))
            elif rule in home_1 and rule in home_3:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " DBSCAN: " + str(home_1[rule]) + " HDBSCAN: " + str(home_3[rule]))
            else:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " OPTICS: " + str(home_2[rule]) + " HDBSCAN: " + str(home_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        f = csv.writer(open("majority_rules_home.csv", "w", encoding = 'utf8', newline=''))     
        f.writerow(["Antecedent", "Consequent", "Support", "Confidence", "Lift"])       
        for rule in majority_home:
            LHS = [x[0] for x in literal_eval(rule[0])]
            RHS = literal_eval(rule[1])[0]
            support = home_1[rule][0] if rule in home_1 else home_2[rule][0]
            confidence = home_1[rule][1] if rule in home_1 else home_2[rule][1]
            lift = home_1[rule][2] if rule in home_1 else home_2[rule][2]
            f.writerow([str(LHS).replace(",","AND"), RHS, support, confidence, lift])
            
        text_file =  open("majority_rules_misc.txt", "w") 
        for rule in majority_misc:
            LHS = [x[0] for x in literal_eval(rule[0])]
            RHS = literal_eval(rule[1])[0]            
            if rule in misc_1 and rule in misc_2:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " DBSCAN: " + str(misc_1[rule]) + " OPTICS: " + str(misc_2[rule]))
            elif rule in misc_1 and rule in misc_3:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " DBSCAN: " + str(misc_1[rule]) + " HDBSCAN: " + str(misc_3[rule]))
            else:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " OPTICS: " + str(misc_2[rule]) + " HDBSCAN: " + str(misc_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        f = csv.writer(open("majority_rules_misc.csv", "w", encoding = 'utf8', newline=''))     
        f.writerow(["Antecedent", "Consequent", "Support", "Confidence", "Lift"])       
        for rule in majority_misc:
            LHS = [x[0] for x in literal_eval(rule[0])]
            RHS = literal_eval(rule[1])[0]
            support = misc_1[rule][0] if rule in misc_1 else misc_2[rule][0]
            confidence = misc_1[rule][1] if rule in misc_1 else misc_2[rule][1]
            lift = misc_1[rule][2] if rule in misc_1 else misc_2[rule][2]
            f.writerow([str(LHS).replace(",","AND"), RHS, support, confidence, lift])
            
        text_file =  open("majority_rules_restaurant.txt", "w") 
        for rule in majority_restaurant:
            LHS = [x[0] for x in literal_eval(rule[0])]
            RHS = literal_eval(rule[1])[0]            
            if rule in restaurant_1 and rule in restaurant_2:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " DBSCAN: " + str(restaurant_1[rule]) + " OPTICS: " + str(restaurant_2[rule]))
            elif rule in restaurant_1 and rule in restaurant_3:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " DBSCAN: " + str(restaurant_1[rule]) + " HDBSCAN: " + str(restaurant_3[rule]))
            else:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " OPTICS: " + str(restaurant_2[rule]) + " HDBSCAN: " + str(restaurant_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        f = csv.writer(open("majority_rules_restaurant.csv", "w", encoding = 'utf8', newline=''))     
        f.writerow(["Antecedent", "Consequent", "Support", "Confidence", "Lift"])       
        for rule in majority_restaurant:
            LHS = [x[0] for x in literal_eval(rule[0])]
            RHS = literal_eval(rule[1])[0]
            support = restaurant_1[rule][0] if rule in restaurant_1 else restaurant_2[rule][0]
            confidence = restaurant_1[rule][1] if rule in restaurant_1 else restaurant_2[rule][1]
            lift = restaurant_1[rule][2] if rule in restaurant_1 else restaurant_2[rule][2]
            f.writerow([str(LHS).replace(",","AND"), RHS, support, confidence, lift])
            
        text_file =  open("majority_rules_total.txt", "w") 
        for rule in majority_total:
            LHS = [x[0] for x in literal_eval(rule[0])]
            RHS = literal_eval(rule[1])[0]            
            if rule in total_1 and rule in total_2:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " DBSCAN: " + str(total_1[rule]) + " OPTICS: " + str(total_2[rule]))
            elif rule in total_1 and rule in total_3:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " DBSCAN: " + str(total_1[rule]) + " HDBSCAN: " + str(total_3[rule]))
            else:
                text_file.write(str(LHS) + ' ==> ' + str(RHS) + " OPTICS: " + str(total_2[rule]) + " HDBSCAN: " + str(total_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        f = csv.writer(open("majority_rules_total.csv", "w", encoding = 'utf8', newline=''))     
        f.writerow(["Antecedent", "Consequent", "Support", "Confidence", "Lift"])       
        for rule in majority_total:
            LHS = [x[0] for x in literal_eval(rule[0])]
            RHS = literal_eval(rule[1])[0]
            support = total_1[rule][0] if rule in total_1 else total_2[rule][0]
            confidence = total_1[rule][1] if rule in total_1 else total_2[rule][1]
            lift = total_1[rule][2] if rule in total_1 else total_2[rule][2]
            f.writerow([str(LHS).replace(",","AND"), RHS, support, confidence, lift])
            
        '''        
        for rule in clothes_1.keys():
            if rule in clothes_2 and rule in clothes_3:
                common_clothes.append(rule)        
                #print(rule)
            elif rule in clothes_2 or rule in clothes_3:
                majority_clothes.append(rule)    
                
        for rule in department_1.keys():
            if rule in department_2 and rule in department_3:
                common_department.append(rule)     
            elif rule in department_2 or rule in department_3:
                majority_department.append(rule)       
                #print(rule)
        
        for rule in electrics_1.keys():
            if rule in electrics_2 and rule in electrics_3:
                common_electrics.append(rule)        
                #print(rule)
            elif rule in electrics_2 or rule in electrics_3:
                majority_electrics.append(rule)    
        
        for rule in food_1.keys():
            if rule in food_2 and rule in food_3:
                common_food.append(rule)        
                #print(rule)
            elif rule in food_2 or rule in food_3:
                majority_food.append(rule)    
        
        for rule in health_1.keys():
            if rule in health_2 and rule in health_3:
                common_health.append(rule)        
                #print(rule)
            elif rule in health_2 or rule in health_3:
                majority_health.append(rule)    
        
        for rule in home_1.keys():
            if rule in home_2 and rule in home_3:
                common_home.append(rule)        
                #print(rule)
            elif rule in home_2 or rule in home_3:
                majority_home.append(rule)    
                
        for rule in misc_1.keys():
            if rule in misc_2 and rule in misc_3:
                common_misc.append(rule)        
                #print(rule)
            elif rule in misc_2 or rule in misc_3:
                majority_misc.append(rule)    
                
        for rule in restaurant_1.keys():
            if rule in restaurant_2 and rule in restaurant_3:
                common_restaurant.append(rule)        
                #print(rule)        
            elif rule in restaurant_2 or rule in restaurant_3:
                majority_restaurant.append(rule)    
                
        for rule in total_1.keys():
            if rule in total_2 and rule in total_3:
                common_total.append(rule)        
                #print(rule)
            elif rule in total_2 or rule in total_3:
                majority_total.append(rule)   
                
                
                
        for rule in clothes_2.keys():
            if rule in clothes_3:
                majority_clothes.append(rule) 
                
        for rule in department_2.keys():
            if rule in department_3:
                majority_department.append(rule) 
        
        for rule in electrics_2.keys():
            if rule in electrics_3:
                majority_electrics.append(rule) 
        
        for rule in food_2.keys():
            if rule in food_3:
                majority_food.append(rule) 
                
        for rule in health_2.keys():
            if rule in health_3:
                majority_health.append(rule) 
                
        for rule in home_2.keys():
            if rule in home_3:
                majority_home.append(rule) 
                
        for rule in misc_2.keys():
            if rule in misc_3:
                majority_misc.append(rule) 
                
        for rule in restaurant_2.keys():
            if rule in restaurant_3:
                majority_restaurant.append(rule) 
                
        for rule in total_2.keys():
            if rule in total_3:
                majority_total.append(rule)                                                                        
                
        majority_clothes = set(majority_clothes) - set(common_clothes)
        majority_department = set(majority_department) - set(common_department)
        majority_electrics = set(majority_electrics) - set(common_electrics)
        majority_food = set(majority_food) - set(common_food)
        majority_health = set(majority_health) - set(common_health)
        majority_home = set(majority_home) - set(common_home)
        majority_misc = set(majority_misc) - set(common_misc)
        majority_restaurant = set(majority_restaurant) - set(common_restaurant)
        majority_total = set(majority_total) - set(common_total)                                                                     
                
        print("Common clothes rules : ", len(common_clothes), " and majority is : ", len(majority_clothes))  
        print("Common department rules : ", len(common_department), " and majority is : ", len(majority_department))
        print("Common electrics rules : ", len(common_electrics), " and majority is : ", len(majority_electrics))
        print("Common food rules : ", len(common_food), " and majority is : ", len(majority_food))
        print("Common health rules : ", len(common_health), " and majority is : ", len(majority_health))
        print("Common home rules : ", len(common_home), " and majority is : ", len(majority_home))
        print("Common misc rules : ", len(common_misc), " and majority is : ", len(majority_misc))
        print("Common restaurant rules : ", len(common_restaurant), " and majority is : ", len(majority_restaurant))
        print("Common total rules : ", len(common_total), " and majority is : ", len(majority_total)) 
        
        text_file =  open("common_rules_clothes.txt", "w") 
        for rule in common_clothes:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(clothes_1[rule]) + " medium: " + str(clothes_2[rule]) + " large: " + str(clothes_3[rule]))
                text_file.write('\n')      
        text_file.close()
        
        text_file =  open("common_rules_department.txt", "w") 
        for rule in common_department:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(department_1[rule]) + " medium: " + str(department_2[rule]) + " large: " + str(department_3[rule]))
                text_file.write('\n')      
        text_file.close()
        
        text_file =  open("common_rules_electrics.txt", "w") 
        for rule in common_electrics:
            text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(electrics_1[rule]) + " medium: " + str(electrics_2[rule]) + " large: " + str(electrics_3[rule]))
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("common_rules_food.txt", "w") 
        for rule in common_food:
            text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(food_1[rule]) + " medium: " + str(food_2[rule]) + " large: " + str(food_3[rule]))
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("common_rules_health.txt", "w") 
        for rule in common_health:
            text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(health_1[rule]) + " medium: " + str(health_2[rule]) + " large: " + str(health_3[rule]))
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("common_rules_home.txt", "w") 
        for rule in common_home:
            text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(home_1[rule]) + " medium: " + str(home_2[rule]) + " large: " + str(home_3[rule]))
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("common_rules_misc.txt", "w") 
        for rule in common_misc:
            text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(misc_1[rule]) + " medium: " + str(misc_2[rule]) + " large: " + str(misc_3[rule]))
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("common_rules_restaurant.txt", "w") 
        for rule in common_restaurant:
            text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(restaurant_1[rule]) + " medium: " + str(restaurant_2[rule]) + " large: " + str(restaurant_3[rule]))
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("common_rules_total.txt", "w") 
        for rule in common_total:
            text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(total_1[rule]) + " medium: " + str(total_2[rule]) + " large: " + str(total_3[rule]))
            text_file.write('\n')      
        text_file.close()
        
        
        text_file =  open("majority_rules_clothes.txt", "w") 
        for rule in majority_clothes:
            if rule in clothes_1 and rule in clothes_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(clothes_1[rule]) + " medium: " + str(clothes_2[rule]))
            elif rule in clothes_1 and rule in clothes_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(clothes_1[rule]) + " large: " + str(clothes_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " medium: " + str(clothes_2[rule]) + " large: " + str(clothes_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_department.txt", "w") 
        for rule in majority_department:
            if rule in department_1 and rule in department_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(department_1[rule]) + " medium: " + str(department_2[rule]))
            elif rule in department_1 and rule in department_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(department_1[rule]) + " large: " + str(department_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " medium: " + str(department_2[rule]) + " large: " + str(department_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_electrics.txt", "w") 
        for rule in majority_electrics:
            if rule in electrics_1 and rule in electrics_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(electrics_1[rule]) + " medium: " + str(electrics_2[rule]))
            elif rule in electrics_1 and rule in electrics_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(electrics_1[rule]) + " large: " + str(electrics_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " medium: " + str(electrics_2[rule]) + " large: " + str(electrics_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_food.txt", "w") 
        for rule in majority_food:
            if rule in food_1 and rule in food_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(food_1[rule]) + " medium: " + str(food_2[rule]))
            elif rule in food_1 and rule in food_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(food_1[rule]) + " large: " + str(food_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " medium: " + str(food_2[rule]) + " large: " + str(food_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_health.txt", "w") 
        for rule in majority_health:
            if rule in health_1 and rule in health_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(health_1[rule]) + " medium: " + str(health_2[rule]))
            elif rule in health_1 and rule in health_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(health_1[rule]) + " large: " + str(health_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " medium: " + str(health_2[rule]) + " large: " + str(health_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_home.txt", "w") 
        for rule in majority_home:
            if rule in home_1 and rule in home_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(home_1[rule]) + " medium: " + str(home_2[rule]))
            elif rule in home_1 and rule in home_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(home_1[rule]) + " large: " + str(home_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " medium: " + str(home_2[rule]) + " large: " + str(home_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_misc.txt", "w") 
        for rule in majority_misc:
            if rule in misc_1 and rule in misc_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(misc_1[rule]) + " medium: " + str(misc_2[rule]))
            elif rule in misc_1 and rule in misc_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(misc_1[rule]) + " large: " + str(misc_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " medium: " + str(misc_2[rule]) + " large: " + str(misc_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_restaurant.txt", "w") 
        for rule in majority_restaurant:
            if rule in restaurant_1 and rule in restaurant_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(restaurant_1[rule]) + " medium: " + str(restaurant_2[rule]))
            elif rule in restaurant_1 and rule in restaurant_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(restaurant_1[rule]) + " large: " + str(restaurant_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " medium: " + str(restaurant_2[rule]) + " large: " + str(restaurant_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_total.txt", "w") 
        for rule in majority_total:
            if rule in total_1 and rule in total_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(total_1[rule]) + " medium: " + str(total_2[rule]))
            elif rule in total_1 and rule in total_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(total_1[rule]) + " large: " + str(total_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " medium: " + str(total_2[rule]) + " large: " + str(total_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        '''
        
    def spatial_rules_majority_vote(self):

        clothes_1 = {}
        clothes_2 = {}
        clothes_3 = {}
        department_1 = {}
        department_2 = {}
        department_3 = {}
        electrics_1 = {}
        electrics_2 = {}
        electrics_3 = {}
        food_1 = {}
        food_2 = {}
        food_3 = {}
        health_1 = {}
        health_2 = {}
        health_3 = {}
        home_1 = {}
        home_2 = {}
        home_3 = {}
        misc_1 = {}
        misc_2 = {}
        misc_3 = {}
        restaurant_1 = {}
        restaurant_2 = {}
        restaurant_3 = {}
        total_1 = {}
        total_2 = {}
        total_3 = {}
        '''
        common_clothes = []
        common_department = []
        common_electrics = []
        common_food = []
        common_health = []
        common_home = []
        common_misc = []
        common_restaurant = []
        common_total = []
        majority_clothes = []
        majority_department = []
        majority_electrics = []
        majority_food = []
        majority_health = []
        majority_home = []
        majority_misc = []
        majority_restaurant = []
        majority_total = []
        '''
        for filename in os.listdir("C:/Users/ahmed/OneDrive/Documents/Masters Courses/Dissertation/python_code/dbscan_rules"):
            print(filename)
            file = os.path.join("C:/Users/ahmed/OneDrive/Documents/Masters Courses/Dissertation/python_code/dbscan_rules", filename)
            with open(file, "r" ,encoding='utf8') as filestream:
                count = -1
                for line in filestream:
                    count += 1
                    if count == 0:
                        continue
                    splitline = re.findall('\"(:?.+?)\"',line)
                    LHS = splitline[1].replace("{","")
                    LHS = LHS.replace("}","")
                    RHS = splitline[2].replace("{","")
                    RHS = RHS.replace("}","")
                    if RHS in self.relative_leadernumber_majority: RHS = (RHS,)
                    splitline = re.findall('\d+\.?\d*',line)
                    #print(splitline)
                    support = splitline[1]
                    confidence = splitline[2]
                    lift = splitline[3]
                    if "," in LHS:
                        LHS = LHS.split(",")
                    else:
                        LHS = [LHS] 
                              
                        if filename == "R_Rules_clothes.txt": 
                            clothes_1[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_department.txt":
                            department_1[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_electrics.txt":
                            electrics_1[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_food.txt":
                            food_1[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_health.txt":
                            health_1[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_home.txt":
                            home_1[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_misc.txt":
                            misc_1[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_restaurant.txt":
                            restaurant_1[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_pruned.txt":
                            total_1[(str(LHS),RHS)] = (support,confidence,lift)                            
                        
                            
        for filename in os.listdir("C:/Users/ahmed/OneDrive/Documents/Masters Courses/Dissertation/python_code/optics_rules"):   
            print(filename)     
            file = os.path.join("C:/Users/ahmed/OneDrive/Documents/Masters Courses/Dissertation/python_code/optics_rules", filename)
            with open(file, "r" ,encoding='utf8') as filestream:
                count = -1
                for line in filestream:
                    count += 1
                    if count == 0:
                        continue
                    splitline = re.findall('\"(:?.+?)\"',line)
                    LHS = splitline[1].replace("{","")
                    LHS = LHS.replace("}","")
                    RHS = splitline[2].replace("{","")
                    RHS = RHS.replace("}","")
                    splitline = re.findall('\d+\.?\d*',line)
                    #print(splitline)
                    support = splitline[1]
                    confidence = splitline[2]
                    lift = splitline[3]
                    if "," in LHS:
                        LHS = LHS.split(",")
                    else:
                        LHS = [LHS] 
                    
                        if filename == "R_Rules_clothes.txt":    
                            clothes_2[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_department.txt":
                            department_2[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_electrics.txt":
                            electrics_2[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_food.txt":
                            food_2[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_health.txt":
                            health_2[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_home.txt":
                            home_2[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_misc.txt":
                            misc_2[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_restaurant.txt":
                            restaurant_2[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_pruned.txt":
                            total_2[(str(LHS),RHS)] = (support,confidence,lift)      
        
        for filename in os.listdir("C:/Users/ahmed/OneDrive/Documents/Masters Courses/Dissertation/python_code/hdbscan_rules"):     
            print(filename)       
            file = os.path.join("C:/Users/ahmed/OneDrive/Documents/Masters Courses/Dissertation/python_code/hdbscan_rules", filename)
            with open(file, "r" ,encoding='utf8') as filestream:
                count = -1
                for line in filestream:
                    count += 1
                    if count == 0:
                        continue
                    splitline = re.findall('\"(:?.+?)\"',line)
                    LHS = splitline[1].replace("{","")
                    LHS = LHS.replace("}","")
                    RHS = splitline[2].replace("{","")
                    RHS = RHS.replace("}","")
                    splitline = re.findall('\d+\.?\d*',line)
                    #print(splitline)
                    support = splitline[1]
                    confidence = splitline[2]
                    lift = splitline[3]
                    if "," in LHS:
                        LHS = LHS.split(",")
                    else:
                        LHS = [LHS] 
                                
                    
                        if filename == "R_Rules_clothes.txt":    
                            clothes_3[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_department.txt":
                            department_3[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_electrics.txt":
                            electrics_3[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_food.txt":
                            food_3[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_health.txt":
                            health_3[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_home.txt":
                            home_3[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_misc.txt":
                            misc_3[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_restaurant.txt":
                            restaurant_3[(str(LHS),RHS)] = (support,confidence,lift)
                        elif filename == "R_Rules_pruned.txt":
                            total_3[(str(LHS),RHS)] = (support,confidence,lift)  
        
        common_clothes = list(clothes_1.keys()) + list(clothes_2.keys()) + list(clothes_3.keys())
        common_department = list(department_1.keys()) + list(department_2.keys()) + list(department_3.keys())
        common_electrics = list(electrics_1.keys()) + list(electrics_2.keys()) + list(electrics_3.keys())
        common_food = list(food_1.keys()) + list(food_2.keys()) + list(food_3.keys())
        common_health = list(health_1.keys()) + list(health_2.keys()) + list(health_3.keys())
        common_home = list(home_1.keys()) + list(home_2.keys()) + list(home_3.keys())
        common_misc = list(misc_1.keys()) + list(misc_2.keys()) + list(misc_3.keys())
        common_restaurant = list(restaurant_1.keys()) + list(restaurant_2.keys()) + list(restaurant_3.keys())
        common_total = list(total_1.keys()) + list(total_2.keys()) + list(total_3.keys())
        
        
        majority_clothes = [item for item, count in Counter(common_clothes).items() if count > 1]
        majority_department = [item for item, count in Counter(common_department).items() if count > 1]
        majority_electrics = [item for item, count in Counter(common_electrics).items() if count > 1]
        majority_food = [item for item, count in Counter(common_food).items() if count > 1]
        majority_health = [item for item, count in Counter(common_health).items() if count > 1]
        majority_home = [item for item, count in Counter(common_home).items() if count > 1]
        majority_misc = [item for item, count in Counter(common_misc).items() if count > 1]
        majority_restaurant = [item for item, count in Counter(common_restaurant).items() if count > 1]
        temp = majority_clothes + majority_department + majority_electrics + majority_food + majority_health + majority_home + majority_misc + majority_restaurant
        majority_total = [item for item, count in Counter(common_total).items() if count > 1 and item not in temp]
        majority_clothes.sort(key= lambda x:len(x[0]), reverse=False)
        majority_department.sort(key= lambda x:len(x[0]), reverse=False)
        majority_electrics.sort(key= lambda x:len(x[0]), reverse=False)
        majority_food.sort(key= lambda x:len(x[0]), reverse=False)
        majority_health.sort(key= lambda x:len(x[0]), reverse=False)
        majority_home.sort(key= lambda x:len(x[0]), reverse=False)
        majority_misc.sort(key= lambda x:len(x[0]), reverse=False)
        majority_restaurant.sort(key= lambda x:len(x[0]), reverse=False)
        majority_total.sort(key= lambda x:len(x[0]), reverse=False)
        print("Common clothes rules : ", len(common_clothes), " and majority is : ", len(majority_clothes))  
        print("Common department rules : ", len(common_department), " and majority is : ", len(majority_department))
        print("Common electrics rules : ", len(common_electrics), " and majority is : ", len(majority_electrics))
        print("Common food rules : ", len(common_food), " and majority is : ", len(majority_food))
        print("Common health rules : ", len(common_health), " and majority is : ", len(majority_health))
        print("Common home rules : ", len(common_home), " and majority is : ", len(majority_home))
        print("Common misc rules : ", len(common_misc), " and majority is : ", len(majority_misc))
        print("Common restaurant rules : ", len(common_restaurant), " and majority is : ", len(majority_restaurant))
        print("Common total rules : ", len(common_total), " and majority is : ", len(majority_total)) 
        
        text_file =  open("majority_rules_clothes.txt", "w") 
        for rule in majority_clothes:
            if rule in clothes_1 and rule in clothes_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " DBSCAN: " + str(clothes_1[rule]) + " OPTICS: " + str(clothes_2[rule]))
            elif rule in clothes_1 and rule in clothes_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " DBSCAN: " + str(clothes_1[rule]) + " HDBSCAN: " + str(clothes_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " OPTICS: " + str(clothes_2[rule]) + " HDBSCAN: " + str(clothes_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_department.txt", "w") 
        for rule in majority_department:
            if rule in department_1 and rule in department_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " DBSCAN: " + str(department_1[rule]) + " OPTICS: " + str(department_2[rule]))
            elif rule in department_1 and rule in department_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " DBSCAN: " + str(department_1[rule]) + " HDBSCAN: " + str(department_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " OPTICS: " + str(department_2[rule]) + " HDBSCAN: " + str(department_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_electrics.txt", "w") 
        for rule in majority_electrics:
            if rule in electrics_1 and rule in electrics_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " DBSCAN: " + str(electrics_1[rule]) + " OPTICS: " + str(electrics_2[rule]))
            elif rule in electrics_1 and rule in electrics_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " DBSCAN: " + str(electrics_1[rule]) + " HDBSCAN: " + str(electrics_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " OPTICS: " + str(electrics_2[rule]) + " HDBSCAN: " + str(electrics_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_food.txt", "w") 
        for rule in majority_food:
            if rule in food_1 and rule in food_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " DBSCAN: " + str(food_1[rule]) + " OPTICS: " + str(food_2[rule]))
            elif rule in food_1 and rule in food_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " DBSCAN: " + str(food_1[rule]) + " HDBSCAN: " + str(food_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " OPTICS: " + str(food_2[rule]) + " HDBSCAN: " + str(food_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_health.txt", "w") 
        for rule in majority_health:
            if rule in health_1 and rule in health_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " DBSCAN: " + str(health_1[rule]) + " OPTICS: " + str(health_2[rule]))
            elif rule in health_1 and rule in health_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " DBSCAN: " + str(health_1[rule]) + " HDBSCAN: " + str(health_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " OPTICS: " + str(health_2[rule]) + " HDBSCAN: " + str(health_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_home.txt", "w") 
        for rule in majority_home:
            if rule in home_1 and rule in home_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " DBSCAN: " + str(home_1[rule]) + " OPTICS: " + str(home_2[rule]))
            elif rule in home_1 and rule in home_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " DBSCAN: " + str(home_1[rule]) + " HDBSCAN: " + str(home_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " OPTICS: " + str(home_2[rule]) + " HDBSCAN: " + str(home_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_misc.txt", "w") 
        for rule in majority_misc:
            if rule in misc_1 and rule in misc_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " DBSCAN: " + str(misc_1[rule]) + " OPTICS: " + str(misc_2[rule]))
            elif rule in misc_1 and rule in misc_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " DBSCAN: " + str(misc_1[rule]) + " HDBSCAN: " + str(misc_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " OPTICS: " + str(misc_2[rule]) + " HDBSCAN: " + str(misc_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_restaurant.txt", "w") 
        for rule in majority_restaurant:
            if rule in restaurant_1 and rule in restaurant_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " DBSCAN: " + str(restaurant_1[rule]) + " OPTICS: " + str(restaurant_2[rule]))
            elif rule in restaurant_1 and rule in restaurant_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " DBSCAN: " + str(restaurant_1[rule]) + " HDBSCAN: " + str(restaurant_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " OPTICS: " + str(restaurant_2[rule]) + " HDBSCAN: " + str(restaurant_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_total.txt", "w") 
        for rule in majority_total:
            if rule in total_1 and rule in total_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " DBSCAN: " + str(total_1[rule]) + " OPTICS: " + str(total_2[rule]))
            elif rule in total_1 and rule in total_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " DBSCAN: " + str(total_1[rule]) + " HDBSCAN: " + str(total_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " OPTICS: " + str(total_2[rule]) + " HDBSCAN: " + str(total_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        '''        
        for rule in clothes_1.keys():
            if rule in clothes_2 and rule in clothes_3:
                common_clothes.append(rule)        
                #print(rule)
            elif rule in clothes_2 or rule in clothes_3:
                majority_clothes.append(rule)    
                
        for rule in department_1.keys():
            if rule in department_2 and rule in department_3:
                common_department.append(rule)     
            elif rule in department_2 or rule in department_3:
                majority_department.append(rule)       
                #print(rule)
        
        for rule in electrics_1.keys():
            if rule in electrics_2 and rule in electrics_3:
                common_electrics.append(rule)        
                #print(rule)
            elif rule in electrics_2 or rule in electrics_3:
                majority_electrics.append(rule)    
        
        for rule in food_1.keys():
            if rule in food_2 and rule in food_3:
                common_food.append(rule)        
                #print(rule)
            elif rule in food_2 or rule in food_3:
                majority_food.append(rule)    
        
        for rule in health_1.keys():
            if rule in health_2 and rule in health_3:
                common_health.append(rule)        
                #print(rule)
            elif rule in health_2 or rule in health_3:
                majority_health.append(rule)    
        
        for rule in home_1.keys():
            if rule in home_2 and rule in home_3:
                common_home.append(rule)        
                #print(rule)
            elif rule in home_2 or rule in home_3:
                majority_home.append(rule)    
                
        for rule in misc_1.keys():
            if rule in misc_2 and rule in misc_3:
                common_misc.append(rule)        
                #print(rule)
            elif rule in misc_2 or rule in misc_3:
                majority_misc.append(rule)    
                
        for rule in restaurant_1.keys():
            if rule in restaurant_2 and rule in restaurant_3:
                common_restaurant.append(rule)        
                #print(rule)        
            elif rule in restaurant_2 or rule in restaurant_3:
                majority_restaurant.append(rule)    
                
        for rule in total_1.keys():
            if rule in total_2 and rule in total_3:
                common_total.append(rule)        
                #print(rule)
            elif rule in total_2 or rule in total_3:
                majority_total.append(rule)   
                
                
                
        for rule in clothes_2.keys():
            if rule in clothes_3:
                majority_clothes.append(rule) 
                
        for rule in department_2.keys():
            if rule in department_3:
                majority_department.append(rule) 
        
        for rule in electrics_2.keys():
            if rule in electrics_3:
                majority_electrics.append(rule) 
        
        for rule in food_2.keys():
            if rule in food_3:
                majority_food.append(rule) 
                
        for rule in health_2.keys():
            if rule in health_3:
                majority_health.append(rule) 
                
        for rule in home_2.keys():
            if rule in home_3:
                majority_home.append(rule) 
                
        for rule in misc_2.keys():
            if rule in misc_3:
                majority_misc.append(rule) 
                
        for rule in restaurant_2.keys():
            if rule in restaurant_3:
                majority_restaurant.append(rule) 
                
        for rule in total_2.keys():
            if rule in total_3:
                majority_total.append(rule)                                                                        
                
        majority_clothes = set(majority_clothes) - set(common_clothes)
        majority_department = set(majority_department) - set(common_department)
        majority_electrics = set(majority_electrics) - set(common_electrics)
        majority_food = set(majority_food) - set(common_food)
        majority_health = set(majority_health) - set(common_health)
        majority_home = set(majority_home) - set(common_home)
        majority_misc = set(majority_misc) - set(common_misc)
        majority_restaurant = set(majority_restaurant) - set(common_restaurant)
        majority_total = set(majority_total) - set(common_total)                                                                     
                
        print("Common clothes rules : ", len(common_clothes), " and majority is : ", len(majority_clothes))  
        print("Common department rules : ", len(common_department), " and majority is : ", len(majority_department))
        print("Common electrics rules : ", len(common_electrics), " and majority is : ", len(majority_electrics))
        print("Common food rules : ", len(common_food), " and majority is : ", len(majority_food))
        print("Common health rules : ", len(common_health), " and majority is : ", len(majority_health))
        print("Common home rules : ", len(common_home), " and majority is : ", len(majority_home))
        print("Common misc rules : ", len(common_misc), " and majority is : ", len(majority_misc))
        print("Common restaurant rules : ", len(common_restaurant), " and majority is : ", len(majority_restaurant))
        print("Common total rules : ", len(common_total), " and majority is : ", len(majority_total)) 
        
        text_file =  open("common_rules_clothes.txt", "w") 
        for rule in common_clothes:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(clothes_1[rule]) + " medium: " + str(clothes_2[rule]) + " large: " + str(clothes_3[rule]))
                text_file.write('\n')      
        text_file.close()
        
        text_file =  open("common_rules_department.txt", "w") 
        for rule in common_department:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(department_1[rule]) + " medium: " + str(department_2[rule]) + " large: " + str(department_3[rule]))
                text_file.write('\n')      
        text_file.close()
        
        text_file =  open("common_rules_electrics.txt", "w") 
        for rule in common_electrics:
            text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(electrics_1[rule]) + " medium: " + str(electrics_2[rule]) + " large: " + str(electrics_3[rule]))
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("common_rules_food.txt", "w") 
        for rule in common_food:
            text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(food_1[rule]) + " medium: " + str(food_2[rule]) + " large: " + str(food_3[rule]))
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("common_rules_health.txt", "w") 
        for rule in common_health:
            text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(health_1[rule]) + " medium: " + str(health_2[rule]) + " large: " + str(health_3[rule]))
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("common_rules_home.txt", "w") 
        for rule in common_home:
            text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(home_1[rule]) + " medium: " + str(home_2[rule]) + " large: " + str(home_3[rule]))
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("common_rules_misc.txt", "w") 
        for rule in common_misc:
            text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(misc_1[rule]) + " medium: " + str(misc_2[rule]) + " large: " + str(misc_3[rule]))
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("common_rules_restaurant.txt", "w") 
        for rule in common_restaurant:
            text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(restaurant_1[rule]) + " medium: " + str(restaurant_2[rule]) + " large: " + str(restaurant_3[rule]))
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("common_rules_total.txt", "w") 
        for rule in common_total:
            text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(total_1[rule]) + " medium: " + str(total_2[rule]) + " large: " + str(total_3[rule]))
            text_file.write('\n')      
        text_file.close()
        
        
        text_file =  open("majority_rules_clothes.txt", "w") 
        for rule in majority_clothes:
            if rule in clothes_1 and rule in clothes_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(clothes_1[rule]) + " medium: " + str(clothes_2[rule]))
            elif rule in clothes_1 and rule in clothes_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(clothes_1[rule]) + " large: " + str(clothes_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " medium: " + str(clothes_2[rule]) + " large: " + str(clothes_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_department.txt", "w") 
        for rule in majority_department:
            if rule in department_1 and rule in department_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(department_1[rule]) + " medium: " + str(department_2[rule]))
            elif rule in department_1 and rule in department_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(department_1[rule]) + " large: " + str(department_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " medium: " + str(department_2[rule]) + " large: " + str(department_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_electrics.txt", "w") 
        for rule in majority_electrics:
            if rule in electrics_1 and rule in electrics_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(electrics_1[rule]) + " medium: " + str(electrics_2[rule]))
            elif rule in electrics_1 and rule in electrics_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(electrics_1[rule]) + " large: " + str(electrics_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " medium: " + str(electrics_2[rule]) + " large: " + str(electrics_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_food.txt", "w") 
        for rule in majority_food:
            if rule in food_1 and rule in food_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(food_1[rule]) + " medium: " + str(food_2[rule]))
            elif rule in food_1 and rule in food_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(food_1[rule]) + " large: " + str(food_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " medium: " + str(food_2[rule]) + " large: " + str(food_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_health.txt", "w") 
        for rule in majority_health:
            if rule in health_1 and rule in health_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(health_1[rule]) + " medium: " + str(health_2[rule]))
            elif rule in health_1 and rule in health_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(health_1[rule]) + " large: " + str(health_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " medium: " + str(health_2[rule]) + " large: " + str(health_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_home.txt", "w") 
        for rule in majority_home:
            if rule in home_1 and rule in home_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(home_1[rule]) + " medium: " + str(home_2[rule]))
            elif rule in home_1 and rule in home_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(home_1[rule]) + " large: " + str(home_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " medium: " + str(home_2[rule]) + " large: " + str(home_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_misc.txt", "w") 
        for rule in majority_misc:
            if rule in misc_1 and rule in misc_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(misc_1[rule]) + " medium: " + str(misc_2[rule]))
            elif rule in misc_1 and rule in misc_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(misc_1[rule]) + " large: " + str(misc_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " medium: " + str(misc_2[rule]) + " large: " + str(misc_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_restaurant.txt", "w") 
        for rule in majority_restaurant:
            if rule in restaurant_1 and rule in restaurant_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(restaurant_1[rule]) + " medium: " + str(restaurant_2[rule]))
            elif rule in restaurant_1 and rule in restaurant_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(restaurant_1[rule]) + " large: " + str(restaurant_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " medium: " + str(restaurant_2[rule]) + " large: " + str(restaurant_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        
        text_file =  open("majority_rules_total.txt", "w") 
        for rule in majority_total:
            if rule in total_1 and rule in total_2:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(total_1[rule]) + " medium: " + str(total_2[rule]))
            elif rule in total_1 and rule in total_3:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " small: " + str(total_1[rule]) + " large: " + str(total_3[rule]))
            else:
                text_file.write(rule[0] + ' ==> ' + rule[1] + " medium: " + str(total_2[rule]) + " large: " + str(total_3[rule]))   
            text_file.write('\n')      
        text_file.close()
        '''
                                               
               
        
        
            
def main():
    
    x = clusterer()
    
    #x.hdbscan_test("finalData_FinalVersion.csv")
    #x.dbscan_test("finalData_FinalVersion.csv")
    #x.OPTICS_test("finalData_FinalVersion.csv")
    #x.clusterStores("final_Data_with_Range.csv")  
    
    x.DBSCAN_Clustering("finalData_FinalVersion.csv")
    dataset = x.leaderfollowerModel("final_Data_with_Range_Cluster.csv", "dbscan")
    x.association_mining(dataset)  
    x.generateFiles(dataset)
    retcode = subprocess.call(["C:/Program Files/R/R-3.3.3/bin/x64/Rscript.exe", "C:/Users/ahmed/OneDrive/Documents/R/dbscan_rules.R"])
    
    x.OPTICS_Clustering("finalData_FinalVersion.csv")
    dataset = x.leaderfollowerModel("final_Data_with_Range_Cluster_optics.csv", "optics")
    x.association_mining(dataset)  
    x.generateFiles(dataset)
    retcode = subprocess.call(["C:/Program Files/R/R-3.3.3/bin/x64/Rscript.exe", "C:/Users/ahmed/OneDrive/Documents/R/optics_rules.R"])
    
    x.HDBSCAN_Clustering("finalData_FinalVersion.csv")
    dataset = x.leaderfollowerModel("final_Data_with_Range_Cluster_hdbscan.csv", "hdbscan")
    x.association_mining(dataset)  
    x.generateFiles(dataset)
    retcode = subprocess.call(["C:/Program Files/R/R-3.3.3/bin/x64/Rscript.exe", "C:/Users/ahmed/OneDrive/Documents/R/hdbscan_rules.R"])
    
    x.spatio_temporal_rules_majority_vote()
    
    
    #x.meanShift("finalData_FinalVersion.csv")   
    #x.association_mining(dataset)  
    #x.generateFiles(dataset)
    #x.spatio_temporal()
    '''
    retcode = subprocess.call(["C:/Program Files/R/R-3.3.3/bin/x64/Rscript.exe", "C:/Users/ahmed/OneDrive/Documents/R/script_for_python.R"])
    R_output = {"All_rules":[],"restaurant":[],"food":[],"department":[],"health":[],"clothes":[],"electrics":[],"misc":[],"home":[]}
    with open("rule_results.csv",encoding='utf8') as sqldbCSV:
                    
        csvread = csv.reader(sqldbCSV, delimiter=",")
        count = -1
        for line in csvread:
            #print(line)
            count = count + 1
            # Ignoring the header row
            if count == 0:
                continue
                        
            R_output["All_rules"].append(line[0])
            R_output["restaurant"].append(line[1])
            R_output["food"].append(line[2])
            R_output["department"].append(line[3])
            R_output["health"].append(line[5])
            R_output["clothes"].append(line[4])
            R_output["electrics"].append(line[6])
            R_output["misc"].append(line[7])
            R_output["home"].append(line[8])
                
    for k,v in R_output.items():
        print(k , v)
    '''   
    #count = 0
    #count2 = 0
    #with open("Food_and_Grocery.txt","r",encoding = 'utf8') as myText:
    #    for line in myText:
    #        companies = line.split(",")
    #        if "TESCO STORES LIMITED" in companies and "CO-OPERATIVE GROUP LIMITED" in companies:
    #            count +=1
    #            
    #print(count)            
                    
if __name__ == "__main__":
    main()
    
                             
        

                    