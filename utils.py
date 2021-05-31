import numpy as np
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import datetime as dt
from tqdm import tqdm

from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.preprocessing import normalize
import random

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.pyplot import text
from mpl_toolkits import mplot3d
import json
import gmaps
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter1d

from os import path, listdir
import pickle
import geopandas as gpd
import requests
import pycountry
from shapely.ops import nearest_points
import holidays as holiday_module
import matplotlib.dates as mdates

import requests

def load_prepare(path,path_zips,ref_date):
    data = pd.read_csv(f'{path}', parse_dates=['date'],
                      dtype={'origin_area_code': 'int', 'destination_area_code': 'int','counts_anon':'float'}).rename(columns={'counts_anon':'all'})
    data = compute_relative_change(data,ref_date)

    zips = pd.read_csv(f'{path_zips}zipcodes.csv')
    zips = zips.drop_duplicates(subset=['city_code', 'city']).set_index('city_code')
    zips['city'] = zips['city'].replace({"Aarhus":"Århus","Vesthimmerlands":"Vesthimmerland"}).str.replace('-',' ')


    data = pd.merge(data, zips[['city']], how='left', left_on='origin_area_code', right_on='city_code').rename(
        columns={'city': 'source_kommune'})
    data = pd.merge(data, zips[['city']], how='left', left_on='destination_area_code', right_on='city_code').rename(
        columns={'city': 'target_kommune'}).dropna(axis=0)
    data = data.rename(columns={'all_ref': 'n_baseline', 'all': 'n_crisis', 'rel_change': 'percent_change'})

    return data

def compute_relative_change(df,
                            ref_date=dt.datetime(2020, 3, 1)):

    # Relative change
    # Compute total trips by origin
    trips_by_day_by_place = df.groupby(['date', 'origin_area_code', 'destination_area_code']).sum().reset_index()
    trips_by_day_by_place['weekday'] = trips_by_day_by_place['date'].apply(lambda x: x.weekday())

    # Compute the reference (median on that weekday in period before ref_data)
    tot_trips_ref = trips_by_day_by_place[trips_by_day_by_place.date < ref_date].groupby(
        ['weekday', 'origin_area_code', 'destination_area_code'])['all'].median().reset_index()

    # Merge with actual data on weekday
    df_change = pd.merge(trips_by_day_by_place, tot_trips_ref,
                         on=['weekday', 'origin_area_code', 'destination_area_code'],
                         how='left',
                         suffixes=['', '_ref']).fillna(5)

    # Compute relative change
    df_change['rel_change'] = (df_change['all'] - df_change['all_ref']) / df_change['all_ref']

    return df_change

def load_shape(country, adm=2, force_recompute=False):
    shape_file_name = country + '.pkl'

    if path.isfile(shape_file_name) and (not force_recompute):
        with open(shape_file_name, 'rb') as f:
            shapeDF = pickle.load(f)
    else:
        shapeDF = geo_utils.create_shape_file(country, adm, return_geo_pd=True).drop(['NAME_0'], axis=1)
        shapeDF = shapeDF[shapeDF.columns[shapeDF.columns.str.startswith('NAME_')].union(['geometry'])]
        shapeDF.rename(
            columns={'NAME_0': 'adm0', 'NAME_1': 'adm1', 'NAME_2': 'adm2', 'NAME_3': 'adm3', 'NAME_4': 'adm4'},
            inplace=True)
        shapeDF['centroid'] = shapeDF.centroid
        with open(shape_file_name, 'wb') as f:
            pickle.dump(shapeDF, f)
    return shapeDF

def date_check(date): # This should be changed depending on the start date
    # Lockdown 11 marts
    # 11 Maj aabner restauranter osv
    # 16 december lukker alt ned igen
    # 5 Januar, strengere restrictioner
    if date in holiday_module.DK():
        return 8
    elif date < dt.date(2020,3,14):
        return date.weekday()%7//5
    elif date < dt.date(2020,5,21):
        return date.weekday()%7//5+2
    elif date < dt.date(2020,12,16):
        return date.weekday()%7//5+4
    else:
        return date.weekday()%7//5+6
    
def distance_quantiles(locations, quantiles = [0,0.25,0.5,1]):
    # load geographical locations
    with open("Denmark.pkl", 'rb') as f:
        geoDF = pickle.load(f).set_index('adm2')
    
    qt_dict = {}
    for i, source_loc in enumerate(locations):
        distances = np.array([geoDF.loc[source_loc]['centroid'].distance(geoDF.loc[loc]['centroid']) for loc in locations ])
        quan_values = [np.quantile(sorted(distances),quan) for quan in quantiles]
        qt_dict[source_loc] = {quantiles[i+1]: np.array(locations)[(distances>quan_values[i]) & (distances<=quan_values[i+1])] for i in range(len(quantiles)-1)}
        qt_dict[source_loc][quantiles[1]] = np.insert(qt_dict[source_loc][quantiles[1]],-1,source_loc)
        qt_dict[source_loc]['distances'] = distances
    return qt_dict

def load_demo_and_pop():
    pd_demo = pd.read_csv('20209113258294831926INDKF11146984702163.csv',encoding='ISO-8859-1',header=None,sep=';')[[3,4]].rename(columns={3:'kommune',4:'dis income'})
    pd_pep = pd.read_csv('202011593255302782322FOLK1A34420899857.csv',encoding='ISO-8859-1',header=None,sep=';', skiprows=6)[[3,4]].rename(columns={3:'kommune',4:'population'})
    dict_names = {'Faaborg-Midtfyn':'Faaborg Midtfyn',
     'Vesthimmerlands':'Vesthimmerland',
     'Ikast-Brande':'Ikast Brande',
     'Lyngby-Taarbæk':'Lyngby Taarbæk',
     'Aarhus':'Århus',
     'Ringkøbing-Skjern':'Ringkøbing Skjern',
     'Copenhagen':'København',
     'Høje-Taastrup':'Høje Taastrup'}
    pd_demo.kommune = pd_demo.kommune.replace(dict_names)
    pd_pep.kommune = pd_pep.kommune.replace(dict_names)
    pd_demo = pd_demo.sort_values("kommune").reset_index(drop=True)
    pd_pep = pd_pep.sort_values("kommune").reset_index(drop=True)
    return pd_demo, pd_pep

def identity_function(X):
    return X

class brain:
    def __init__(self, data, location = None, t_method = np.sqrt, N = 3, population = None, save_figs = False, source = 'one', in_out = 'out', path_full_file = None):
        self.source = source
        if path_full_file:
            with open('data.pkl', 'rb') as f:
                self.X, self.missing_dates, self.locations, self.loc_replace, self.dates, self.overall_trend = pickle.load(f)
            
        else:
            self.X, self.missing_dates, self.locations, self.loc_replace = self.create_data_matrix(data, location, population = population, in_out = in_out)
            self.dates = [data.date.min()+dt.timedelta(days=i) for i in range((data.date.max()-data.date.min()).days+1)] 
            self.dates = np.delete(self.dates,self.missing_dates)
            self.overall_trend = data.groupby('date').sum()['n_crisis']
            self.overall_trend.drop(self.overall_trend.index[96], inplace=True)
        self.X_t = None
        self.geo_df = None
        if population is not None:
            self.normalized = True
        else:
            self.normalized = False
        self.N = N
        self.save_figs = save_figs
        self.nmf_t, self.nmf_w, self.c_deaths, self.c_weekends = None, None, None, None
        self.t_method = t_method
        
    def create_data_matrix(self, data, location = None, population = None, in_out = 'out'):
        # Creating the data matrix given if it's of all locations or just one.    
        self.locations = sorted(data.target_kommune.unique())
        self.loc_replace = {loc: i for i,loc in enumerate(self.locations)}
        if population is None:
            population = np.ones(len(self.locations)*len(self.locations))

        print('Loading in data')
        # Convert to numpy for faster computing
        if location != None:
            data_np = data[data.source_kommune==location].to_numpy()
            X = np.zeros([len(self.locations),(data.date.max()-data.date.min()).days+1])
        else:
            data_np = data.to_numpy()
            X = np.zeros([len(self.locations)*len(self.locations),(data.date.max()-data.date.min()).days+1])

        start_date = data.date.min()   
        # Iterate through each row and insert "n_crisis" on the correct location in the matrix
        for date, n_crisis, source, target, percent_change in tqdm(data_np[:,[0,3,-2,-1,6]]):
            idx = (date - start_date).days #Date index
            idy = int(self.loc_replace[target if in_out=='out' else source ])    # Target
            if location == None:
                i = int(self.loc_replace[source if in_out=='out' else target ]) # Source
                X[idy+i*len(self.locations),idx] = n_crisis
            else:
                X[idy,idx] = n_crisis

        # Finding the missing dates and removing them
        missing_dates = np.where(~X.any(axis=0))[0]
        if self.source == 'one':
            missing_dates = np.insert(missing_dates, 0, missing_dates.min()-1)
        X = np.delete(X,missing_dates, axis=1)
        if location is None:
            if in_out=='in':
                population = np.array([population[(i%len(self.locations))*len(self.locations)] for i in range(population.shape[0])])
            X = X/population[:,None]
        # Error correction for log methods. TODO: Make sure it's not a problem.
        #X[X<1]=1
        return X, missing_dates, self.locations, self.loc_replace

    def transform_X(self):
        # Do the transform
        X_t = self.t_method(self.X)
        X_t_n = X_t#/X_t.max()
        self.X_t = X_t_n

    def compute_NMF(self, N = None, t_method = None):
        if ((N is not None) and (N != self.N)) or ((t_method is not None) and (t_method != self.t_method)):
            if N is not None:
                self.N = N
            if t_method is not None:
                self.t_method = t_method
            self.reset_all()
        elif self.nmf_w is not None:
            return None 
        
        print(f'Computing NMF with {self.N} components')
        if self.X_t is None:
            self.transform_X()
        #NMF model
        model = NMF(n_components=self.N, init = 'random', random_state=0,max_iter=100000)
        self.nmf_t = model.fit_transform(self.X_t) 
        self.nmf_w = model.components_
        self.nmf_error = model.reconstruction_err_

    def plot_nmf_3d(self, title='', return_ax = False):
        if self.nmf_w is None:
            self.compute_NMF()
        # Coloring the dates according to period
        group = [date_check(date) for date in self.dates] 
        cdict = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4:'pink',5:'purple', 6:'orange',7:'teal', 8:'black'}
        labeldict = {0: 'Wday before lockdown', 1: 'Wend before lockdown', 2: 'Wday within lockdown', 3: 'Wend within lockdown', 4:'Wday after lockdown',5:'Wend after lockdown', 6: 'Wday in second lockdown', 7: 'Wend in second lockdown', 8:'Holidays'}

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        for g in np.unique(group):
            ix = np.where(group==g)
            ax.scatter(self.nmf_w[2,ix],self.nmf_w[1,ix],self.nmf_w[0,ix],c = cdict[g], label = labeldict[g])
        ax.legend()
        ax.set_title(title)
        if return_ax:
            return fig, ax
        

    def plot_nmf_topic_time(self, return_ax=False, ax=None):
        if self.nmf_w is None:
            self.compute_NMF()
        weekend_idx = [i for i, date in enumerate(self.dates) if date.isoweekday()>=6]

        if ax == None:
            fig, ax = plt.subplots(figsize=(18,12))
        
        for idx in weekend_idx:
            plt.axvspan(self.dates[idx]-dt.timedelta(hours=12), self.dates[idx]+dt.timedelta(hours=12), facecolor='gray', alpha=0.2, edgecolor='none')

        for i in range(self.nmf_w.shape[0]):
            plt.plot(self.dates,self.nmf_w[i,:],linewidth=1)

        events = ['First lockdown', 'Restaurants open', 'Second lockdown']
        for i, date in enumerate([dt.date(2020,3,11),dt.date(2020,5,11),dt.date(2020,12,16)]):
            plt.axvline(date, linestyle='--',color='black', alpha=0.8)
            text(date, self.nmf_w.max(), f"{events[i]} ",horizontalalignment='right')

        plt.legend([f'Topic {i}' for i in range(self.nmf_w.shape[0])],fontsize = 10)
        if return_ax:
            return ax
        
        
#         if self.save_figs:
#             fig.savefig(f'censored', format='png')

    def create_geo_dat(self, normalize_flow = True):
        if self.nmf_t is None:
            self.compute_NMF()
        N_loc = len(self.locations)
        nmf_t = self.nmf_t
        if normalize_flow:
            nmf_t = normalize(nmf_t,axis=1)

        if N_loc!=nmf_t.shape[0]:
            nmf_t = np.array([nmf_t[i*N_loc:i*N_loc+N_loc].mean(axis=0) for i in range(N_loc)])

        with open("Denmark.pkl", 'rb') as f:
            geo_df = pickle.load(f)
        #Only get the zipcodes we need
        geo_df = geo_df[geo_df['adm2'].isin(self.locations)] 
        #Join duplicates and only save zipcode and geometry
        geo_df = geo_df[['adm2','geometry']].dissolve(by='adm2', aggfunc='sum') 
        geo_df = geo_df.to_crs("EPSG:3395")

        for topic in range(nmf_t.shape[1]):
            geo_df[f"topic_{topic}"] = nmf_t[:,topic]
            geo_df[f"topic_{topic}_norm"] = nmf_t[:,topic]/nmf_t[:,topic].max()
            geo_df[f"topic_{topic}_log"] = np.log(geo_df[f"topic_{topic}"]+geo_df[f"topic_{topic}"][geo_df[f"topic_{topic}"]!=0].min())
        geo_df['centroid'] = geo_df.centroid
        self.geo_df = geo_df

    def plot_nmf_geo_map(self, title = '', column_suffix = '', normalize_flow = True, return_ax=False):
        # Check if geo_df is computed
        self.create_geo_dat(normalize_flow=normalize_flow)
        title_list = ['Holidays', 'Weekend', 'Workday']
        
        # create the colorbar
        vmax = self.geo_df[['topic_0_log','topic_1_log','topic_2_log']].max().max()
        vmin = self.geo_df[['topic_0_log','topic_1_log','topic_2_log']].min().min()

        norm = colors.Normalize(vmin=0, vmax=vmax)
        cbar = plt.cm.ScalarMappable(norm=norm)

        fig, ax = plt.subplots(int(np.ceil(self.N/2)), 2, figsize=(12, 5*int(np.ceil(self.N/2))))
        for i in range(self.N):
            self.geo_df.plot(column=f'topic_{i}'+column_suffix, ax=ax[i//2,i%2])
        #self.geo_df.plot(column='topic_0'+column_suffix, ax=ax[0,0])
        #self.geo_df.plot(column='topic_1'+column_suffix, ax=ax[0,1])
        #self.geo_df.plot(column='topic_2'+column_suffix, ax=ax[1,0])

        for i in range(ax.shape[0]*ax.shape[1]):
            ax_tmp = ax[i//2,i%2]
            ax_tmp.xaxis.set_visible(False)
            plt.setp(ax_tmp.spines.values(), visible=False)
            ax_tmp.tick_params(left=False, labelleft=False)
            ax_tmp.ticklabel_format(style='plain')
            if i<self.N:
                ax_tmp.set_title(f"{title_list[i]}",fontsize=20)
        fig.suptitle(title)
        if return_ax:
            return fig, ax
        plt.show() 

    def get_corona_numbers(self, country = 'Denmark'):
        print('Fetching Corona numbers')
        corona_daily_cases = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv")
        corona_daily_cases = corona_daily_cases[['location','new_deaths','date']]
        corona_daily_cases.date = pd.to_datetime(corona_daily_cases.date)
        corona_daily_cases.new_deaths = corona_daily_cases.new_deaths.fillna(0)
        
        #Computing the daily cases
        t_delta = dt.timedelta(days=13)
        nmf_weekends = [1 if (date.isoweekday()>=6) else 0 for date in self.dates]
        nmf_vacation = [1 if (vacation_check(date, country)) else 0 for date in self.dates]
        corona_daily_country = corona_daily_cases[corona_daily_cases.location == country]
        mask = (corona_daily_country.date >= np.min(self.dates)+t_delta) & (corona_daily_country.date<= np.max(self.dates)+t_delta)
        corona_daily_country = corona_daily_country.loc[mask]
        corona_daily_country[corona_daily_country.new_deaths<0]=0
        cases = corona_daily_country.new_deaths.tolist()
        cases = np.delete(cases,self.missing_dates)
        weekends = nmf_weekends
        vacation = nmf_vacation
        self.c_deaths = cases
        self.c_weekends = weekends
        self.c_workdays = [1-i for i in self.c_weekends]
        self.c_vacation = vacation

    def compute_corr(self, country = 'Denmark', f_print = True):
        if self.nmf_t is None:
            self.compute_NMF()
        if self.c_deaths is None:
            self.get_corona_numbers(country = country)
        # Weekend and workday correlation is done per week to avoid the overall changing trend as much as possible.
        self.weekend_corr, self.weekend_p = np.array([pearsonr(self.c_weekends, self.nmf_w[i,:]-uniform_filter1d(self.nmf_w[i,:],7)) for i in range(self.nmf_w.shape[0])]+[pearsonr(self.c_weekends,self.overall_trend)]).T
        self.workday_corr, self.workday_p = np.array([pearsonr(self.c_workdays, self.nmf_w[i,:]-uniform_filter1d(self.nmf_w[i,:],7)) for i in range(self.nmf_w.shape[0])]+[pearsonr(self.c_workdays,self.overall_trend)]).T
        self.vacation_corr, self.vacation_p= np.array([pearsonr(self.t_method(self.c_vacation), self.nmf_w[i,:]) for i in range(self.nmf_w.shape[0])]+[pearsonr(self.c_vacation,self.overall_trend)]).T
        self.corona_corr, self.corona_p = np.array([pearsonr(self.t_method(self.c_deaths), self.nmf_w[i,:]) for i in range(self.nmf_w.shape[0])]+[pearsonr(self.c_deaths,self.overall_trend)]).T
        if f_print:
            print('Correlations in the order of weekend, workday and corona:')
            print(pd.DataFrame(data = np.array([self.weekend_corr, self.workday_corr, self.corona_corr, self.vacation_corr]).T, columns = ['Weekends', 'Weekdays','Corona','Vacation']))
            print(pd.DataFrame(data = np.array([self.weekend_p, self.workday_p, self.corona_p, self.vacation_p]).T, columns = ['Weekends', 'Weekdays','Corona','Vacation']))
            
    def reset_all(self):
        self.X_t, self.geo_df, self.nmf_t, self.nmf_w = None, None, None, None
        
    def compute_all(self, N=None, t_method=None):
        self.compute_NMF(N, t_method) 
        self.plot_nmf_3d()
        self.plot_nmf_topic_time()
        self.plot_nmf_geo_map()
        self.compute_corr()
    
    def check_map_mean(self, location = 'København', normalize_dat = True):
        if self.nmf_t is None:
            self.compute_NMF()
        # Distribution for all outgoing weights of a given location to see if it makes sense to take the mean.
        if len(self.locations)==self.nmf_t.shape[0]:
            print('Not possible for locationwise NMF')
            return None
        if normalize_dat == True:
            nmf_t_norm = normalize(self.nmf_t,axis=1)
        else:
            nmf_t_norm = self.nmf_t
            
        i = np.where(np.array(self.locations) == location)[0][0]
        print(f'Distribution for {location} for each topic after normalization')
        fig, ax = plt.subplots(self.N,1,figsize=(9,2*self.N))
        N_loc = len(self.locations)
        for n in range(self.N):
            ax[n].hist(nmf_t_norm[i*N_loc:i*N_loc+N_loc][:,n],bins=20,alpha = 0.8, range=[0,1])
        plt.show()
        
    def add_neighbors(self, N_neigh):
        self.geo_df["seen_neighbors"] = ""
        self.geo_df["neighbors_1"] = ""
        for index, row in self.geo_df.iterrows():  
            neighbors = self.geo_df[self.geo_df.geometry.touches(row['geometry'])].index.tolist() 
            self.geo_df.at[index, "neighbors_1"] = ", ".join(neighbors)
            self.geo_df.at[index, "seen_neighbors"] = ", ".join(neighbors)+", "+index


        for n in range(2, N_neigh+2):
            self.geo_df[f"neighbors_{n}"] = ""
            for index, row in self.geo_df.iterrows():
                if len(row[f"neighbors_{n-1}"])==0:
                    continue
                neighbors = set()
                for neighbor in row[f"neighbors_{n-1}"].split(', '):
                    neighbors.update(self.geo_df.loc[neighbor]["neighbors_1"].split(', '))
                neighbors = neighbors - set(self.geo_df.loc[index].seen_neighbors.split(', '))
                self.geo_df.at[index, f"neighbors_{n}"] = ", ".join(neighbors)
                self.geo_df.at[index, "seen_neighbors"] = self.geo_df.loc[index].seen_neighbors + ", " + ", ".join(neighbors)
                
    
    
def daily_network(date):
    one_day = d[(d.date==date) & (d.origin_area_code!=d.destination_area_code)][['origin_area_code','destination_area_code','all']].copy()
    one_day_mirror = one_day.merge(one_day, left_on=['origin_area_code','destination_area_code'],right_on=['destination_area_code','origin_area_code'],suffixes = ['','_m'])
    one_day_mirror['total'] = (one_day_mirror['all']+one_day_mirror['all_m'])
    one_day_mirror['total_normalized']= (one_day_mirror['total']/(one_day_mirror['origin_area_code'].map(pop_)+one_day_mirror['destination_area_code'].map(pop_)))
    one_day_mirror['A'] = one_day_mirror['origin_area_code'].map(centroids)
    one_day_mirror['B'] = one_day_mirror['destination_area_code'].map(centroids)
    return one_day_mirror[['origin_area_code','destination_area_code','A','B','total_normalized','total']]

def compute_factor(min_l, max_l, min_data, max_data):
    factor = (max_l - min_l)/(max_data-min_data)
    coeff = min_l - factor*min_data
    return factor, coeff


def plot_network(ax, dataframe, geo_df, factor, coeff, color=False,col='total_normalized', bornholm=''):
    geo_df.plot(ax=ax,
                  color = '#DCDCDC',
                  edgecolor = 'w')
    square = bornholm.geometry.apply(lambda x: x.buffer(10).envelope)
    square.plot(ax=ax,color = 'white',edgecolor = 'k',zorder=0,lw = 0.6,alpha = 0.8)

    if color==False: 
        for index, row in dataframe.iterrows():
            x1,y1 = row['A'].x, row['A'].y
            x2,y2 = row['B'].x, row['B'].y,
            ln = ax.plot([x1,x2],
                    [y1,y2],
                    color = 'k',
                    alpha = 0.5,
                    lw = coeff+factor*(row[col]),
                    solid_capstyle='round')
    elif type(color) == str: 
        for index, row in dataframe.iterrows():
            x1,y1 = row['A'].x, row['A'].y
            x2,y2 = row['B'].x, row['B'].y,
            ln = ax.plot([x1,x2],
                    [y1,y2],
                    color = color,
                    alpha = 0.5,
                    lw = coeff+factor*(row[col]),
                    solid_capstyle='round')
    elif color==True:
        max_value = 1#max([np.abs(min(dataframe[col])),max(dataframe[col])])
        colors_norm = colors.SymLogNorm(vmin=-max_value,
                                        vmax=max_value,
                                        linthresh=0.01)
        cmap = mpl.cm.get_cmap('RdBu')
        for index, row in dataframe.iterrows():
            x1,y1 = row['A']
            x2,y2 = row['B']
            ax.plot([x1,x2],
                    [y1,y2],
                    color = cmap(colors_norm(row[col])),
                    alpha = 0.5,
                    lw = coeff+factor*(row['total_normalized_l']),
                    solid_capstyle='round')

        x0,y0,width,heigth = axes[1].get_position().bounds
        cbar_ax = fig.add_axes([x0, y0-0.04, width, 0.03])
        c = fig.colorbar(mpl.cm.ScalarMappable(norm=colors_norm, cmap=cmap), cax=cbar_ax,orientation='horizontal')
        c.set_label('relative change')
        c.set_ticks([-1,-0.1,-0.01,0,0.01,0.1,1])
        c.set_ticklabels(["-100%","-10%","-1%", "","+1%","+10%","+100%"])

    ax.margins(0)
    ax.axis('off')

def create_flow_dataset(data, source_locations, target_locations, A, B):
    df_temp = pd.DataFrame(list(zip(source_locations, target_locations, A, B, data)), columns=['origin_area_code','destination_area_code','A','B','total_normalized'])
    return df_temp[df_temp.origin_area_code!=df_temp.destination_area_code]


def get_lockdown_dates(countries, threshold = 65, plot = False):
    w = 'https://covidtrackerapi.bsg.ox.ac.uk/api/v2/stringency/date-range/2020-01-01/2020-12-31'
    response = requests.get(w)
    a = response.json()
    data = [[list(a['data'][date][country].values()) for date in a['data'].keys() if country in a['data'][date]] for country in countries]
    df = pd.DataFrame(np.concatenate(data),columns =a['data']['2020-12-31']['ESP'].keys())

    df['date_value'] = df['date_value'].apply(lambda x: dt.datetime.strptime(x,'%Y-%m-%d'))
    df['stringency'] = df['stringency'].astype(float)
    df['stringency_actual'] = df['stringency_actual'].astype(float)
    df['stringency_legacy'] = df['stringency_legacy'].astype(float)
    if plot == True:
        plt.figure(figsize = (6.3,3))
        df.sort_values(by='date_value').set_index('date_value').groupby('country_code')['stringency'].plot(marker='.', xlabel='Date', ylabel='Stringency')
        plt.axhline(65,color='k', linestyle='--')
        plt.legend()

    lockdown_dates = df[df.stringency >= threshold].groupby('country_code')
    return lockdown_dates

def create_fb_figures(country, N, adm, link, file_type, return_nmf = False, population = False, ax = None, save_fig = False, lockdown = None, overall = False, return_nmf_only = False, path_full_file = None):
    if population:
        extension = '_pop'
    else:
        extension = ''
    country_conversion = {'Spain':'ESP','Denmark':'DNK','France':'FRA','Italy':'ITA'}
    if path_full_file:
        with open(f'fb_{country}.pkl', 'rb') as f:
            X_fb, fb_locations, dates = pickle.load(f)
        
    else:
        print('Path is censored')
        print('Run with "path_full_file = Ture"')
#         path_face = f"censored"
#         fb_files = sorted(listdir(path_face))[:-9]
#         fb_files = [fb_files[i:i+3] for i in np.arange(0,len(fb_files),3)]
#         fb_dates = [i[0] for i in fb_files]
#         with open(f"censored",'rb') as f:
#             shape = pickle.load(f)
#         fb_locations = sorted(shape[f"adm{adm}"].unique())
#         loc_replace_fb = {loc: i for i,loc in enumerate(fb_locations)}

#         if link:
#             X_fb = np.zeros([len(fb_locations)**2,len(fb_files)])
#         else:
#             X_fb = np.zeros([len(fb_locations), len(fb_files)])

#         for idx, fb_dates_i in enumerate(fb_files):
#             df_fb = pd.concat([pd.read_csv(path_face + file) for file in fb_dates_i],axis=0)
#             df_fb = df_fb[["n_crisis",f"start_adm{adm}",f"end_adm{adm}"]].groupby([f"start_adm{adm}",f"end_adm{adm}"]).sum().reset_index().to_numpy()
#             for source, target, n_crisis in df_fb:
#                 target = target.rstrip()
#                 source = source.rstrip()
#                 i = int(loc_replace_fb[source]) # Source
#                 idy = int(loc_replace_fb[target])    # Target
#                 if link:
#                     X_fb[idy+i*len(fb_locations),idx] += n_crisis
#                 else:
#                     X_fb[i,idx] += n_crisis

        dates = pd.to_datetime(["-".join(i.split("_")[1:4])+" 00:00:00" for i in fb_dates])
    X_fb_return = X_fb.copy()
                
    if population:
        population = get_pop_country(country)['population'].astype(int).to_list()
        population = np.array([population[i//len(fb_locations)] for i in range(len(fb_locations)*len(fb_locations))])
        X_fb = X_fb/population[:,None] 
    X_sqrt_norm_fb = np.sqrt(X_fb)
    #X_sqrt_norm_fb = X_sqrt_fb/X_sqrt_fb.max()

    #NMF model
    model = NMF(n_components=N, init = 'random', random_state=0,max_iter=100000)
    nmf_topics_fb = model.fit_transform(X_sqrt_norm_fb)
    #nmf_topics = model.fit_transform(X_norm)
    nmf_weights_fb = model.components_
    nmf_topics_fb = nmf_topics_fb[:,[2,0,1]]
    nmf_weights_fb= nmf_weights_fb[[2,0,1],:]
    weekend_idx = [i for i in range(nmf_weights_fb.shape[1]-1) if dates[i].isoweekday()>=6]
    
    if return_nmf_only == True:
        return nmf_topics_fb, nmf_weights_fb, dates, X_fb_return
        
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(20,10))

    for idx in weekend_idx:
        ax.axvspan(dates[idx], dates[idx+1], facecolor='gray', alpha=0.3, edgecolor='none')

    for i in range(nmf_weights_fb.shape[0]):
        ax.plot(dates,nmf_weights_fb[i,:],linewidth=1)

    if country =='France':
        ax.legend(['Holiday', 'Weekend', 'Workday'])
    ax.set_title(f"{country}",fontsize = 12, loc = 'left')
    if country=='Italy':
        ax.set_ylabel(f"Loadings")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    if lockdown:
        lockdown_dates = lockdown.get_group(country_conversion[country]).date_value.to_list()
        for period in lockdown_dates:
            if period<=dates[-1]:
                ax.axvspan(period-dt.timedelta(hours=12),period+dt.timedelta(hours=12),facecolor='red', alpha=0.2, edgecolor='none')
    
#     if save_fig: 
#         plt.savefig(f"some path")

    if return_nmf:
        return nmf_topics_fb, nmf_weights_fb, dates, X_fb_return
    
def get_pop_country(country):
    loc_rename = {}
    missing_loc = []
    drop_loc = []
    if country == 'France':
        df = pd.read_html('https://en.wikipedia.org/wiki/List_of_French_departments_by_population')[0]
        df = df.rename(columns = {'Legal population in 2013':'population', 'Department':'kommune'})[['kommune','population']]
        drop_loc = ['Guadeloupe', 'Guyane', 'Martinique', 'Mayotte', 'Réunion']
    elif country == 'Spain':
        df = pd.read_html('https://en.wikipedia.org/wiki/List_of_subdivisions_of_Spain_by_population')[2]
        df = df.rename(columns = {'Province':'kommune',r'Population as of1 January 2013[2]':'population'})[['kommune', 'population']]
        loc_rename = {'Navarre':'Navarra','Seville':'Sevilla','Gipuzkoa':'Guipúzcoa','Balearic Islands':'Baleares','La Coruña':'A Coruña',
                     'Biscay':'Vizcaya'}
        drop_loc = ['Spain']
    elif country == 'Italy':
        loc_rename = {'Bolzano   Bozen':'Bolzano',"Firenze":'Florence','Monza e della Brianza':'Monza and Brianza','Reggio di Calabria':'Reggio Di Calabria',
                      'Reggio nell Emilia':'Reggio Nell Emilia', 'Forlì Cesena':'Forli Cesena', 'Siracusa':'Syracuse', 'Padova':'Padua',
                      'Pesaro e Urbino': 'Pesaro E Urbino', 'Mantova':'Mantua', 'Valle d Aosta':'Aosta', 'Valle d Aosta   Vallée d Aoste':'Aosta', 
                     }
        missing_loc = [['Carbonia Iglesias', 124239],['Medio Campidano', 96774],['Ogliastra',56362],['Olbia Tempio',161360]]
        drop_loc = ['Sud Sardegna']
        
        
        df = pd.read_csv('provinces.csv', sep=';')
        df = df.rename(columns={'Territorio':'kommune','Unnamed: 1':'population'})
        df = df[df['kommune'].str.startswith(' '*6)]
        df['kommune'] = df['kommune'].str.lstrip()
        
    elif country == 'Denmark':
        _, df = utils.load_demo_and_pop()
        
    #Removing special cases
    df['kommune'] = df['kommune'].str.replace(r"[-_'/]" ," ") 
    
    #Replacing location names
    df['kommune'] = df['kommune'].replace(loc_rename)
    
    #Adding missing locations manually
    for row in missing_loc:
        df = df.append({'kommune':row[0],'population':row[1]}, ignore_index = True)
        
    df = df.set_index('kommune')
    df = df.drop(drop_loc).sort_index()
    return df


def get_corona_numbers(country = 'Denmark', dates = None):
    print('Fetching Corona numbers')
    corona_daily_cases = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv")
    corona_daily_cases = corona_daily_cases[['location','new_deaths','date']]
    corona_daily_cases.date = pd.to_datetime(corona_daily_cases.date)
    corona_daily_cases.new_deaths = corona_daily_cases.new_deaths.fillna(0)

    #Computing the daily cases
    t_delta = dt.timedelta(days=13)
    nmf_weekends = [1 if (date.isoweekday()>=6) else 0 for date in dates]
    nmf_vacation = [1 if (vacation_check(date, country)) else 0 for date in dates]
    corona_daily_country = corona_daily_cases[corona_daily_cases.location == country]
    mask = (corona_daily_country.date >= np.min(dates)+t_delta) & (corona_daily_country.date<= np.max(dates)+t_delta)
    corona_daily_country = corona_daily_country.loc[mask]
    corona_daily_country[corona_daily_country.new_deaths<0]=0
    cases = corona_daily_country.new_deaths.tolist()
    weekends = nmf_weekends
    vacation = nmf_vacation
    c_deaths = cases
    c_weekends = weekends
    c_workdays = [1-i for i in c_weekends]
    c_vacation = vacation
    return c_deaths, c_weekends, c_workdays, c_vacation

def vacation_check(date, country):
    if country == 'France':
        summer_start = dt.datetime(2020,7,4)
        summer_end = dt.datetime(2020,9,1)
    elif country =='Spain':
        summer_start = dt.datetime(2020,6,18)
        summer_end = dt.datetime(2020,9,12)
    elif country == 'Italy':
        summer_start = dt.datetime(2020,6,8)
        summer_end = dt.datetime(2020,9,17)
    elif country == 'Denmark':
        summer_start = dt.datetime(2020,6,29)
        summer_end = dt.datetime(2020,8,7)
    if date in holiday_module.CountryHoliday(country) or (summer_start<=date<=summer_end):
        return 1
    else:
        return 0
    
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def get_cv_curve(data, t_method, subset_n, N_range):
    # Amount of Ns to check
    Ns = np.arange(1,N_range)
    
    # Transform data
    data.t_method = t_method
    data.transform_X()
    
    # Method for CV best N is explained in "Structural and functional discovery in dynamic networks with non-negative matrix factorization" 
    # Get data matrix
    X = data.X_t

    # Compute subset masks.
    I_r = np.array_split(random.sample(range(X.shape[0]), X.shape[0]), subset_n)
    I_c = np.array_split(random.sample(range(X.shape[1]), X.shape[1]), subset_n)

    # Initilize error list
    error = np.zeros(len(Ns))

    # Loop over amount of topics
    for i, N in tqdm(enumerate(Ns),total = len(Ns)):
        # Loop over subsets
        for sub in range(subset_n):
            # Compute data mask.
            mask_r, mask_c = np.ones(X.shape[0], dtype=bool), np.ones(X.shape[1], dtype=bool)
            mask_r[I_r[sub]], mask_c[I_c[sub]] = False, False 

            # Compute NMF on the dataset without the given subset.
            W, H, _ = non_negative_factorization(X[mask_r,:][:,mask_c], n_components=N, init = 'random', random_state=0,max_iter=100000)
            # Compute NMF with fixed H from the leave one subset dataset and use it on the dataset where we use that specifik subset
            W_r, _ , _ = non_negative_factorization(X[~mask_r,:][:,mask_c], n_components=N, init = 'custom', update_H=False, H=H, random_state=0,max_iter=100000)
            # Compute NMF with fixed W from the leave one subset dataset and use it on the dataset where we use that specifik subset
            # Since non_negative_factorization only has the possibility to lock H we transpose the dataset and W to fake H.
            H_c, _ , _ = non_negative_factorization(X[mask_r,:][:,~mask_c].T, n_components=N, init = 'custom', update_H=False, H=W.T, random_state=0,max_iter=100000)
            # Compute the reconstruction of the subset given the two subset reconstructed H and W
            X_recon = W_r@H_c.T
            # Compute the error from the original reconstructioW_r@H_c.n
            error[i] += np.linalg.norm(X_recon-X[~mask_r,:][:,~mask_c],2)
    return error