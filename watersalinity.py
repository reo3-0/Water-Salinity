# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 21:29:20 2021

@author: Ruairi
"""
import os
import requests
from bs4 import BeautifulSoup 
import pandas as pd
import geopandas 
from pandas_datareader import wb 
import wbdata
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import statsmodels.formula.api as smf
from math import log

def skip_this_csv(name, rivers, lakes, ground):
    """
    A helper function to check each of the read_and_download conditions to 
    see if the user does not want to use any of the 3 available raw datasets.
    """
    if('river' in name.lower() and not rivers):
        return True
    elif('lake' in name.lower() and not lakes):
        return True
    elif('ground' in name.lower() and not ground):
        return True
    else:
        return False

def read_and_download_primary_datasets(path, rivers=True, lakes=True, ground=True):
    # https://stackoverflow.com/questions/5815747/beautifulsoup-getting-href
    # https://stackoverflow.com/questions/9572490/find-index-of-last-occurrence-of-a-substring-in-a-string?rq=1
    # https://stackoverflow.com/questions/45978295/saving-a-downloaded-csv-file-using-python
    # https://stackoverflow.com/questions/58257251/how-to-check-if-a-file-exists-in-another-folder
    print("Warning: This data requires 1.5 GB of storage ")
    url = 'https://doi.pangaea.de/10.1594/PANGAEA.913939?format=html#download'
    website = requests.get(url)
    assert website, "Issue with website. Request not found."
    soup = BeautifulSoup(website.text, 'lxml')
    dl_links = soup.find_all('a', {'class':"dl-link"})
    database_csv_links = [link['href'] for link in dl_links if 'database.csv' in link['href']]

    csv_dict = {}
    for csv_link in database_csv_links:
        name = csv_link[csv_link.rfind('/')+1:]
        if(skip_this_csv(name, rivers, lakes, ground)):
            continue
        else:
            if(not os.path.isfile(os.path.join(path, name))): #os.exists??
                print(f'Downloading {name}...' )
                this_df = pd.read_csv(csv_link, low_memory=False)
                csv_dict[name] = this_df
                this_df.to_csv(os.path.join(path,name)) # Save csv to repo. 
            else:
                print(f'{name} already exists. Reading it in...')
                csv_dict[name] = pd.read_csv(os.path.join(path,name), low_memory=False)
    return csv_dict
  
def convert_EC_to_floats(EC_val):
    """
    A helper function to fix an issue in the EC values and convert to floats.
    """
    if(isinstance(EC_val,int) or isinstance(EC_val,float)):
        EC_val = float(EC_val)
    else:
        EC_val = float(EC_val.replace(',','.'))
    return EC_val
     
def clean_salinity_country_names(country_name):
    """
    A helper function to clean the salinity data country names so they match 
    the natrualearth_lowres dataset.
    """
    if(country_name == 'UK'):
        return 'United Kingdom'
    elif(country_name =='USA'):
        return 'United States of America'
    else:
        return country_name

def merge_dataset_list(csv_dict):
    # Groundwater dataset has some conversions we need to use to fill NAs
    if('Groundwaters_database.csv' in csv_dict.keys()):
        df_groundwater = csv_dict['Groundwaters_database.csv']
        # https://stackoverflow.com/questions/30357276/how-to-pass-another-entire-column-as-argument-to-pandas-fillna
        df_groundwater['EC'] = df_groundwater['EC'].fillna(df_groundwater['EC_conv'])
        df_groundwater = df_groundwater.drop(['Depth', 'TDS', 'EC_conv'], axis=1)
        csv_dict['Groundwaters_database.csv'] = df_groundwater
    final_df = pd.concat(list(csv_dict.values()))
    final_df['Year'] = pd.DatetimeIndex(final_df['Date']).year
    final_df['Month'] = pd.DatetimeIndex(final_df['Date']).month
    final_df['Day'] = pd.DatetimeIndex(final_df['Date']).day
    final_df['EC'] = final_df['EC'].apply(convert_EC_to_floats)
    final_df['Country'] = final_df['Country'].apply(clean_salinity_country_names)
    return final_df

def agg_dataset_stations_by(df, *levels):
    # https://note.nkmk.me/en/python-args-kwargs-usage/
    agg_df = df.groupby(['Station_ID']+list(levels),
                         as_index=False).agg(Avg_EC=('EC','mean'),
                                             Count=('Station_ID','count'),
                                             Lat=('Lat','first'),
                                             Lon=('Lon','first'),
                                             Continent=('Continent','first'),
                                             Country=('Country','first'))
    return agg_df

def match_nan_coord_with_geo(df):
    """
    A function to take a dataset with lat/long coordinates and uses geopandas
    naturalearth_lowres to match coordinates with country names. 
    Note: There are about 100 rows that do not get matched for my data.
    """
    df_nans = df[df['Country'].isna()].copy()
    df_nans_geo = geopandas.GeoDataFrame(df_nans, 
                                         geometry=geopandas.points_from_xy(df_nans['Lon'],
                                                                           df_nans['Lat']))
    df_nans_geo.set_crs(epsg=4326, inplace=True)
    world_geo = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    df_geo_merged_nans_world = world_geo.sjoin(df_nans_geo, how='inner', predicate='contains')
    df_pd_merged_nans_world = pd.DataFrame(df_geo_merged_nans_world)
    # https://stackoverflow.com/questions/56842140/pandas-merge-dataframes-with-shared-column-fillna-in-left-with-right
    df.update(df[['Country']].merge(df_pd_merged_nans_world,
                                    how='left',
                                    left_on='Country',
                                    right_on='name'))      
    return df  

def agg_countries_by_yr(df, outlier_cutoff=False):
    print("Aggregating data to year-level. This may take a moment...")
    # lat/lon chooses median value to approximate country's distance from equator.
    df = match_nan_coord_with_geo(df)
    if(outlier_cutoff):
        df = df[df['EC'] < outlier_cutoff]
    agg_df = df.groupby(['Country', 'Water_type','Year'],
                         as_index=False).agg(Avg_EC=('EC','mean'),
                                          Count=('Station_ID','count'),
                                          Stat_ID_Count = ('Station_ID', 'nunique'),
                                          Lat=('Lat','median'),
                                          Lon=('Lon','median'),
                                          Continent=('Continent','first'),
                                          Country=('Country','first'),
                                          Year=('Year','first'))
    agg_df['Year'] = agg_df['Year'].apply(lambda x: int(x))
    return agg_df

def match_country_name_to_wb_code(country_name):
    """
    A helper function to map a full country name to its world bank iso2Code
    so that wb.download can easily recognize countries of interest. 
    """
    # https://mcmayer.net/getting-worldbank-data-with-python-easily/
    countries = wbdata.get_country()
    country_code = "Not Found"
    for country_dict in countries:
        if(country_dict['name'] == country_name):
            country_code = country_dict['iso2Code']
        elif(country_name in country_dict['name']):
            country_code = country_dict['iso2Code']
        elif('Slovak' in country_dict['name'] and 'Slovak' in country_name):
            country_code = country_dict['iso2Code']
        elif('United States' in country_dict['name'] and 'United States' in country_name):
            country_code = country_dict['iso2Code']
    return country_code

def read_in_wb_data_and_merge(agg_df, save_title=False):
    # Predictor dictionary that is easily adjusted by the user.
    indicators_wb = {'SE.XPD.PRIM.PC.ZS':'Expend_Per_Stud',
                     'SE.XPD.TOTL.GB.ZS': 'Perc_Govt_Expend_on_Edu',
                     'SE.SEC.ENRR': 'Perc_Sec_Enroll_Gross',
                     'SE.SEC.NENR': 'Perc_Sec_Enroll_Net',
                     'SP.URB.TOTL.IN.ZS': 'Perc_Urban_Pop',
                     'EG.ELC.ACCS.ZS': 'Perc_Elec_Access',
                     'AG.LND.IRIG.AG.ZS': 'Perc_Agro_Land_Irr',
                     'NY.GDP.PCAP.CD': 'GDP_Per_Cap',
                     'NY.GDP.PCAP.PP.CD': 'GDP_Per_Cap_PPP'}
    # Modifies the aggregated primary data country names so wb data merges. 
    agg_df['Country_Code'] = agg_df['Country'].apply(match_country_name_to_wb_code)
    agg_df = agg_df[agg_df['Country_Code'] != 'Not Found'] # Drops French Guiana
    df_wb = wb.download(indicator=list(indicators_wb.keys()),
                        country=list(agg_df['Country_Code'].unique()),
                        start=int(agg_df['Year'].min()),
                        end=int(agg_df['Year'].max())).reset_index()
    df_wb = df_wb.rename(columns=indicators_wb)
    df_wb['Country_Code'] = df_wb['country'].apply(match_country_name_to_wb_code)
    df_wb['year'] = df_wb['year'].apply(lambda x: int(x))
    # Lastly, merge the secondary WB data with the year-aggregated salinity.
    df_final = agg_df.merge(df_wb, 
                            how='left',
                            left_on=['Country_Code', 'Year'], 
                            right_on=['Country_Code', 'year'])
    df_final = df_final.drop(columns=['country','year'])
    if(save_title):
        df_final.to_csv(os.path.join(path,f'{save_title}.csv'))
    return df_final

# Summary Stats: Tables

def group_EC_by_range(EC_val):
    if(EC_val >= 85000):
        return 'EC>85000'
    elif(EC_val >= 60000 and EC_val < 85000):
        return '60000>EC>85000'
    elif(EC_val >= 45000 and EC_val < 60000):
        return '45000>EC>60000'
    elif(EC_val >= 3000 and EC_val < 45000):
        return '3000>EC>45000'
    elif(EC_val >= 800 and EC_val < 3000):
        return '800>EC>3000'
    elif(EC_val >= 300 and EC_val < 800):
        return '300>EC>800'
    elif(EC_val < 300):
        return 'EC<300'

def summary_EC_distribution(df, save_title=False):
    df['EC_Range'] = df['EC'].apply(group_EC_by_range)
    # https://stackoverflow.com/questions/19384532/get-statistics-for-each-group-such-as-count-mean-etc-using-pandas-groupby
    summary = df.groupby(['Water_type',
                          'EC_Range']).size().reset_index()
    summary = summary.rename(columns={0 : 'Percent'})
    summary['Percent'] = summary['Percent'].apply(lambda x : x/len(df))
    summary_wide = summary.pivot(index='Water_type',
                                 columns='EC_Range',
                                 values='Percent').reset_index()
    # https://stackoverflow.com/questions/20804673/appending-column-totals-to-a-pandas-dataframe/56720533
    summary_wide.loc['Total'] = summary_wide.sum()
    if(save_title):
        summary_wide.to_html(os.path.join(path, f'{save_title}.html'),
                             index=False)
    return summary_wide[['Water_type', 'EC<300', '300>EC>800', '800>EC>3000',
                         '3000>EC>45000', '45000>EC>60000', '60000>EC>85000',
                         'EC>85000']]

def continent_sum_stats(df_final, dataset='Primary', by_water_type=False, save_title=False):
    '''
    Takes the aggregated, country-level dataset and generates basic summary
    statistics for the primary salinity or secondary data.
    It also has an option to aggregate by water type and to save to html. 
    '''
    assert dataset in ['Primary', 'Secondary'], "Dataset must be 'Primary' or 'Secondary"
    if(dataset == 'Primary'):
        if(by_water_type):
            grouping = ['Continent', 'Water_type']
        else:
            grouping = ['Continent']
        # https://stackoverflow.com/questions/18554920/pandas-aggregate-count-distinct
        df_prime_stats = df_final.groupby(grouping, 
                                          as_index=False).agg(Avg_EC=('Avg_EC','mean'),
                                                              Max_EC=('Avg_EC','max'),
                                                              n_Country=('Country','nunique'),
                                                              n_obs=('Count','sum'),
                                                              Avg_Station_Count=('Stat_ID_Count','mean'),
                                                              Avg_yr=('Year','mean'))
        df_final_summary = df_prime_stats
    elif(dataset == 'Secondary'):
        secondary_columns = df_final.columns[-9:]
        df_sec_stats = df_final[secondary_columns].describe()
        # https://stackoverflow.com/questions/26266362/how-to-count-the-nan-values-in-a-column-in-pandas-dataframe
        percent_na = list(df_final.isnull().sum(axis = 0)/len(df_final))[-len(df_sec_stats.columns):]
        # https://www.kite.com/python/answers/how-to-append-a-list-as-a-row-to-a-pandas-dataframe-in-python
        df_sec_stats.loc[len(df_sec_stats)] = percent_na
        # https://note.nkmk.me/en/python-pandas-dataframe-rename/
        df_sec_stats = df_sec_stats.rename(index={8:'Percent Missing Data'})
        df_final_summary = df_sec_stats        
        
    if(save_title):
        df_final_summary.to_html(os.path.join(f'{save_title}.html'))
    return df_final_summary
                                
# Summary Stat: Plots
# Colors: https://matplotlib.org/stable/gallery/color/named_colors.html

def plot_missing_yr_rates(df_final, save_title=False):
    df_cntry_yr_agg = df_final.groupby(['Country','Year'],
                                     as_index=False).agg(Avg_EC=('Avg_EC',
                                                                 'mean'))
    # https://thispointer.com/pandas-convert-dataframe-index-into-column-using-dataframe-reset_index-in-python/                                                                    
    # https://stackoverflow.com/questions/40575067/matplotlib-bar-chart-space-out-bars/40575741
    yr_cntry_summary = df_cntry_yr_agg.pivot(index='Country', 
                                         columns='Year',
                                         values='Avg_EC').reset_index()
    yr_cntry_summary['Missing'] = yr_cntry_summary.isnull().sum(axis=1)/len(yr_cntry_summary.columns)
    df_in_order = yr_cntry_summary.sort_values('Missing')
    fig_bar, ax_bar = plt.subplots(figsize=(5,12))
    # colors = cm.get_cmap(viridis, len(agg_df['Country'].unique()))
    ax_bar.barh(df_in_order['Country'],
                df_in_order['Missing'],
                label=df_in_order['Country'],
                color='firebrick')
    ax_bar.set_xlabel('Proportion of Years with Missing Data',
                      weight='bold')
    ax_bar.set_title('Proportion of Missing Years from 1980-2019',
                     weight='bold')
    ax_bar.set_facecolor('lightgrey')
    fig_bar.set_facecolor('lightgrey')
    if(save_title):
        # https://pretagteam.com/question/python-plot-cut-off-when-saving-figure
        fig_bar.savefig(os.path.join(f'{save_title}.png'),
                        bbox_inches='tight')

def plot_agg_hist(df, var, n_bin, neat=False, save_title=False):
    """
    Function that allows the user to plot a histogram of any single variable 
    in a dataframe (for this project, it is intended to be used for the 
    the aggregated final dataframe)
    """
    fig_hist, ax_hist = plt.subplots(figsize=(12,5))
    ax_hist.yaxis.grid(color='grey', linestyle='--')
    if(not neat):
        ax_hist = sns.histplot(data=df, x=var)
    else:
        n, bins, patches = ax_hist.hist(df[var],
                                        bins=n_bin,
                                        edgecolor='black',
                                        rwidth=0.9,
                                        color='darkviolet')
        ax_hist.xaxis.set_major_formatter(FormatStrFormatter('%0.0f'))
        bin_w = (bins[1] - bins[0]) 
        ax_hist.set_xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w))
        ax_hist.set_xlim(bins[0], bins[-1])
    ax_hist.set_ylabel('Frequency', weight='bold')
    var_name = var.replace('_', ' ')
    ax_hist.set_xlabel(var_name, weight='bold')  
    # https://stackabuse.com/rotate-axis-labels-in-matplotlib/
    ax_hist.tick_params(axis='x', labelrotation = 45)
    ax_hist.set_title(f'Histogram of {var_name}', weight='bold')
    if(save_title):
        fig_hist.savefig(f'{save_title}.png')

# Other Helpful (Non-Spatial) Plots

def plot_timeseries_by_country(df, var, water_type, country_list=[], save_title=False):
    if(len(country_list) > 0):
        plot_df = df[df['Country'].isin(country_list)]
    else:
        plot_df = df
    plot_df = plot_df[plot_df['Water_type'] == water_type]
    # https://stackoverflow.com/questions/38197964/pandas-plot-multiple-time-series-dataframe-into-a-single-plot
    fig_ts, ax_ts = plt.subplots()
    for country_name, filtered_data in plot_df.groupby(['Country']):
            ax_ts.plot(filtered_data['Year'],
                       filtered_data[var],
                       label=country_name)
            ax_ts.set_title(f'{var} Timeseries')
            # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
            ax_ts.legend(loc='center left', bbox_to_anchor=(1,0.5),
                       fancybox=True, shadow=True)
            # https://www.w3schools.com/python/matplotlib_grid.asp
            ax_ts.grid(color='grey', linestyle='--', linewidth=0.25)
            ax_ts.set_facecolor('whitesmoke')
            fig_ts.set_facecolor('whitesmoke')
            cleaned_var_name = var.replace('_',' ')
            ax_ts.set_ylabel(cleaned_var_name)
            ax_ts.set_title(f'{cleaned_var_name} Timeseries (for {water_type} data)')
    if(save_title):
        fig_ts.savefig(f'{save_title}.png',
                       bbox_inches='tight')

# Static Spatial Plotting

def reconcile_world_country_names(this_country, df):
    """
    A helper function that changes the country names in naturalearth_lowres 
    based on the desired dataframe being merged.
    I did somewhat hardcode the two countries that don't match in this case,
    but my hope is by organizing as such it allows users to easily add more
    if-cases.
    """
    countries_to_match = df['Country'].unique()
    updated_country = this_country
    for country in countries_to_match:
        if(this_country.startswith('Bosnia') and country.startswith('Bosnia')):
            updated_country =  country
        elif(this_country.startswith('Czech') and country.startswith('Czech')):
            updated_country = country
    return updated_country

def world_pd_2_world_geo(df_final):
    """
    A helper function that takes a pd dataframe and merges it to the 
    naturalearth_lowres and returns a merged geopandas df.
    """
    world_geo = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    world_geo['name'] = world_geo['name'].apply(lambda x : reconcile_world_country_names(x, df_final))
    df_merged_pandas = df_final.merge(world_geo[['name', 'geometry']], left_on='Country', right_on='name')
    df_merged_geo = geopandas.GeoDataFrame(df_merged_pandas, geometry='geometry')
    df_merged_geo.set_crs(epsg=4326)
    return df_merged_geo

def plot_grand_summary_world_map(df_final, summary_var, save_title=False):
    df_final_geo = world_pd_2_world_geo(df_final)
    df_grand_total = df_final_geo.groupby('Country').aggregate(geometry=('geometry','first'),
                                                               Count=('Count','sum'),
                                                               Avg_EC=('Avg_EC','mean'),
                                                               Avg_Ann_Station_Count=('Stat_ID_Count', 'mean'))
    fig_total, ax_total = plt.subplots(figsize=(7,7))
    divider = make_axes_locatable(ax_total)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    ax_total = df_grand_total.plot(ax=ax_total,
                                   column=summary_var,
                                   edgecolor='grey',
                                   legend=True,
                                   cmap='Spectral',
                                   cax=cax)
    ax_total.set_title(f'Total {summary_var} Across All Data and Years', 
                       weight='bold')
    if(save_title):
        fig_total.savefig(os.path.join(path,f'{save_title}.png'))

def plot_single_var_choropleth(df_final, year, water_type, var, save_title=False):
    # https://stackabuse.com/change-figure-size-in-matplotlib/
    # https://gis.stackexchange.com/questions/152920/changing-colours-in-geopandas
    color_map_dict = {'Count': 'GnBu', 'Avg_EC': 'OrRd'}
    df_final_geo = world_pd_2_world_geo(df_final)
    this_yr_df = df_final_geo[df_final_geo['Year'] == year]
    this_yr_df = this_yr_df[this_yr_df['Water_type'] == water_type]
    if(var == 'General Overview'):
        fig_chor, ax_chor = plt.subplots(1,2,figsize=(15,15))
        cols = ['Count', 'Avg_EC']
        for col_ind in range(len(cols)):
            this_ax = ax_chor[col_ind]
            divider = make_axes_locatable(this_ax)
            cax = divider.append_axes('right', size='5%', pad=0.1)
            this_ax = this_yr_df.plot(ax=this_ax,
                             column=cols[col_ind],
                             edgecolor='grey',
                             legend=True,
                             cmap=color_map_dict[cols[col_ind]],
                             cax=cax)
            this_ax.set_title(f'Density of {water_type} Stations {cols[col_ind]} in {year}',
                              weight='bold')
    else:
        fig_chor, ax_chor = plt.subplots(figsize=(7,7))
        divider = make_axes_locatable(ax_chor)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        ax_chor = this_yr_df.plot(ax=ax_chor,
                         column=var,
                         edgecolor='grey',
                         legend=True,
                         cmap='viridis',
                         cax=cax)
        ax_chor.set_title(f'Density of {var} for Nations with {water_type} Stations in {year}',
                          weight='bold')
    if(save_title):
        fig_chor.savefig(os.path.join(path, f'{save_title}.png'),
                         bbox_inches='tight')
        
def plot_2_by_2_choropleth(df_final, year, var_list, save_title=False):
    colors = ['PuRd_r', 'cividis', 'YlOrBr_r', 'RdBu']
    color_map_dict = dict(zip(var_list, colors))
    df_final_geo = world_pd_2_world_geo(df_final)
    this_df = df_final_geo[df_final_geo['Year'] == year]
    fig_chlor, ax_chlor = plt.subplots(2,2,figsize=(10,5), constrained_layout=True)
    # https://www.kite.com/python/answers/how-to-set-the-spacing-between-subplots-in-matplotlib-in-python
    fig_chlor.tight_layout(pad=0.5)
    fig_chlor.set_facecolor('lightsteelblue')
    axs_flat = [ax for sublist in ax_chlor for ax in sublist]
    for this_ax, var in zip(axs_flat, var_list):
        divider = make_axes_locatable(this_ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        this_ax = this_df.plot(ax=this_ax,
                                  column=var,
                                  edgecolor='black',
                                  legend=True,
                                  cmap=color_map_dict[var],
                                  cax=cax)
        var_title = var.replace('_',' ')
        this_ax.set_title(f'{var_title} in {year}', weight='bold')
        this_ax.set_xticklabels([])
        this_ax.set_yticklabels([])
        this_ax.set_facecolor('lightsteelblue')
    if(save_title):
        fig_chlor.savefig(os.path.join(path, f'{save_title}.png'),
                          bbox_inches='tight')

# Regression analysis

def regress_salinity_on(df_final, predictor_list, log_dep=True, log_all=False, save_title=False):
    df_reg = df_final.copy()
    # Convert longitude to a useful regressor: abs(Lon) (dist from equator)
    if('Lon' in predictor_list):
        df_reg['Abs_Lon'] = df_reg['Lon'].apply(abs)
        predictor_list.remove('Lon')
        predictor_list.append('Abs_Lon')
    if(log_dep):
        df_reg['log_avg_EC'] = df_reg['Avg_EC'].apply(log)
        model = smf.ols('log_avg_EC ~ ' + ' + '.join(predictor_list),
                        data=df_reg)
    elif(log_all):
        for predictor in predictor_list + ['Avg_EC']:
            if(isinstance(df_reg[predictor][0], (int, float, np.int64))):
                df_reg['log_' + predictor] = df_reg[predictor].apply(log)
                if(predictor != 'Avg_EC'):
                    predictor_list.remove(predictor)  
                    predictor_list.append('log_' + predictor)
        model = smf.ols('log_Avg_EC ~ ' + ' + '.join(predictor_list),
                        data=df_reg)
    else:
        model = smf.ols('Avg_EC ~ ' + ' + '.join(predictor_list),
                        data=df_reg)
    result = model.fit()
    return result.summary()  



# Putting it all together: 

# Final Raw Data can be found here:
# https://uchicago.box.com/s/22cq9f5aey64gdpof5moeddw8g86pq4n    

path = r'C:\Users\Ruairi\documents\github\final-project-final-project-ruairi-ocearuil'

# These first lines will take approximately 5 minutes...  
csv_dict = read_and_download_primary_datasets(path)
df_raw_salinity = merge_dataset_list(csv_dict)
df_agg_sal_country_yr = agg_countries_by_yr(df_raw_salinity) 
df_final = read_in_wb_data_and_merge(df_agg_sal_country_yr, save_title='df_final')
df_agg_sal_country_yr_rm_outliers = agg_countries_by_yr(df_raw_salinity, outlier_cutoff=85000)
df_final_rm_outliers = read_in_wb_data_and_merge(df_agg_sal_country_yr_rm_outliers, save_title='df_final_reduced')

# Summary tables 
sum_dist = summary_EC_distribution(df_raw_salinity, save_title='Saved_salinity_ranges')

continent_sum_stats(df_final_rm_outliers, 
                    'Primary', 
                    by_water_type=True,
                    save_title='Saved_Primary_Data_Summary_Stats')

continent_sum_stats(df_final_rm_outliers, 
                    'Secondary', 
                    save_title='Saved_Secondary_Data_Summary_Stats')

# Summary (and other useful) plots
plot_missing_yr_rates(df_final_rm_outliers,
                      save_title="Saved_Percent_Yrs_Missing_by_Country")

plot_agg_hist(df_final_rm_outliers, 
              'GDP_Per_Cap_PPP', 
              n_bin=50, neat=True, 
              save_title='Saved_Hist_GDP_Per_Cap_PPP')

plot_timeseries_by_country(df_final_rm_outliers,
                           var='Avg_EC', 
                           water_type='River',
                           country_list=['Germany',
                                         'France',
                                         'United Kingdom',
                                         'Switzerland',
                                         'Netherlands',
                                         'Ireland',
                                         'Luxembourg'],
                           save_title='Saved_Timeseries_Avg_EC_Riv')

# Spatial Plots
plot_grand_summary_world_map(df_final_rm_outliers, 
                             'Count',
                             save_title='Saved_Total_Measurement_Counts')

plot_grand_summary_world_map(df_final_rm_outliers, 
                             'Avg_EC',
                             save_title='Saved_Total_Avg_EC')

plot_single_var_choropleth(df_final_rm_outliers, 
                 year=2015, 
                 water_type='Lake/Reservoir', 
                 var='Perc_Sec_Enroll_Gross',
                 save_title='Saved_Choro_Sec_Enroll_Lakes')

plot_2_by_2_choropleth(df_final_rm_outliers,
                        year=2017, 
                        var_list=['Perc_Urban_Pop',
                                  'Perc_Elec_Access',
                                  'GDP_Per_Cap_PPP',
                                  'Perc_Sec_Enroll_Gross'],
                        save_title='Saved_Choro_2_by_2') 

# Regression analysis
regress_salinity_on(df_final_rm_outliers,
                    ['Year', 'GDP_Per_Cap_PPP', 'Lon',
                     'Perc_Urban_Pop', 'Perc_Elec_Access', 
                     'Perc_Sec_Enroll_Gross', 'Water_type'],
                     log_dep=False,
                     log_all=True)
