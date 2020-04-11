import pandas as pd
import seaborn as sns
import scipy.stats as st
import numpy as np
import os
from datetime import datetime, timedelta, date
import requests
import urllib
import json

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

import argparse

from scrapper import *

def save_csv(countries=['Germany']):

	#datatypes = ['current', 'confirmed', 'recovered', 'deaths', 'new']
	datatypes = ['confirmed','deaths','new']

	folder = 'data/johnhopkins/'
	if not os.path.exists(folder):
		os.makedirs(folder)

	df = get_data()

	for country in countries:

		for datatype in datatypes:
			file_save = folder + country.replace('*','').lower() + '_' + datatype + '.csv'
			df[df['Country/Region']==country][['date',datatype]].to_csv(file_save, index=False)

def get_data():

	data_types = ['confirmed', 'deaths']

	df = download_johnhopkins('confirmed', 'field')
	df2 = download_johnhopkins('deaths', 'field')

	#TODO: check dfs are equal

	#Joins data
	df['deaths'] = df2['deaths']
	del df2


	#Calculates new infections
	country_list = df['Country/Region'].unique()

	#Calculates difference in infections
	df['new'] = df['confirmed'].diff()
	df.loc[df['date']==df['date'].min(), 'new'] = 0

	return df

def plot_countries(df,country_list, lag_countries=None, variable='confirmed', xmin='2020-02-21', ymin=1,ax=None,savefig=None, plot_order=None, plot_color = None, xlabel_days=False,**kwargs):

	#Set defaults
	if ax is None:
		fig = plt.figure(figsize=(4.5,3.5))

	if plot_order is None:
		plot_order = np.arange(len(country_list))
		
	if plot_color is None:
		#plot_color = [plt.get_cmap('viridis')(i) for i in np.arange(len(country_list))]
		plot_color =  [None for i in range(len(country_list))]

	if ax is None:
		ax = plt.gca()

	#Copies relevant data and aligns it
	df2 = df[df['Country/Region'].isin(country_list)]

	if lag_countries is not None and xlabel_days is False:
		lag_dict = dict(zip(country_list, lag_countries))
		df2['date_lag'] = df2['date'] - df2['Country/Region'].apply(lambda x:pd.to_timedelta(lag_dict[x], unit='d'))
		x_var = 'date_lag'

	elif lag_countries is not None and xlabel_days is True:
		lag_dict = dict(zip( country_list, lag_countries))
		df2['date_lag'] = df2['date'] - df2['Country/Region'].apply(lambda x:pd.to_timedelta(lag_dict[x], unit='d')) - datetime.fromisoformat(xmin)
		df2['date_lag_days'] = df2['date_lag'].apply(lambda x: x.days)

		x_var = 'date_lag_days'	
			
		#return df2
	else:
		x_lim_0 = pd.Timestamp(xmin)
		x_lim_1 = df2[x_var].max() + pd.to_timedelta(1, unit='d')
		x_var = 'date'

	for i in range(len(country_list)):
		x_data = df2[df2['Country/Region']==country_list[i]][x_var]
		y_data = df2[df2['Country/Region']==country_list[i]][variable]
		ax.plot(x_data,y_data, zorder = plot_order[i], color=plot_color[i], **kwargs)
	#ax = sns.lineplot(data=df2,hue='Country/Region', x=x_var,y=variable, ax=ax,**kwargs)
	
	#Beautifies plot
	ax.set_yscale('log')

	#Sets Y
	y_lim_0 = ymin
	y_lim_1 = np.power(10,np.ceil(np.log10(df2[variable].max())))
	ax.set_ylim([y_lim_0, y_lim_1])
	ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

	#Sets X
	if xlabel_days is False:
		x_lim_0 = pd.Timestamp(xmin)
		x_lim_1 = df2[x_var].max() + pd.to_timedelta(1, unit='d')
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

	else:
		x_lim_0 = 0
		x_lim_1 = df2[x_var].max() + 3

	ax.set_xlim([x_lim_0, x_lim_1])
	
	
	#ax.xaxis.set_major_locator(mdates.DayLocator())
	
	#ax.autoscale(enable=True,axis='y', tight=True)

	plt.tight_layout()

	if savefig is not None:
		plt.savefig(savefig, dpi=200)
		plt.close('all')

def plot_prediction(file, days_pred = 3, m_days=3, title=None, datatype = 'confirmed', labels = ['Confirmed cases','Forecast'], ax=None, x_min = None,color='r', savefig=None, **kwargs):

	#Parameters
	interval_cl = 0.95
	color_m = 'k'
	color_pred = sns.set_hls_values(color,l=0.2)
	plt.rcParams.update({'font.size': 8})

	#Default labels
	str_xlabel = 'Date'

	#Parses input
	if type(file) == str:
		df = pd.read_csv(file)
	elif type(file) == pd.core.frame.DataFrame:
		df = file
	else:
		TypeError('file is neither a .csv location or a pandas dataframe.')

	df['date'] = pd.to_datetime(df['date'])

	if x_min is None:
		if (df[datatype] > 0).any():
			x_min = df[df[datatype] > 0]['date'].min()
			date_max = df['date'].max()
		else:
			x_min = df['date'].min()
			date_max = df['date'].max()

	#df_pred = pd.DataFrame(columns=['date', 'prediction_min', 'prediction_max'])

	#Calculates observables
	df['m'] = df[datatype]/df[datatype].shift(1)
	df['m_median'] = df['m'].rolling(m_days).median()
	df['m_cl0'] = df['m'].rolling(m_days).apply(lambda x: st.t.interval(interval_cl, len(x)-1, loc=np.mean(x), scale=st.sem(x))[0])
	df['m_cl1'] = df['m'].rolling(m_days).apply(lambda x: st.t.interval(interval_cl, len(x)-1, loc=np.mean(x), scale=st.sem(x))[1])
	df['m_std'] = df['m'].rolling(m_days).std()
	df['increase_p'] = 100*(df['m']-1)

	df['prediction'] = df[datatype].shift(1)*df['m_median'].shift(1)

	#Uses last m, and calculades predictions on a new df

	m_last = df['m_median'].iloc[-1]
	data_last = df[datatype].iloc[-1]

	df_pred = pd.DataFrame(columns=['date', 'prediction'])
	date_min = df['date'].max() + pd.to_timedelta(1,unit='d')
	df_pred['date'] = pd.date_range(start=date_min,periods=days_pred)
	for i in range(days_pred):
		df_pred['prediction'][i] = m_last**(i+1)*data_last


	# for i in range(days):
	# 	df_pred['prediction_min'][i] = m_range[0]**(i+1)*df[datatype][df.index[-1]]
	# 	df_pred['prediction_max'][i] = m_range[1]**(i+1)*df[datatype][df.index[-1]]
	# df_pred['prediction'] = (df_pred['prediction_max'] + df_pred['prediction_min'])/2

	if ax is None:
		fig = plt.figure(figsize=(4.5,3.5))
		ax = plt.gca()



	#label_1 = 'Confirmed cases' + str(df['date'].iloc[-1])[:-9]
	# label_1 = 'Confirmed cases'
	# label_2 = 'Forecast'
	label_1 = 'Bestätigte Fälle'
	label_2 = 'Vorhersage'
	#label_2 = 'Prediction (previous {:d} days)'.format(m_days)
	#label_3 = 'Increase factor (previous {:d} days)'.format(m_days)

	ax.bar(data=df, x='date', height=datatype, color=color,**kwargs)
	#ax.plot(df['date'], df['prediction'], color=color_pred,lw=3,**kwargs)
	ax.bar(data=df_pred, x='date', height='prediction', color=color_pred,**kwargs)
	#ax2 = ax.twinx() 
	#ax2.plot(df['date'], df['m_median'], '--', color=color_m, lw=2)
	#ax2.plot(df['date'], df['increase_p'], '--', color=color_m, label='Z')
	#ax2.set_ylim([0,3])
	#ax2.tick_params(axis='y', labelcolor=color_m)

	#Beautifies plot
	ax.set_xlim([pd.Timestamp(x_min), df_pred['date'].max() + pd.to_timedelta(1,unit='d')])
	ax.set_xlim([pd.Timestamp(x_min), df['date'].max() + pd.to_timedelta(days_pred + 1,unit='d')])
	#ax.set_title(title)
	#ax.set_ylabel(ylabel)
	if title is not None:
		ax.set_title(title)
	#ax.set_xlabel(r'$\qquad\qquad\qquad\qquad$Date$\qquad$ (updated on ' + str(df['date'].iloc[-1])[:-9] + ')')
	#ax.set_xlabel(r'$\qquad\qquad\qquad\qquad$Datum$\qquad$ (updated on ' + str(df['date'].iloc[-1])[:-9] + ')')
	ax.set_xlabel(str_xlabel)
	#ax.set_xlabel(r"Date $\qquad$")
	#ax2.set_ylabel('Increase factor ')
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
	#fig.tight_layout()

	#custom_lines = [Line2D([0], [0], color=color, lw=4), Line2D([0], [0], color=color_pred, lw=4), Line2D([0], [0], linestyle='--', lw=2, color=color_m)]
	#ax2.legend(custom_lines, [label_1, label_2, label_3], framealpha=1, facecolor='white', loc='upper left')
	custom_lines = [Line2D([0], [0], color=color, lw=4), Line2D([0], [0], color=color_pred, lw=4)]
	ax.legend(custom_lines, [labels[0], labels[1]], framealpha=1, facecolor='white', loc='upper left')
	plt.tight_layout()

	#str_ann = 'Updated on ' + str(df['date'].iloc[-1])[:-9] + '\nData source: https://systems.jhu.edu/research/public-health/ncov/'
	#str_ann = 'Data source: https://systems.jhu.edu/research/public-health/ncov/'
	#ax.annotate(str_ann, xy=(1,-0.02), xycoords=('axes fraction','figure fraction'), xytext=(0,6), textcoords='offset points', ha='right')

	if savefig is not None:
		plt.savefig(savefig, dpi=200)
		plt.close('all')

def plot_prediction_m(file, days_pred = 3, m_days=3, title=None, datatype = 'confirmed', labels = ['Confirmed cases','Forecast', 'Average increase (%)'], ax=None, x_min = None,color='r', savefig=None, **kwargs):

	#Parameters
	interval_cl = 0.95
	color_m = 'b'
	color_pred = sns.set_hls_values(color,l=0.2)
	plt.rcParams.update({'font.size': 8})

	#Default labels
	str_xlabel = 'Date'

	#Parses input
	if type(file) == str:
		df = pd.read_csv(file)
	elif type(file) == pd.core.frame.DataFrame:
		df = file
	else:
		TypeError('file is neither a .csv location or a pandas dataframe.')

	df['date'] = pd.to_datetime(df['date'])

	if x_min is None:
		if (df[datatype] > 0).any():
			x_min = df[df[datatype] > 0]['date'].min()
			date_max = df['date'].max()
		else:
			x_min = df['date'].min()
			date_max = df['date'].max()

	#df_pred = pd.DataFrame(columns=['date', 'prediction_min', 'prediction_max'])

	#Calculates observables
	df['m'] = df[datatype]/df[datatype].shift(1)
	df['m_median'] = df['m'].rolling(m_days).median()
	df['m_median_p'] = 100*(df['m_median'] - 1)
	df['m_cl0'] = df['m'].rolling(m_days).apply(lambda x: st.t.interval(interval_cl, len(x)-1, loc=np.mean(x), scale=st.sem(x))[0])
	df['m_cl1'] = df['m'].rolling(m_days).apply(lambda x: st.t.interval(interval_cl, len(x)-1, loc=np.mean(x), scale=st.sem(x))[1])
	df['m_std'] = df['m'].rolling(m_days).std()
	df['increase_p'] = 100*(df['m']-1)

	df['prediction'] = df[datatype].shift(1)*df['m_median'].shift(1)

	#Uses last m, and calculades predictions on a new df
	m_last = df['m_median'].iloc[-1]
	data_last = df[datatype].iloc[-1]
	date_last = df['date'].iloc[-1]
	df_pred = df[['date', 'prediction']]

	df_pred_temp = pd.DataFrame(columns=['date', 'prediction'])
	df_pred_temp['date'] = [date_last + pd.to_timedelta(i+1,unit='d') for i in range(days_pred)]
	df_pred_temp['prediction'] = [data_last*m_last**(i+1) for i in range(days_pred)]
	df_pred = df_pred.append(df_pred_temp)


	# #df_pred = pd.DataFrame(columns=['date', 'prediction'])
	# date_min = df['date'].max() + pd.to_timedelta(1,unit='d')
	# df_pred['date'] = pd.date_range(start=date_min,periods=days_pred)
	# for i in range(days_pred):
	# 	df_pred['prediction'][i] = m_last**(i+1)*data_last


	# for i in range(days):
	# 	df_pred['prediction_min'][i] = m_range[0]**(i+1)*df[datatype][df.index[-1]]
	# 	df_pred['prediction_max'][i] = m_range[1]**(i+1)*df[datatype][df.index[-1]]
	# df_pred['prediction'] = (df_pred['prediction_max'] + df_pred['prediction_min'])/2

	if ax is None:
		fig = plt.figure(figsize=(4.5,3.5))
		ax = plt.gca()



	#label_1 = 'Confirmed cases' + str(df['date'].iloc[-1])[:-9]
	# label_1 = 'Confirmed cases'
	# label_2 = 'Forecast'
	label_1 = 'Bestätigte Fälle'
	label_2 = 'Vorhersage'
	#label_2 = 'Prediction (previous {:d} days)'.format(m_days)
	#label_3 = 'Increase factor (previous {:d} days)'.format(m_days)

	ax.bar(data=df, x='date', height=datatype, color=color,**kwargs)
	ax.plot(df_pred['date'], df_pred['prediction'],'D', fillstyle='full',color=color_pred,lw=3,**kwargs)
	#ax.bar(data=df_pred, x='date', height='prediction', color=color_pred,**kwargs)

	ax2 = ax.twinx() 
	ax2.plot(df['date'], df['m_median_p'], '--', color=color_m, lw=2)
	#ax2.plot(df['date'], df['increase_p'], '--', color=color_m, label='Z')
	#ax2.set_ylim([0,3])
	ax2.tick_params(axis='y', labelcolor=color_m)

	#Beautifies plot
	ax.set_xlim([pd.Timestamp(x_min), df_pred['date'].max() + pd.to_timedelta(1,unit='d')])
	ax.set_xlim([pd.Timestamp(x_min), df['date'].max() + pd.to_timedelta(days_pred + 1,unit='d')])
	#ax.set_title(title)
	#ax.set_ylabel(ylabel)
	ax.set_title(title)
	#ax.set_xlabel(r'$\qquad\qquad\qquad\qquad$Date$\qquad$ (updated on ' + str(df['date'].iloc[-1])[:-9] + ')')
	#ax.set_xlabel(r'$\qquad\qquad\qquad\qquad$Datum$\qquad$ (updated on ' + str(df['date'].iloc[-1])[:-9] + ')')
	ax.set_xlabel(str_xlabel)
	#ax.set_xlabel(r"Date $\qquad$")
	#ax2.set_ylabel('Increase (%)')
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
	#fig.tight_layout()
	#ax.grid(axis='y', zorder=0)

	#custom_lines = [Line2D([0], [0], color=color, lw=4), Line2D([0], [0], color=color_pred, lw=4), Line2D([0], [0], linestyle='--', lw=2, color=color_m)]
	#ax2.legend(custom_lines, [label_1, label_2, label_3], framealpha=1, facecolor='white', loc='upper left')
	custom_lines = [Line2D([0], [0], color=color, lw=4), Line2D([0], [0], color=color_pred, lw=0, marker='D'), Line2D([0], [0], linestyle='--', lw=2, color=color_m)]
	ax.legend(custom_lines, [labels[0], labels[1], labels[2]], framealpha=0, facecolor='white', loc='upper left')
	plt.tight_layout()

	#str_ann = 'Updated on ' + str(df['date'].iloc[-1])[:-9] + '\nData source: https://systems.jhu.edu/research/public-health/ncov/'
	#str_ann = 'Data source: https://systems.jhu.edu/research/public-health/ncov/'
	#ax.annotate(str_ann, xy=(1,-0.02), xycoords=('axes fraction','figure fraction'), xytext=(0,6), textcoords='offset points', ha='right')

	if savefig is not None:
		plt.savefig(savefig, dpi=200)
		plt.close('all')

def update_countries(days_pred = 3, m_days = 3):

	#Sets backend
	old_backend = matplotlib.get_backend()
	matplotlib.use('Agg')

	#datatypes = ['current', 'confirmed', 'recovered', 'deaths', 'new']
	datatypes = ['confirmed', 'deaths', 'new']

	# ylabel = {'current':'Current cases',
	#  'confirmed':'Confirmed cases',
	#  'recovered':'Recovered cases',
	#  'deaths': 'Number of deaths', 
	#  'new': 'New cases'}

	ylabel = {'confirmed':'Confirmed cases',
	 'deaths': 'Number of deaths', 
	 'new': 'New cases'}	


	#Gets all unique countries
	df = get_data()
	country_list = df['Country/Region'].unique().tolist()
	del df

	#country_list = [s.replace('*','') for s in country_list]

	#Sets up specific x_min for certain countries
	x_min = dict.fromkeys(country_list)
	x_min['Germany'] = '2020-03-04'
	x_min['Italy'] = '2020-02-23'
	x_min['France'] = '2020-02-28'
	x_min['Canada'] = '2020-02-22'
	x_min['Brazil'] = '2020-03-07'
	x_min['India'] = '2020-03-04'
	x_min['Korea, South'] = '2020-02-21'
	x_min['Spain'] = '2020-03-02'
	x_min['Sweden'] = '2020-03-01'
	x_min['US'] = '2020-03-01'

	labels = ['Confirmed','Forecast', 'Average increase (%)']

	#Saves countries to unique csv files
	save_csv(country_list)

	#Plots folders
	folder = 'plots/johnhopkins/'
	if not os.path.exists(folder):
		os.makedirs(folder)

	for country in country_list:
		for datatype in datatypes:
			file_load = 'data/johnhopkins/' + country.replace('*','').lower() + '_' + datatype + '.csv'
			plot_save = folder + country.replace('*','').lower() + '_' + datatype  + '.png'
			plot_prediction_m(file_load, days_pred, m_days, country, datatype, savefig=plot_save, x_min=x_min[country], labels=labels)


	#Restores backend
	matplotlib.use(old_backend)

def update_website(lang='all', savefig=True, update_csv=False):

	#folder location
	folder_current = 'plots/'
	folder_daily = 'plots/website_daily/'

	#Updates date
	str_now = datetime.now().strftime("%Y-%m-%d")
	f = open('.last_updated','w')
	f.write(str_now)
	f.close()

	#Rare actually useful usage of recursion
	if lang == 'all':
		update_website('en', update_csv = True)
		update_website('de', update_csv = False)
		return 

	#Plots local short-term forecast
	if lang == 'en':
		ylabel_de = 'Confirmed cases in Germany'
		ylabel_ls = 'Confirmed cases in Lower Saxony'
		xlabel = 'Date'
		str_save = 'germany_local_en'
		str_save_m = 'germany_local_pred_en'
		labels = ['Confirmed cases', 'Forecast']
		labels_m = ['Confirmed cases', 'Forecast', 'Average increase (%)']
		str_ann = 'Updated: ' + datetime.now().strftime("%d/%m/%Y")

	elif lang == 'de':
		ylabel_de = 'Gesamtzahl bestätigter Fälle in Deutschland'
		ylabel_ls = 'Gesamtzahl bestätigter Fälle in Niedersachsen'
		xlabel = 'Datum'
		labels = ['Bestätigte Fälle', 'Vorhersage']
		labels_m = ['Bestätigte Fälle', 'Vorhersage', 'Durchschnittlicher Anstieg (%)']
		str_save = 'germany_local_de'
		str_save_m = 'germany_local_pred_de'
		str_ann = 'Aktualisiert: ' + datetime.now().strftime("%d/%m/%Y")

	fig = plt.figure(figsize=(9,3.5))
	gs = fig.add_gridspec(1,2)
	ax_germany = fig.add_subplot(gs[0])
	ax_lowersaxony = fig.add_subplot(gs[1])

	#Updates RKI data files
	if update_csv:
		df_rki = download_rki_bundesland_current()
		cases_de_now = df_rki.Fallzahl.sum()
		cases_ns_now = df_rki[df_rki.LAN_ew_GEN == 'Niedersachsen']['Fallzahl'].values[0]

		str_now_de = '{:d}/{:d}/{:d},{:d}\n'.format(datetime.now().month,datetime.now().day,datetime.now().year,int(cases_de_now))
		str_now_ns = '{:d}/{:d}/{:d},{:d}\n'.format(datetime.now().month,datetime.now().day,datetime.now().year,int(cases_ns_now))

		with open('data/germany_confirmed.csv', "a") as file:
		    file.write(str_now_de)
		
		with open('data/lowersaxony_confirmed.csv', "a") as file:
		    file.write(str_now_ns)	  

	# new_entry_de = {'date':str_now, 'confirmed':cases_de_now}
	# df_de = pd.read_csv('data/germany_confirmed.csv')
	# df_de_new = df_de.append(pd.DataFrame([new_entry_de]), ignore_index=True)
	# df_de_new.to_csv('data/germany_confirmed.csv', index=False)
	
	# new_entry_ns = {'date':str_now, 'confirmed':cases_ns_now}
	# df_ns = pd.read_csv('data/lowersaxony_confirmed.csv')
	# df_ns_new = df_ns.append(pd.DataFrame([new_entry_ns]), ignore_index=True)
	# df_ns_new.to_csv('data/lowersaxony_confirmed.csv', index=False)

	#Plots data for Germand and Lower Saxony
	plot_prediction('data/germany_confirmed.csv', datatype='confirmed', x_min='2020-03-04', labels=labels, ax=ax_germany, title=ylabel_de)
	plot_prediction('data/lowersaxony_confirmed.csv', datatype='confirmed', x_min='2020-03-04', labels=labels, ax=ax_lowersaxony, title = ylabel_ls)

	ax_germany.set_xlabel(xlabel)
	ax_lowersaxony.set_xlabel(xlabel)

	ax_lowersaxony.annotate(str_ann, xy=(1,-0.01), xycoords=('axes fraction','figure fraction'), xytext=(0,6), textcoords='offset points', ha='right')
	ax_germany.annotate('A', xy=(-0.12,0.9), xycoords=('axes fraction','figure fraction'), xytext=(0,7), textcoords='offset points', ha='left', weight='bold')
	ax_lowersaxony.annotate('B', xy=(-0.10,0.9), xycoords=('axes fraction','figure fraction'), xytext=(0,7), textcoords='offset points', ha='left', weight='bold')

	plt.tight_layout()

	#Saves both current and dated versions
	if savefig:
		plt.savefig(folder_current + str_save + '.png', dpi=200)
		plt.savefig(folder_daily + datetime.now().strftime("%Y_%m_%d_") + str_save + '.png', dpi = 200)
		plt.close('all')

	#Plots prediction with m
	fig = plt.figure(figsize=(5,7))
	gs = fig.add_gridspec(2,1)
	ax_germany = fig.add_subplot(gs[0,0])
	ax_lowersaxony = fig.add_subplot(gs[1,0])

	plot_prediction_m('data/germany_confirmed.csv', datatype='confirmed', x_min='2020-03-04', labels=labels_m, ax=ax_germany, title=ylabel_de)
	plot_prediction_m('data/lowersaxony_confirmed.csv', datatype='confirmed', x_min='2020-03-04', labels=labels_m, ax=ax_lowersaxony, title = ylabel_ls)

	ax_germany.set_xlabel(xlabel)
	ax_lowersaxony.set_xlabel(xlabel)

	ax_lowersaxony.annotate(str_ann, xy=(1,-0.01), xycoords=('axes fraction','figure fraction'), xytext=(0,6), textcoords='offset points', ha='right')
	#ax_germany.annotate('A', xy=(-0.12,0.9), xycoords=('axes fraction','figure fraction'), xytext=(0,7), textcoords='offset points', ha='left', weight='bold')
	#ax_lowersaxony.annotate('B', xy=(-0.10,0.9), xycoords=('axes fraction','figure fraction'), xytext=(0,7), textcoords='offset points', ha='left', weight='bold')
	
	plt.tight_layout()

	if savefig:
		plt.savefig(folder_current + str_save_m, dpi=200)
		plt.close('all')

	#Plots comparison between countries_list
	countries_list = ['Germany', 'Italy', 'Korea, South']
	color_palette = ['r','b','g']

	fig = plt.figure(figsize=(9,3.5))
	gs = fig.add_gridspec(1,2)
	ax_cases = fig.add_subplot(gs[0])
	ax_deaths = fig.add_subplot(gs[1])
	df = get_data()
	#plot_countries(df,countries_list,variable='current', lag_countries=[0,-7.5,-10], xmin='2020-03-01', ylim=[100,1e5], ax=ax_cases,**{'linewidth':4, 'palette': color_palette})
	#plot_countries(df,countries_list,variable='deaths', lag_countries=[0,-17,-18], xmin='2020-03-09', ymin=1,ax=ax_deaths,**{'linewidth':4, 'palette': color_palette})
	plot_countries(df,countries_list,variable='confirmed', lag_countries=[-0.5,-7.5,-10], xmin='2020-03-01', ymin = 100,  ax=ax_cases, plot_order=[100,10,1], plot_color=color_palette, xlabel_days=True, **{'linewidth':4})
	plot_countries(df,countries_list,variable='deaths', lag_countries=[0,-17,-18], xmin='2020-03-09', ymin = 1, ax=ax_deaths, plot_order=[100,10,1], plot_color=color_palette, xlabel_days=True, **{'linewidth':4})

	if lang == 'de':

		ax_cases.legend(['Deutschland','Italien', 'Südkorea'], loc='upper left')
		ax_deaths.legend(['Deutschland','Italien', 'Südkorea'], loc='upper left')
		ax_cases.set_xlabel('Tage seit dem 100. Krankheitsfall')
		ax_cases.set_title('Coronaerkrankungen')
		ax_deaths.set_xlabel('Tage seit dem 1. Todesfall')
		ax_deaths.set_title('Todesfälle')

		str_save = 'evolution_de'

	elif lang == 'en':
		ax_cases.legend(['Germany','Italy', 'South Korea'], loc='upper left')
		ax_deaths.legend(['Germany','Italy', 'South Korea'], loc='upper left')
		ax_cases.set_xlabel('Days since 100th case')
		ax_cases.set_title('Confirmed cases')
		ax_deaths.set_xlabel('Days since 1st death')
		ax_deaths.set_title('Deaths')

		str_save = 'evolution_en'

	#Aligns to Germany
	#df_de = df[df['Country/Region']=='Germany']
	#date_range_cases = df_de['date'].max() - datetime.fromisoformat('2020-03-01')
	#date_range_deaths = df_de['date'].max() - datetime.fromisoformat('2020-03-09')
	ax_cases.set_ylabel('')
	ax_deaths.set_ylabel('')

	ax_deaths.annotate(str_ann, xy=(1,-0.02), xycoords=('axes fraction','figure fraction'), xytext=(0,6), textcoords='offset points', ha='right')
	ax_cases.annotate('A', xy=(-0.19,0.9), xycoords=('axes fraction','figure fraction'), xytext=(0,7), textcoords='offset points', ha='left', weight='bold')
	ax_deaths.annotate('B', xy=(-0.17,0.9), xycoords=('axes fraction','figure fraction'), xytext=(0,7), textcoords='offset points', ha='left', weight='bold')

	plt.tight_layout()

	if savefig:
		plt.savefig(folder_current + str_save + '.png', dpi=200)
		plt.savefig(folder_daily + datetime.now().strftime("%Y_%m_%d_") + str_save + '.png', dpi = 200)
		plt.close('all')

def age_distribution(location):

	if location is 'Germany':
		#From PopulationPyramid.net, data from 2019
		age_distribution = {'A15-A34':0.227686061, 'A35-A59':0.350849027, 'A60-A79':0.214771991, 'A80+':0.06869733, 'A05-A14':0.090251541, 'A00-A04': 0.04774405}

	elif location is 'Niedersachsen':
		pass

	elif location is 'Goettingen':
		pass

	else:
		ValueError('Invalid location')

	return age_distribution

def plot_age_distribution_rki(df_age, variable = 'AnzahlFall', delta_x = 30, ax=None, age_groups = ['A00-A04','A05-A14','A15-A34','A35-A59', 'A60-A79', 'A80+'], location='Germany'):

	#Colors
	color_ages = {'A00-A04': '#1f77b4ff',
	'A05-A14': '#ff7f0eff' ,'A15-A34': '#2ca02cff',
	'A35-A59': '#d62728ff', 'A60-A79': '#9467bdff' , 'A80+':'#8c564bff'}

	#Parses input
	if variable not in ['AnzahlFall', 'AnzahlTodesfall']:
		ValueError('Invalid variable. Valid options: "AnzahlFall", "AnzahlTodesfall"')

	#Empty dict that stores colors from data plots and reuses for '--'' plots
	#age_colors = dict.fromkeys(all_ages)

	#Plots results
	if ax is None:
		fig = plt.figure()
		ax = plt.gca()

	for age in age_groups:
		#p = ax.plot(df_age.index, df_age[age + '_total_p'], label=age, color=age_colors[age], linewidth=3)
		p = ax.plot(df_age.index, df_age[age + '_total_p'], label=age, color=color_ages[age], linewidth=3)
		#age_colors[age] = p[-1].get_color()

	#Plots comparison to population distribution
	if variable == 'AnzahlFall' and location == 'Germany':

		#Gets age distribution from static data
		age_dist = age_distribution(location)

		for age in age_groups:
			#ax.plot(df_age.index, 100*np.ones(len(df_age.index))*age_dist[age], '--', color=age_colors[age],linewidth=2)
			ax.plot(df_age.index, 100*np.ones(len(df_age.index))*age_dist[age], '--', color=color_ages[age],linewidth=2)

	#Beautifies plots
	ax.autoscale()
	ax.set_xlabel('date')
	ax.set_ylabel('%')
	ax.set_title(variable)
	#ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
	x_max = df_age.index.max()
	x_min = x_max - pd.to_timedelta(delta_x,unit='d')
	ax.set_xlim([x_min, x_max])

def plot_age(file=None, delta_x=21, location='Germany', landkreis = None, bundesland = None, savefig = None):

	#Downloads entire dataset from RKI
	if file is None:
		file = 'data/rki_latest.csv'
		download_rki(file)

	#Creates figure
	fig = plt.figure(figsize=(8,8))
	gs = fig.add_gridspec(2,2)
	ax_total = fig.add_subplot(gs[0,0])
	ax_cases = fig.add_subplot(gs[0,1])
	ax_deaths = fig.add_subplot(gs[1,1])
	ax_cases_old = fig.add_subplot(gs[1,0])

	#Loads dfs
	df_cases = load_rki_age(file, variable='AnzahlFall', landkreis = landkreis, bundesland = bundesland)
	df_deaths = load_rki_age(file, variable='AnzahlTodesfall', landkreis = landkreis, bundesland = bundesland)

	#Plots total cases and deaths
	str_label_cases = 'total cases: {:d}'.format(df_cases.total[-1]) 
	str_label_deaths = 'deaths: {:d}'.format(df_deaths.total[-1]) 
	ax_total.plot(df_cases.index, df_cases.total, label=str_label_cases, linewidth=3, color='k')
	ax_total.plot(df_deaths.index, df_deaths.total, 'D', label=str_label_deaths,linewidth=3, color='k')

	#Plots cases and deaths age distribution
	plot_age_distribution_rki(df_cases, variable = 'AnzahlFall', delta_x=delta_x, ax=ax_cases, location=location)
	plot_age_distribution_rki(df_deaths, variable = 'AnzahlTodesfall', delta_x=delta_x, ax=ax_deaths, location=location)
	plot_age_distribution_rki(df_cases, variable = 'AnzahlFall', delta_x=delta_x, ax=ax_cases_old, location=location, age_groups = ['A60-A79', 'A80+'])

	#Beautifies plots
	# ax_cases.autoscale()
	# ax_deaths.autoscale()
	# ax_cases_old.autoscale()
	ax_total.set_yscale('log')
	ax_total.set_ylim([1,ax_total.get_ylim()[1]])
	#ax_cases.set_ylim([0,80])
	ax_cases_old.set_ylim([0,ax_cases_old.get_ylim()[1]])
	#ax_deaths.set_ylim([0,100])
	ax_cases.legend(loc='upper center',ncol=2, fancybox=False, shadow=False, framealpha=0.5)


	#Sets x_range to range with cases
	x_max = df_cases.index.max()
	x_min_cases = df_cases[df_cases.total > 0].index.min()
	x_min_delta = x_max - pd.to_timedelta(delta_x,unit='d')
	x_min = max(x_min_cases, x_min_delta)
	ax_total.set_xlim([x_min, x_max])
	ax_cases.set_xlim([x_min, x_max])
	ax_cases_old.set_xlim([x_min, x_max])
	ax_deaths.set_xlim([x_min, x_max])

	#Rotate xticks
	rot_angle = 45
	ax_total.set_xticklabels(ax_total.get_xticklabels(), rotation=rot_angle)
	ax_cases_old.set_xticklabels(ax_cases_old.get_xticklabels(), rotation=rot_angle)
	ax_cases.set_xticklabels(ax_cases.get_xticklabels(), rotation=rot_angle)
	ax_deaths.set_xticklabels(ax_deaths.get_xticklabels(), rotation=rot_angle)

	ax_total.legend(loc='upper left', framealpha=0)
	ax_total.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
	ax_cases.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
	ax_cases_old.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
	ax_deaths.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

	ax_total.set_title('Total')
	ax_cases.set_title('% of total cases in age group')
	ax_deaths.set_title('% of total deaths in age group')
	ax_cases_old.set_title('% of total cases for A60+')

	ax_cases.set_xlabel('')
	ax_deaths.set_xlabel('')
	ax_cases_old.set_xlabel('')
	ax_cases.set_ylabel('')
	ax_deaths.set_ylabel('')
	ax_cases_old.set_ylabel('')


	plt.suptitle(location, y=1)
	str_ann = 'updated: ' + datetime.now().strftime("%d/%m/%Y") + '\n data source: Robert Koch Institute'
	ax_deaths.annotate(str_ann, xy=(1,0.01), xycoords=('axes fraction','figure fraction'), xytext=(0,6), textcoords='offset points', ha='right')

	plt.tight_layout()

	if savefig is not None:
		plt.savefig(savefig, dpi=200)
		plt.close('all')

def update_age(file=None, delta_x=21, sleep=0):

	if file is None:
		file = 'data/rki_latest.csv'
		download_rki(file,sleep=0)

	#Creates the local plots
	str_germany = 'plots/germany/age_germany.png'
	str_lowersaxony = 'plots/germany/age_lowersaxony.png'
	str_goettingen = 'plots/germany/age_goettingen.png'

	plot_age(file, delta_x, location='Germany', landkreis = None, bundesland = None, savefig = str_germany)
	plot_age(file, delta_x, location='Lower Saxony', landkreis = None, bundesland = 'Niedersachsen', savefig = str_lowersaxony)
	plot_age(file, delta_x, location='LK Göttingen', landkreis = 'LK Göttingen', bundesland = None, savefig = str_goettingen)
	
	#Bundesland plots
	states = ['Rheinland-Pfalz', 'Nordrhein-Westfalen', 'Hessen', 'Sachsen','Schleswig-Holstein', 'Baden-Württemberg', 'Mecklenburg-Vorpommern', 'Bayern', 'Sachsen-Anhalt','Niedersachsen', 'Berlin', 'Thüringen', 'Brandenburg', 'Hamburg', 'Bremen', 'Saarland']

	for bundesland in states:
		str_save = 'plots/germany/bundesland/' + bundesland.replace("ü", "ue").lower() + '.png'
		plot_age(file, delta_x, location=bundesland, landkreis = None, bundesland = bundesland, savefig = str_save)

def plot_comparison_de():

	df_jhu = download_johnhopkins()
	df_rki = download_rki()

	df_rki_all['sum_cases'] = df_rki['AnzahlFall']  + df_rki['NeuerFall']

	df_rki = pd.DataFrame()
	df_rki['date'] = pd.to_datetime(df_rki_all['date'], format='%d-%m-%Y')
	df_rki['new_sum'] = df_rki.groupby('date')['sum_cases'].sum()
	df_rki['new'] = df_rki.groupby('date')['AnzahlFall'].sum()

if __name__== "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--plots",type=str, nargs='?', const=1, default='website')
	args = parser.parse_args()
	run_type = args.plots
	
	if run_type == 'website':
		update_website('all')
	elif run_type == 'countries':
		update_countries(3, 3)
	elif run_type == 'age':
		update_age(delta_x=21,sleep=1)
	else:
		ValueError('Invalid --plots.')

