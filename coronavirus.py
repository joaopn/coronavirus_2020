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

def download_johnhopkins_old(case_type = 'confirmed', df_type='field'):

	#Downloads data
	if case_type == 'confirmed':
		url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
	elif case_type == 'recovered':
		url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv'
	elif case_type == 'deaths':
		url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv'
	else:
		ValueError('Invalid case_type')

	df = pd.read_csv(url, sep=',')

	#Reshapes df
	if df_type == 'column':

		df = df.drop(columns=['Province/State', 'Lat','Long']).groupby('Country/Region').sum().T
		df.reset_index(inplace=True)
		df.rename(columns={'index':'date',"Country/Region":'country'}, inplace=True)
		df['date'] = pd.to_datetime(df['date'])

	elif df_type == 'field':

		df = df.drop(columns=['Province/State', 'Lat','Long']).groupby('Country/Region').sum().T
		df.reset_index(inplace=True)
		df.rename(columns={'index':'date',"Country/Region":'country'}, inplace=True)
		df['date'] = pd.to_datetime(df['date'])
		df = pd.melt(df, id_vars = 'date').rename(columns={'value':case_type})

	else:
		ValueError('Invalid df_type')

	return df

def save_csv(countries=['Germany']):

	datatypes = ['current', 'confirmed', 'recovered', 'deaths', 'new']

	folder = 'data/johnhopkins/'
	if not os.path.exists(folder):
		os.makedirs(folder)

	df = get_data()

	for country in countries:

		for datatype in datatypes:
			file_save = folder + country.replace('*','').lower() + '_' + datatype + '.csv'
			df[df['Country/Region']==country][['date',datatype]].to_csv(file_save, index=False)

def download_johnhopkins(case_type = 'confirmed', df_type='field'):

	#Downloads data
	if case_type == 'confirmed':
		url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
	elif case_type == 'deaths':
		url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
	else:
		ValueError('Invalid case_type')

	df = pd.read_csv(url, sep=',')

	#Reshapes df
	if df_type == 'column':

		df = df.drop(columns=['Province/State', 'Lat','Long']).groupby('Country/Region').sum().T
		df.reset_index(inplace=True)
		df.rename(columns={'index':'date',"Country/Region":'country'}, inplace=True)
		df['date'] = pd.to_datetime(df['date'])

	elif df_type == 'field':

		df = df.drop(columns=['Province/State', 'Lat','Long']).groupby('Country/Region').sum().T
		df.reset_index(inplace=True)
		df.rename(columns={'index':'date',"Country/Region":'country'}, inplace=True)
		df['date'] = pd.to_datetime(df['date'])
		df = pd.melt(df, id_vars = 'date').rename(columns={'value':case_type})

	else:
		ValueError('Invalid df_type')

	return df

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

def plot_countries(df,country_list, lag_countries=None, variable='confirmed', xmin='2020-02-21', ymin=1,ax=None,savefig=None, plot_order=None, plot_color = None, **kwargs):

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

	if lag_countries is not None:
		lag_dict = dict(zip(country_list, lag_countries))
		df2['date_lag'] = df2['date'] - df2['Country/Region'].apply(lambda x:pd.to_timedelta(lag_dict[x], unit='d'))
		x_var = 'date_lag'
	else:
		x_var = 'date'

	for i in range(len(country_list)):
		x_data = df2[df2['Country/Region']==country_list[i]][x_var]
		y_data = df2[df2['Country/Region']==country_list[i]][variable]
		ax.plot(x_data,y_data, zorder = plot_order[i], color=plot_color[i], **kwargs)
	#ax = sns.lineplot(data=df2,hue='Country/Region', x=x_var,y=variable, ax=ax,**kwargs)
	
	#Beautifies plot
	ax.set_yscale('log')

	#Sets limits
	x_lim_0 = pd.Timestamp(xmin)
	x_lim_1 = df2[x_var].max() + pd.to_timedelta(1, unit='d')
	y_lim_0 = ymin
	y_lim_1 = np.power(10,np.ceil(np.log10(df2[variable].max())))

	ax.set_xlim([x_lim_0, x_lim_1])
	ax.set_ylim([y_lim_0, y_lim_1])
	ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
	#ax.xaxis.set_major_locator(mdates.DayLocator())
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
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
	ax.plot(df_pred['date'], df_pred['prediction'], color=color_pred,lw=3,**kwargs)
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
	custom_lines = [Line2D([0], [0], color=color, lw=4), Line2D([0], [0], color=color_pred, lw=4), Line2D([0], [0], linestyle='--', lw=2, color=color_m)]
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

def update_local(lang='all', savefig=True):

	#folder location
	folder_current = 'plots/'
	folder_daily = 'plots/website_daily/'


	#Rare actually useful usage of recursion
	if lang == 'all':
		update_local('en')
		update_local('de')
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

	#Updates Germany data and plots things
	save_csv(countries=['Germany'])
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
	fig = plt.figure(figsize=(9,3.5))
	gs = fig.add_gridspec(1,2)
	ax_germany = fig.add_subplot(gs[0])
	ax_lowersaxony = fig.add_subplot(gs[1])

	plot_prediction_m('data/germany_confirmed.csv', datatype='confirmed', x_min='2020-03-04', labels=labels_m, ax=ax_germany, title=ylabel_de)
	plot_prediction_m('data/lowersaxony_confirmed.csv', datatype='confirmed', x_min='2020-03-04', labels=labels_m, ax=ax_lowersaxony, title = ylabel_ls)

	ax_germany.set_xlabel(xlabel)
	ax_lowersaxony.set_xlabel(xlabel)

	ax_lowersaxony.annotate(str_ann, xy=(1,-0.01), xycoords=('axes fraction','figure fraction'), xytext=(0,6), textcoords='offset points', ha='right')
	ax_germany.annotate('A', xy=(-0.12,0.9), xycoords=('axes fraction','figure fraction'), xytext=(0,7), textcoords='offset points', ha='left', weight='bold')
	ax_lowersaxony.annotate('B', xy=(-0.10,0.9), xycoords=('axes fraction','figure fraction'), xytext=(0,7), textcoords='offset points', ha='left', weight='bold')
	
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
	plot_countries(df,countries_list,variable='confirmed', lag_countries=[-0.5,-7.5,-10], xmin='2020-03-01', ymin = 100,  ax=ax_cases, plot_order=[100,10,1], plot_color=color_palette, **{'linewidth':4})
	plot_countries(df,countries_list,variable='deaths', lag_countries=[0,-17,-18], xmin='2020-03-09', ymin = 1, ax=ax_deaths, plot_order=[100,10,1], plot_color=color_palette,  **{'linewidth':4})

	if lang == 'de':

		ax_cases.legend(['Deutschland','Italien', 'Südkorea'])
		ax_deaths.legend(['Deutschland','Italien', 'Südkorea'])
		ax_cases.set_xlabel('Tage seit dem 100. Krankheitsfall')
		ax_cases.set_title('Coronaerkrankungen')
		ax_deaths.set_xlabel('Tage seit dem 1. Todesfall')
		ax_deaths.set_title('Todesfälle')

		str_save = 'evolution_de'

	elif lang == 'en':
		ax_cases.legend(['Germany','Italy', 'South Korea'])
		ax_deaths.legend(['Germany','Italy', 'South Korea'])
		ax_cases.set_xlabel('Days since 100th case')
		ax_cases.set_title('Confirmed cases')
		ax_deaths.set_xlabel('Days since 1st death')
		ax_deaths.set_title('Deaths')

		str_save = 'evolution_en'

	ax_cases.set_xticklabels(np.arange(0,30,4))
	ax_deaths.set_xticklabels(np.arange(0,30,4))
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

def download_state(file='data/deutschland_bundeslaendern.csv'):

	#Queries arcgis
	url = ''

	current_data_list = requests.get(url).json()['features']

def download_landkreis(file='data/deutschland_landkreis.csv'):

	#Loads data into df
	df = pd.read_csv(file)

	#Queries arcgis
	url = 'https://services7.arcgis.com/mOBPykOjAyBO2ZKk/ArcGIS/rest/services/RKI_Landkreisdaten/FeatureServer/0/query?where=0%3D0&objectIds=&time=&geometry=&geometryType=esriGeometryEnvelope&inSR=&spatialRel=esriSpatialRelIntersects&resultType=none&distance=0.0&units=esriSRUnit_Meter&returnGeodetic=false&outFields=GEN%2C+cases%2C+deaths%2C+EWZ%2C+county&returnGeometry=false&returnCentroid=false&featureEncoding=esriDefault&multipatchOption=xyFootprint&maxAllowableOffset=&geometryPrecision=&outSR=&datumTransformation=&applyVCSProjection=false&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnExtentOnly=false&returnQueryGeometry=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&returnZ=false&returnM=false&returnExceededLimitFeatures=true&quantizationParameters=&sqlFormat=none&f=pjson&token='

	current_data_list = requests.get(url).json()['features']

	#Stores current data on a df
	pass
	
	#Checks if last date on df is equal to current. If yes, overwrites entries. Adds new entries to df otherwise.
	pass

	#Saves df to file

def download_rki(single_date):
	# url = 'https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/RKI_COVID19/FeatureServer/0//query?where=Meldedatum%3D%272020-03-21'
	# + '%27&objectIds=&time=&resultType=standard&outFields=*&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token='

	list_data = []

	#Creates data list
	# def daterange(start_date, end_date):
	# 	for n in range(int ((end_date - start_date).days)):
	# 		yield start_date + timedelta(n)
	# start_date = date(2020, 1, 1)
	# end_date = date.today()
	# list_data = []
	# for single_date in daterange(start_date, end_date):
	# 	url_str = 'https://services7.arcgis.com/mOBPykOjAyBO2ZKk/ArcGIS/rest/services/RKI_COVID19/FeatureServer/0/query?where=Meldedatum%3D%27'+str(single_date.strftime("%Y-%m-%d"))+'%27&objectIds=&time=&resultType=none&outFields=Bundesland%2C+Landkreis%2C+Altersgruppe%2C+Geschlecht%2C+AnzahlFall%2C+AnzahlTodesfall%2C+Meldedatum%2C+NeuerFall&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token='
	# 	with urllib.request.urlopen(url_str) as url:
	# 		json_data = json.loads(url.read().decode())
	# 		list_data.append(json_data)

	url_str = 'https://services7.arcgis.com/mOBPykOjAyBO2ZKk/ArcGIS/rest/services/RKI_COVID19/FeatureServer/0/query?where=Meldedatum%3D%27'+ single_date + '%27&objectIds=&time=&resultType=none&outFields=Bundesland%2C+Landkreis%2C+Altersgruppe%2C+Geschlecht%2C+AnzahlFall%2C+AnzahlTodesfall%2C+Meldedatum%2C+NeuerFall&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token='
	with urllib.request.urlopen(url_str) as url:
		json_data = json.loads(url.read().decode())
		list_data.append(json_data)

	#Parses list of data
	n_data = len(list_data[0]['features'])
	data_flat = []
	data_flat = [list_data[0]['features'][i]['attributes'] for i in range(n_data)]

	return data_flat

def download_rki_landkreis(landkreis='LK Göttingen'):
	
	#url_str = 'https://services7.arcgis.com/mOBPykOjAyBO2ZKk/ArcGIS/rest/services/RKI_COVID19/FeatureServer/0/query?where=Landkreis%3D%27'+ landkreis + '%27&objectIds=&time=&resultType=none&outFields=Bundesland%2C+Landkreis%2C+Altersgruppe%2C+Geschlecht%2C+AnzahlFall%2C+AnzahlTodesfall%2C+Meldedatum%2C+NeuerFall&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token='
	url_str = 'https://services7.arcgis.com/mOBPykOjAyBO2ZKk/ArcGIS/rest/services/RKI_COVID19/FeatureServer/0/query?where=Landkreis%3D%27LK+G%C3%B6ttingen%27&objectIds=&time=&resultType=none&outFields=Bundesland%2C+Landkreis%2C+Altersgruppe%2C+Geschlecht%2C+AnzahlFall%2C+AnzahlTodesfall%2C+Meldedatum%2C+NeuerFall&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token='

	list_data = []

	url = urllib.request.urlopen(url_str)
	json_data = json.loads(url.read().decode())
	list_data.append(json_data)

	#Parses list of data
	n_data = len(list_data[0]['features'])
	data_flat = []
	data_flat = [list_data[0]['features'][i]['attributes'] for i in range(n_data)]

	return pd.DataFrame(data_flat)


#Runs coronavirus.py to update plots
if __name__ == '__main__':

	#Parameters
	days_pred = 3
	m_days = 3

	update_local('all')
	update_countries(days_pred, m_days)