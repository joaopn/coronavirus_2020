import pandas as pd
import seaborn as sns
import scipy.stats as st
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

def download_data_df(case_type = 'confirmed', df_type='field'):

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
			file_save = folder + country.lower() + '_' + datatype + '.csv'
			df[df['Country/Region']==country][['date',datatype]].to_csv(file_save, index=False)

def get_data():

	data_types = ['confirmed', 'recovered', 'deaths']

	df = download_data_df('confirmed', 'field')
	df2 = download_data_df('recovered', 'field')
	df3 = download_data_df('deaths', 'field')

	#TODO: check dfs are equal

	#Joins data
	df['recovered'] = df2['recovered']
	df['deaths'] = df3['deaths']
	del df2, df3

	#Calculates current infections
	df['current'] = df['confirmed'] - df['deaths'] - df['recovered']

	#Calculates new infections
	country_list = df['Country/Region'].unique()

	#Calculates difference in infections
	df['new'] = df['confirmed'].diff()
	df.loc[df['date']==df['date'].min(), 'new'] = 0

	return df

def plot_countries(df,country_list=['Germany', 'Italy', 'France'], lag_countries=None, variable='confirmed', xlim=['2020-02-21','2020-03-17'], ylim=None,ax=None,**kwargs):

	df2 = df[df['Country/Region'].isin(country_list)]

	if lag_countries is not None:
		lag_dict = dict(zip(country_list, lag_countries))
		df2['date_lag'] = df2['date'] - df2['Country/Region'].apply(lambda x:pd.to_timedelta(lag_dict[x], unit='d'))
		x_var = 'date_lag'

	else:
		x_var = 'date'

	if ax is None:
		fig = plt.figure(figsize=(8,4))

	ax = sns.lineplot(data=df2,hue='Country/Region', x=x_var,y=variable, ax=ax,**kwargs)
	
	#Beautifies plot
	ax.set_yscale('log')
	if ylim is not None:
		ax.set_ylim(ylim)
	else:
		ax.set_ylim([1,ax.get_ylim()[1]])

	ax.set_xlim([pd.Timestamp(xlim[0]), pd.Timestamp(xlim[1])])
	ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
	#ax.xaxis.set_major_locator(mdates.DayLocator())
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
	#ax.autoscale(enable=True,axis='y', tight=True)

def plot_prediction(file, days_pred = 3, m_days=3, title='cases', datatype = 'confirmed', ax=None, x_min = None, ylabel='cases',color='r', savefig=None, **kwargs):

	#Parameters
	interval_cl = 0.95
	color_m = 'k'
	color_pred = sns.set_hls_values(color,l=0.2)

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
		plt.figure(figsize=(8,6))
		ax = plt.gca()

	label_1 = 'Data updated on ' + str(df['date'].iloc[-1])[:-9]
	label_2 = 'Prediction (previous {:d} days)'.format(m_days)
	label_3 = 'Increase factor (previous {:d} days)'.format(m_days)

	ax.bar(data=df, x='date', height=datatype, color=color,**kwargs)
	ax.plot(df['date'], df['prediction'], color=color_pred,lw=3,**kwargs)
	ax.bar(data=df_pred, x='date', height='prediction', color=color_pred,**kwargs)
	ax2 = ax.twinx() 
	ax2.plot(df['date'], df['m_median'], '--', color=color_m, lw=2)
	#ax2.plot(df['date'], df['increase_p'], '--', color=color_m, label='Z')
	#ax2.set_ylim([0,3])
	ax2.tick_params(axis='y', labelcolor=color_m)

	#Beautifies plot
	#ax.set_xlim([pd.Timestamp(x_min), df_pred['date'].max() + pd.to_timedelta(1,unit='d')])
	ax.set_xlim([pd.Timestamp(x_min), df['date'].max() + pd.to_timedelta(days_pred + 1,unit='d')])
	ax.set_title(title)
	ax.set_ylabel(ylabel)
	ax2.set_ylabel('Increase factor ')
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

	custom_lines = [Line2D([0], [0], color=color, lw=4), Line2D([0], [0], color=color_pred, lw=4), Line2D([0], [0], linestyle='--', lw=2, color=color_m)]
	ax2.legend(custom_lines, [label_1, label_2, label_3], framealpha=1, facecolor='white', loc='upper left')

	str_ann = 'Source: John Hopkins University'
	ax.annotate(str_ann, xy=(1,0), xycoords=('axes fraction','figure fraction'), xytext=(0,6), textcoords='offset points', ha='right')

	if savefig is not None:
		plt.savefig(savefig)
		plt.close('all')

def update_all(days_pred = 3, m_days = 3):

	#Sets backend
	old_backend = matplotlib.get_backend()
	matplotlib.use('Agg')

	datatypes = ['current', 'confirmed', 'recovered', 'deaths', 'new']

	ylabel = {'current':'Current cases',
	 'confirmed':'Confirmed cases',
	 'recovered':'Recovered cases',
	 'deaths': 'Number of deaths', 
	 'new': 'New cases'}


	#Gets all unique countries
	df = get_data()
	country_list = df['Country/Region'].unique().tolist()
	del df

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

	#Saves countries to unique csv files
	save_csv(country_list)

	#Saves plots
	folder = 'plots/johnhopkins/'
	if not os.path.exists(folder):
		os.makedirs(folder)

	for country in country_list:
		for datatype in datatypes:
			file_load = 'data/johnhopkins/' + country.lower() + '_' + datatype + '.csv'
			plot_save = folder + country.lower() + '_' + datatype  + '.png'
			plot_prediction(file_load, days_pred, m_days, country, datatype, savefig=plot_save, x_min=x_min[country], ylabel=ylabel[datatype])


	#Restores backend
	matplotlib.use(old_backend)