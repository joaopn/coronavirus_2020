import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

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

	folder = 'data/'
	if not os.path.exists(folder):
		os.makedirs(folder)

	df = get_data()
	for country in countries:
		df[df['Country/Region']==country][['date','new']].to_csv(folder + 'new_' + country.lower() + '.csv', index=False)
		df[df['Country/Region']==country][['date','current']].to_csv(folder + 'current_' + country.lower() + '.csv', index=False)
		df[df['Country/Region']==country][['date','confirmed']].to_csv(folder + 'confirmed_' + country.lower() + '.csv', index=False)
		df[df['Country/Region']==country][['date','recovered']].to_csv(folder + 'recovered_' + country.lower() + '.csv', index=False)
		df[df['Country/Region']==country][['date','deaths']].to_csv(folder + 'deaths_' + country.lower() + '.csv', index=False)

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

def plot_countries(df,country_list=['Germany', 'Italy', 'France'], lag_countries=None, variable='confirmed', xlim=['2020-02-21','2020-03-17'], ylim=None):

	df2 = df[df['Country/Region'].isin(country_list)]

	if lag_countries is not None:
		lag_dict = dict(zip(country_list, lag_countries))
		df2['date_lag'] = df2['date'] - df2['Country/Region'].apply(lambda x:pd.to_timedelta(lag_dict[x], unit='d'))
		x_var = 'date_lag'

	else:
		x_var = 'date'

	fig = plt.figure(figsize=(8,4))
	ax = sns.lineplot(data=df2,hue='Country/Region', x=x_var,y=variable)
	
	#Beautifies plot
	ax.set_yscale('log')
	ax = plt.gca()
	if ylim is not None:
		ax.set_ylim(ylim)
	else:
		ax.set_ylim([1,ax.get_ylim()[1]])

	ax.set_xlim([pd.Timestamp(xlim[0]), pd.Timestamp(xlim[1])])
	ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
	#ax.xaxis.set_major_locator(mdates.DayLocator())
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
	#ax.autoscale(enable=True,axis='y', tight=True)

def plot_prediction_csv(csv_file, m_range=[1.19,1.28],days=3, title='cases', datatype = 'confirmed'):

	x_min = '2020-03-01'

	df = pd.read_csv(csv_file)
	df['date'] = pd.to_datetime(df['date'])

	df_pred = pd.DataFrame(columns=['date', 'prediction_min', 'prediction_max'])

	date_min = df['date'].max() + pd.to_timedelta(1,unit='d')
	df_pred['date'] = pd.date_range(start=date_min,periods=days)

	for i in range(days):
		df_pred['prediction_min'][i] = m_range[0]**(i+1)*df[datatype][df.index[-1]]
		df_pred['prediction_max'][i] = m_range[1]**(i+1)*df[datatype][df.index[-1]]
	df_pred['prediction'] = (df_pred['prediction_max'] + df_pred['prediction_min'])/2

	plt.figure(figsize=(12,5))
	plt.bar(data=df, x='date', height=datatype, color='b', label=datatype)
	plt.bar(data=df_pred, x='date', height='prediction', color='r', label='prediction')

	#Beautifies plot
	ax = plt.gca()
	ax.set_xlim([pd.Timestamp(x_min), df_pred['date'].max() + pd.to_timedelta(1,unit='d')])
	ax.set_title(title)
	ax.set_ylabel('Cases')
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

