import pandas as pd
import seaborn as sns
import scipy.stats as st
import numpy as np
import os
from datetime import datetime, timedelta, date
import requests
import urllib
import json
import time

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
	
	url_str = 'https://services7.arcgis.com/mOBPykOjAyBO2ZKk/ArcGIS/rest/services/RKI_COVID19/FeatureServer/0/query?where=Landkreis%3D%27LK+G%C3%B6ttingen%27&objectIds=&time=&resultType=none&outFields=Bundesland%2C+Landkreis%2C+Altersgruppe%2C+Geschlecht%2C+AnzahlFall%2C+AnzahlTodesfall%2C+Meldedatum%2C+NeuerFall&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token='

	list_data = []

	url = urllib.request.urlopen(url_str)
	json_data = json.loads(url.read().decode())
	list_data.append(json_data)

	#Parses list of data
	n_data = len(list_data[0]['features'])
	data_flat = []
	data_flat = [list_data[0]['features'][i]['attributes'] for i in range(n_data)]

	df = pd.DataFrame(data_flat)
	df['date'] = df['Meldedatum'].apply(lambda x: datetime.fromtimestamp(x/1e3).strftime('%d-%m-%Y'))

	return df

def download_rki_idlandkreis(idlandkreis_list=['03159']):
	
	df_keys = ['Bundesland', 'Landkreis', 'Altersgruppe', 'Geschlecht', 'AnzahlFall',
       'AnzahlTodesfall', 'Meldedatum', 'NeuerFall']

	df = pd.DataFrame(columns=df_keys)

	count = 0
	for idlandkreis in idlandkreis_list:
		count+=1
		time.sleep(1)
		print('Downloading {:d} of {:d}'.format(count, len(idlandkreis_list)))
		url_str = 'https://services7.arcgis.com/mOBPykOjAyBO2ZKk/ArcGIS/rest/services/RKI_COVID19/FeatureServer/0/query?where=IdLandkreis%3D%27'+ idlandkreis + '%27&objectIds=&time=&resultType=none&outFields=Bundesland%2C+Landkreis%2C+Altersgruppe%2C+Geschlecht%2C+AnzahlFall%2C+AnzahlTodesfall%2C+Meldedatum%2C+NeuerFall&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token='
		
		with urllib.request.urlopen(url_str) as url:
			json_data = json.loads(url.read().decode())

		n_data = len(json_data['features'])
		data_flat = [json_data['features'][i]['attributes'] for i in range(n_data)]

		df_temp = pd.DataFrame(data_flat)
	
		#Very inneficient, but it will do
		df = pd.concat([df, df_temp], ignore_index=True)

	df['date'] = df['Meldedatum'].apply(lambda x: datetime.fromtimestamp(x/1e3).strftime('%d-%m-%Y'))

	return df

def download_rki(save=None):

	#Gets all unique idlandkreis from data
	url_id = 'https://services7.arcgis.com/mOBPykOjAyBO2ZKk/ArcGIS/rest/services/RKI_COVID19/FeatureServer/0/query?where=0%3D0&objectIds=&time=&resultType=none&outFields=idLandkreis&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=true&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token='
	url = urllib.request.urlopen(url_id)
	json_data = json.loads(url.read().decode())
	n_data = len(json_data['features'])
	unique_ids = [json_data['features'][i]['attributes']['IdLandkreis'] for i in range(n_data)]

	print('Downloading {:d} unique Landkreise'.format(n_data))

	df = download_rki_idlandkreis(unique_ids)

	if save not in [None, 'now']:
		df.to_csv(save, index=False)
	elif save == 'now':
		str_save = datetime.now().strftime('data/rki_data_%y_%m_%d_T%H_%M.csv')
		df.to_csv(str_save, index=False)

	return df

def load_rki_age(file, variable='AnzahlFall', landkreis = None, bundesland = None):

	if variable not in ['AnzahlFall', 'AnzahlTodesfall']:
		ValueError('Invalid variable. Valid options: "AnzahlFall", "AnzahlTodesfall"')

	age_groups = ['A00-A04','A05-A14','A15-A34','A35-A59', 'A60-A79', 'A80+']

	#Loads df and set dates as datetime
	df = pd.read_csv(file)
	df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

	#Index for all days
	idx = pd.date_range(df['date'].min(), df['date'].max())

	#Creates df with only age data
	df2 = pd.DataFrame(index=idx)
	for age in age_groups:

		if landkreis is None and bundesland is None:
			idx = (df['Altersgruppe']==age).index
		elif landkres is None:
			idx =  df.loc[(df.Landkreis == 'SK Bayreuth') & (df.Altersgruppe == 'A15-A34')]['Altersgruppe'].index


		df_temp = df.iloc[idx].sort_values('date').groupby('date')[['date',variable]].sum()
		df_temp = df_temp.reindex(idx, fill_value=0)
		df2[age] = df_temp

	#Calculates daily, total, age_total and age_total_p cols
	df2['total'] = df2.sum(axis=1).cumsum()
	df2['daily'] = df2.sum(axis=1)
	for age in age_groups:
		df2[age + '_total'] = df2[age].cumsum()
		df2[age + '_total_p'] = 100*df2[age + '_total']/df2['total']

	return df2
