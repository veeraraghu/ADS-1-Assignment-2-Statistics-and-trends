#Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def transpose(data):
    data.drop(['Country Code', 'Indicator Name',
              'Indicator Code'], axis=1, inplace=True)
    data_tr = pd.DataFrame.transpose(data)
    data_tr.columns = data_tr.iloc[0]
    data_tr.drop('Country Name', axis=0, inplace=True)

    return data, data_tr


def multi_bar_plot(data, labels):

    plt.figure()
    ax = plt.subplot()
    years = data.columns
    n = len(years)
    b_width = 0.8/n
    offset = b_width/2
    x_ticks = data.index
    for i, year in enumerate(years):
        ax.bar([j+offset+b_width*i for j in range(len(x_ticks))], data[year],
               width=b_width, label=year)

    ax.set_xlabel('Country Name')
    ax.set_ylabel(labels[0])
    ax.set_title(labels[1])

    ax.set_xticks([j+0.4 for j in range(len(x_ticks))])
    ax.set_xticklabels(x_ticks, rotation=45)
    ax.legend()
    plt.show()


def country_data(list_of_dfs, list_of_indicators, country, years):
    country_data = pd.DataFrame()
    for i in range(len(list_of_dfs)):
        if list_of_indicators[i] == 'CO2 emissions':
            country_data[list_of_indicators[i]
                         ] = list_of_dfs[i].loc[years[0]:years[1], country].sum(axis=1)
        else:
            country_data[list_of_indicators[i]
                         ] = list_of_dfs[i].loc[years[0]:years[1], country].astype(int)

    return country_data


def heat_map(data, lab, color):
    corr_matrix = np.array(data.corr())
    print('array is ')
    print(corr_matrix)

    plt.figure()
    plt.imshow(china.corr(), cmap=color,
               interpolation='nearest', aspect='auto')
    plt.xticks(range(len(china.columns)), data.columns, rotation=90)
    plt.yticks(range(len(china.columns)), data.columns)

    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            text = plt.text(j, i, corr_matrix[i, j].round(2),
                            ha="center", va="center", color="w")
    plt.colorbar()
    plt.title(lab)
    plt.show()


#Creating Population Total dataframes with columns as Years and Countires
pop_total_y, pop_total_c = transpose(pd.read_csv('population_total.csv'))
print('Population Total - Statistical data description over few years')
print(pop_total_y.loc[:, ['1960', '1990',
      '2005', '2015', '2021']].describe(), '\n')


'''
Countries in this dataset include non-countries, so sorted the dataframe to 
get countries with highest population in 2021 and exported to csv to gather 
all countries and form a new list/dataframe of top 10 countries with 
highest population
'''
all_countries = pop_total_y.sort_values(
    '2021', ascending=False, ignore_index=True).iloc[:, 0]
all_countries.to_csv('all_countries.csv')

'''
By looking at the csv file created we can see at which indexes we have actual 
countries and can create a new list of top 10 countries with highest 
population in 2021
'''

countries_10_pop = all_countries.iloc[[
    15, 16, 45, 46, 47, 48, 49, 50, 51, 52]].to_list()
print('Top 10 Countries with highest population in 2021')
print(countries_10_pop)

#Now that we have countries for which we need to perform statistics,
#Plotting the population variation for these countries over years

#------
print(pop_total_y.loc[:,
                      ['Country Name', '1990', '1995', '2000', '2005', '2010', '2015']])

pop = pop_total_y.loc[:,
                      ['Country Name', '1990', '1995', '2000', '2005', '2010', '2015']]
pop.index = pop['Country Name']
pop.drop('Country Name', axis=1, inplace=True)
pop = pop.loc[countries_10_pop, :]

print('with country as index')
print(pop)
labels = ['Popuation in Millions', 'Population growth over years']
multi_bar_plot(pop, labels)

co2_emissions_y, co2_emissions_c = transpose(
    pd.read_csv('co2_emissions(kt).csv'))

print('testttt')
print(co2_emissions_y.head())
co2_emissions_y = co2_emissions_y.groupby('Country Name').sum()
# co2_emissions_y.index=co2_emissions_y['Country Name']
# co2_emissions_y.drop('Country Name',inplace=True)
co2_emissions_y = co2_emissions_y.loc[countries_10_pop,
                                      ['1990', '1995', '2000', '2005', '2010', '2015']]

co2 = co2_emissions_y.copy()
print('co2 with country as index')
print(co2)
labels = ['CO2 emissions', 'Total CO2 emissions in kt']
multi_bar_plot(co2, labels)

years = ['1990', '1995', '2000', '2005', '2010', '2015']

power_y, power_c = transpose(pd.read_csv(
    'electric_power_consumption(kWh per capita).csv'))

power_y = power_y.groupby('Country Name').sum()
# co2_emissions_y.index=co2_emissions_y['Country Name']
# co2_emissions_y.drop('Country Name',inplace=True)
power_y = power_y.loc[countries_10_pop, years]

power = power_y.copy()
labels = ['Power kWh per capita', 'Electric Power Consumption kWh per capita']
multi_bar_plot(power, labels)

co2_pc_y, co2_pc_c = transpose(pd.read_csv('co2_emissions_liquid.csv'))

co2_pc = co2_pc_y.loc[:,
                      ['Country Name', '1990', '1995', '2000', '2005', '2010', '2015']]
co2_pc.index = co2_pc['Country Name']
co2_pc.drop('Country Name', axis=1, inplace=True)
co2_pc = co2_pc.loc[countries_10_pop, :]

print(co2_pc)
labels = ['CO2 emissions kt', 'CO2 emissions from liquid fuel consumption']
multi_bar_plot(co2_pc, labels)


co2_g_y, co2_g_c = transpose(pd.read_csv('co2_emissions_gaseous.csv'))

co2_g = co2_g_y.loc[:,
                    ['Country Name', '1990', '1995', '2000', '2005', '2010', '2015']]
co2_g.index = co2_g['Country Name']
co2_g.drop('Country Name', axis=1, inplace=True)
co2_g = co2_g.loc[countries_10_pop, :]

print(co2_g)
labels = ['CO2 gaseous emissions kt',
          'CO2 emissions from gaseous fuel consumption']
multi_bar_plot(co2_g, labels)

methane_y, methane_c = transpose(pd.read_csv('methane_emissions.csv'))

methane = methane_y.loc[:,
                        ['Country Name', '1990', '1995', '2000', '2005', '2010', '2015']]
methane.index = methane['Country Name']
methane.drop('Country Name', axis=1, inplace=True)
methane = methane.loc[countries_10_pop, :]

print(methane)
labels = ['Methane emissions kt', 'Methane emissions']
multi_bar_plot(methane, labels)

greenhouse_y, greenhouse_c = transpose(
    pd.read_csv('total_greenhouse_gases.csv'))

greenhouse = greenhouse_y.loc[:,
                              ['Country Name', '1990', '1995', '2000', '2005', '2010', '2015']]
greenhouse.index = greenhouse['Country Name']
greenhouse.drop('Country Name', axis=1, inplace=True)
greenhouse = greenhouse.loc[countries_10_pop, :]

print(greenhouse)
labels = ['Greenhouse emissions kt',
          'Total Greenhouse gas emissions of kt equivalent']
multi_bar_plot(greenhouse, labels)

agri_land_y, agri_land_c = transpose(pd.read_csv('agriculturan_land.csv'))

agri_land = agri_land_y.loc[:,
                            ['Country Name', '1990', '1995', '2000', '2005', '2010', '2015']]
agri_land.index = agri_land['Country Name']
agri_land.drop('Country Name', axis=1, inplace=True)
agri_land = agri_land.loc[countries_10_pop, :]

print(agri_land)
labels = ['Agriculture land in Sq KM', 'Agricultural land available']
multi_bar_plot(agri_land, labels)

list_of_dfs = [pop_total_c, co2_emissions_c, power_c, co2_pc_c, co2_g_c,
               methane_c, greenhouse_c, agri_land_c]
list_of_indicators = ['Total Population', 'CO2 emissions', 'Electric power consumed',
                      'Liquid fuel emissions', 'Gas fuel emissions', 'Methane emissions',
                      'Greenhouse emissions', 'Agricultural Land']

years = ['1990', '2014']

china = country_data(list_of_dfs, list_of_indicators, 'China', years)
heat_map(china, 'China', 'gist_rainbow')

india = country_data(list_of_dfs, list_of_indicators, 'India', years)
heat_map(india, 'India', 'jet')

united_states = country_data(
    list_of_dfs, list_of_indicators, 'United States', years)
heat_map(united_states, 'United States', 'plasma_r')


elctrcty_access_y, elctrcty_access_c = transpose(
    pd.read_csv('access_to_electricity(% of population).csv'))
print(elctrcty_access_c.head())
acs = elctrcty_access_c.loc['2000':'2020', countries_10_pop].copy()
acs.to_csv('acs_elc.csv')
print(acs)

plt.figure()
for i in acs.columns:
    plt.plot(acs.index, acs[i], label=i, linestyle=':')
plt.xlabel('Year')
plt.ylabel('%Access to electricity')
plt.title('Access to electricity in % to population')
plt.xticks(acs.index[::2])
plt.legend()
plt.show()

pop_growth = pop_total_c.copy()
pop_growth = pop_growth.loc[:'2021', countries_10_pop]
pop_growth.to_csv('pop_growth.csv')
print(pop_growth.head())

plt.figure()
for i in pop_growth.columns:
    plt.plot(pop_growth.index,
             pop_growth[i].pct_change()*100, label=i, linestyle=':')
plt.xlabel('Year')
plt.ylabel('%Population growth')
plt.title('Population growth over years')
plt.xticks(pop_growth.index[::10])
plt.legend(title='Countries', bbox_to_anchor=(1, 1))
plt.show()
