#Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Defining function to transpose data
def transpose_of_data(data):
    '''
    This function will create transpose of the given data and returns 
    both given data and transposed data.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame for which we find transpose.

    Returns
    -------
    data : pandas.DataFrame
        Given dataframe for which transpose is found.
    data_tr : pandas.DataFrame
        Transposed dataframe of given data.

    '''
    data_tr = pd.DataFrame.transpose(data)
    data_tr.header = data_tr.iloc[0, :]
    data_tr = data_tr.iloc[1:, :]

    return data, data_tr


#Defining function to transpose data and remove few non_numerical rows
def transpose(data):
    '''
    This function will create transpose of the given data and removes 
    non-numeric columns such as Country Code, Indicator Code and Indicator 
    Name which are as part of world bank data and returns both given data and 
    transposed data.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame for which we find transpose.

    Returns
    -------
    data : pandas.DataFrame
        Given dataframe for which transpose is found.
    data_tr : pandas.DataFrame
        Transposed dataframe of given data.

    '''
    data.drop(['Country Code', 'Indicator Name',
              'Indicator Code'], axis=1, inplace=True)
    data_tr = pd.DataFrame.transpose(data)
    data_tr.columns = data_tr.iloc[0]
    data_tr.drop('Country Name', axis=0, inplace=True)

    return data, data_tr


#Defining function that plots bargraph for multiple countries and years of a
#indicator using subplots withing same plot
def multi_bar_plot(data, labels):
    '''
    This function produce a barplot of given data indicator values of 
    countries in index  over years in columns.

    Parameters
    ----------
    data : pandas.dataFrame
        Indicator data of selected countries over chosen years as as DataFrame.
    labels : List
        List of ylabel and plot title used for the plot.

    Returns
    -------
    Displays the bar plot.

    '''
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


#Defining function that creates new dataframe to get country specific
#indicators data
def country_data(list_of_dfs, list_of_indicators, country, years):
    '''
    This function will create a new dataframe with indicator data of a 
    specified country over years.

    Parameters
    ----------
    list_of_dfs : List
        List of dataframes where country specific indicator data is collecetd.
    list_of_indicators : List
        List of strings which are used as column names of our new dataframe.
    country : STR
        String of our country name which is used to locate data in indicator 
        dataframes.
    years : List
        List of two years (Starting year and Ending year) which are used to 
        select data from indicators dataframes.

    Returns
    -------
    country_data : pandas.DataFrame
        DataFrame which contains country specific indicator data as columns 
        and years as index.

    '''
    country_data = pd.DataFrame()
    for i in range(len(list_of_dfs)):
        if list_of_indicators[i] == 'CO2 emissions':
            country_data[list_of_indicators[i]
                         ] = list_of_dfs[i].loc[years[0]:years[1],
                                                country].sum(axis=1)
        else:
            country_data[list_of_indicators[i]
                         ] = list_of_dfs[i].loc[years[0]:years[1],
                                                country].astype(int)

    return country_data


#Defining function that plots correlation heatmap
def heat_map(data, lab, color):
    '''
    This function will produce a correlation coefficient heatmap of given data 
    and prints the correlation coefficient matrix.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame for which correlation to be visualised.
    lab : STR
        String of our data name which is used title of the heatmap produced.
    color : STR
        String of cmap attribute for plotting heatmap.

    Returns
    -------
    Displays the heatmap plot.

    '''
    corr_matrix = np.array(data.corr())
    print('Correlation Coefficient matrix of ', lab, ' on indicators is')
    print(corr_matrix, '\n')
    plt.figure()
    plt.imshow(corr_matrix, cmap=color,
               interpolation='nearest', aspect='auto')
    plt.xticks(range(len(data.columns)), data.columns, rotation=90)
    plt.yticks(range(len(data.columns)), data.columns)

    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            plt.text(j, i, corr_matrix[i, j].round(2),
                     ha="center", va="center", color="black")
    plt.colorbar()
    plt.title(lab)
    plt.show()


#Reading Population Total data
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
countries_6_pop = countries_10_pop[:6]
print('Top 6 Countries with highest population in 2021')
print(countries_6_pop, '\n')

'''
Now that we have countries for which we need to perform statistics,
Plotting the population variation for these countries over years
'''

pop_growth = pop_total_c.copy()
pop_growth = pop_growth.loc[:'2021', countries_6_pop]

plt.figure()
'''
pct_change calculates %difference between the value in the row and the row 
ahead, in this way we can achieve the population growth or loss over years
'''
for i in pop_growth.columns:
    plt.plot(pop_growth.index,
             pop_growth[i].pct_change()*100, label=i, linestyle='--')
plt.xlabel('Year')
plt.ylabel('%Population growth')
plt.title('Population growth over years')
plt.xticks(pop_growth.index[::10])
plt.legend(title='Countries', bbox_to_anchor=(1, 1))
plt.show()

#Slicing data for selecting specific countries and years
pop = pop_total_y.loc[:,
                      ['Country Name', '1990', '1995', '2000', '2005',
                       '2010', '2015']]
pop.index = pop['Country Name']
pop.drop('Country Name', axis=1, inplace=True)
pop = pop.loc[countries_10_pop, :]

#Plotting barplot for seeing variation in population over varied years
labels = ['Popuation in Millions', 'Population growth over years']
multi_bar_plot(pop, labels)

#Reading CO2 emissions data
co2_emissions_y, co2_emissions_c = transpose(
    pd.read_csv('co2_emissions(kt).csv'))
years = ['1990', '1995', '2000', '2005', '2010', '2015']

'''
In CO2 emissions data, we have multiple columns with same country and they all 
are CO2 emissions caused by different processes or reasons and total emission 
included.
In order to clear the confusion, I did a groupby over Country Name with a mean 
operation over those which gives the exact CO2 emission of the country.
'''

co2_emissions_y = co2_emissions_y.groupby('Country Name').sum()
co2_emissions_y = co2_emissions_y.loc[countries_6_pop, years]
co2 = co2_emissions_y.copy()
print('CO2 emissions data of selected countries and years')
print(co2, '\n')

#Plotting barplot for seeing variation in CO2 emissions over varied years
labels = ['CO2 emissions', 'Total CO2 emissions in kt']
multi_bar_plot(co2, labels)

#Reading ELectric Power Consumption Data
power_y, power_c = transpose(pd.read_csv(
    'electric_power_consumption(kWh per capita).csv'))

power_y = power_y.set_index('Country Name', drop=True)
power = power_y.loc[countries_6_pop, years].copy()

#Plotting barplot for seeing variation in Power consumption over varied years
labels = ['Power kWh per capita', 'Electric Power Consumption kWh per capita']
multi_bar_plot(power, labels)

#Reading CO2 emissions of liquid fuel consumed data
co2_pc_y, co2_pc_c = transpose(pd.read_csv('co2_emissions_liquid.csv'))
co2_pc = co2_pc_y.loc[:,
                      ['Country Name', '1990', '1995', '2000', '2005',
                       '2010', '2015']].copy()
co2_pc.index = co2_pc['Country Name']
co2_pc.drop('Country Name', axis=1, inplace=True)
co2_pc = co2_pc.loc[countries_6_pop, :]

#Plotting barplot for seeing variation in CO2 emissions over liquid fuel burnt
# over varied years
labels = ['CO2 emissions kt', 'CO2 emissions from liquid fuel consumption']
multi_bar_plot(co2_pc, labels)

#Reading CO2 emissions of gaseous fuel consumed data
co2_g_y, co2_g_c = transpose(pd.read_csv('co2_emissions_gaseous.csv'))
co2_g = co2_g_y.loc[:,
                    ['Country Name', '1990', '1995', '2000', '2005',
                     '2010', '2015']]
co2_g.index = co2_g['Country Name']
co2_g.drop('Country Name', axis=1, inplace=True)
co2_g = co2_g.loc[countries_6_pop, :]

#Plotting barplot for seeing variation in CO2 emissions over gaseous fuel
#burnt over varied years
labels = ['CO2 gaseous emissions kt',
          'CO2 emissions from gaseous fuel consumption']
multi_bar_plot(co2_g, labels)

#Reading Methane emissions data
methane_y, methane_c = transpose(pd.read_csv('methane_emissions.csv'))

methane = methane_y.loc[:,
                        ['Country Name', '1990', '1995', '2000', '2005',
                         '2010', '2015']]
methane.index = methane['Country Name']
methane.drop('Country Name', axis=1, inplace=True)
methane = methane.loc[countries_6_pop, :]

#Plotting barplot for seeing variation in Methane emission over varied years
labels = ['Methane emissions kt', 'Methane emissions']
multi_bar_plot(methane, labels)

#Reading Greenhouse gas emissions data
greenhouse_y, greenhouse_c = transpose(
    pd.read_csv('total_greenhouse_gases.csv'))

greenhouse = greenhouse_y.loc[:,
                              ['Country Name', '1990', '1995', '2000', '2005',
                               '2010', '2015']]
greenhouse.index = greenhouse['Country Name']
greenhouse.drop('Country Name', axis=1, inplace=True)
greenhouse = greenhouse.loc[countries_6_pop, :]

#Plotting barplot for seeing variation in Greenhouse gas emissions
#over varied years
labels = ['Greenhouse emissions kt',
          'Total Greenhouse gas emissions of kt equivalent']
multi_bar_plot(greenhouse, labels)

#Reading Agricultural Land area data
agri_land_y, agri_land_c = transpose(pd.read_csv('agriculturan_land.csv'))

agri_land = agri_land_y.loc[:,
                            ['Country Name', '1990', '1995', '2000', '2005',
                             '2010', '2015']]
agri_land.index = agri_land['Country Name']
agri_land.drop('Country Name', axis=1, inplace=True)
agri_land = agri_land.loc[countries_6_pop, :]

#Plotting barplot for seeing variation in Agricultural land area
#over varied years
labels = ['Agriculture land in Sq KM', 'Agricultural land available']
multi_bar_plot(agri_land, labels)

#Creating dataframes as list for accessing specific country data
#from all indicators
list_of_dfs = [pop_total_c, co2_emissions_c, power_c, co2_pc_c, co2_g_c,
               methane_c, greenhouse_c, agri_land_c]
list_of_indicators = ['Total Population', 'CO2 emissions',
                      'Electric power consumed', 'Liquid fuel emissions',
                      'Gas fuel emissions', 'Methane emissions',
                      'Greenhouse emissions', 'Agricultural Land']
#Years have a list of two years where in between we have all indicators
#data without any missing values
years = ['1990', '2014']

#Creating China country specific data for all indicators and plotting a
#correlation heatmap
china = country_data(list_of_dfs, list_of_indicators, 'China', years)
heat_map(china, 'China', 'gist_rainbow')

#Creating India country specific data for all indicators and plotting a
#correlation heatmap
india = country_data(list_of_dfs, list_of_indicators, 'India', years)
heat_map(india, 'India', 'cool_r')

#Creating United States country specific data for all indicators and plotting
#a correlation heatmap
united_states = country_data(
    list_of_dfs, list_of_indicators, 'United States', years)
heat_map(united_states, 'United States', 'plasma_r')


#Reading Access to Electricity data
elctrcty_access_y, elctrcty_access_c = transpose(
    pd.read_csv('access_to_electricity(% of population).csv'))
acs = elctrcty_access_c.loc['2000':'2020', countries_6_pop].copy()

#Plotting % Population Access to Electricity data of selected countries from
#2000 to 2020
plt.figure()
for i in acs.columns:
    plt.plot(acs.index, acs[i], label=i, linestyle=':')
plt.xlabel('Year')
plt.ylabel('%Access to electricity')
plt.title('Access to electricity in % to population')
plt.xticks(acs.index[::2])
plt.legend(title='Countries', bbox_to_anchor=(1, 1))
plt.show()
