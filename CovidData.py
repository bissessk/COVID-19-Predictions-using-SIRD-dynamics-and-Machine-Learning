import pandas as pd;
import numpy  as np;

class CovidData:
    '''
    The CovidData Class creates objects that allow the user to retreive specific information from the daily updated COVID19 time series data made available from John Hopkins University
    '''
    def __init__(self):
        '''
        Constructor creates three dataframes that contain the original data of the global cases, deaths, and recoveries respectively
        '''
        self.cases_glob_df  = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",
                                     error_bad_lines = False);
        self.deaths_glob_df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv",
                                     error_bad_lines = False);
        self.recovs_glob_df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv",
                                     error_bad_lines = False);

    def __anyDup(self, thelist):
        ''' 
        __anyDup is a private method that takes in a list and returns a set of duplicates that are contained in the list
        '''
        seen = set()
        dups = []
        for x in thelist:
            if x in seen:
                dups.append(x)
            seen.add(x)
        return set(dups);

    def __getDupCountries(self):
        '''
        __getDupCountries is a private method that returns all of the duplicate countries contained in the data
        '''
        countries = self.cases_glob_df;
        dup_countries = list( self.__anyDup(countries) );
        return dup_countries;

    def __sep_loc_dfs(self, location):
        '''
        __sep_loc_dfs is a private method that subsets the three original dataframes so that it only contains the location of interest
        '''
        cas_df = self.cases_glob_df.loc  [self.cases_glob_df  ['Country/Region'] == location];
        dea_df = self.deaths_glob_df.loc [self.deaths_glob_df ['Country/Region'] == location];
        rec_df = self.recovs_glob_df.loc [self.recovs_glob_df ['Country/Region'] == location];
        return cas_df, dea_df, rec_df

    def __trimDfByStartDate(self, df) :
        cols = list(df);
        count = 0;
        for col in cols:
            if df[col].sum() == 0:
                count +=1;
            else:
                break;
        if count == 0:
            return df;
        else:
            df = df.iloc[ : , count:]
            return df;

    def get(self, location = "Total"):
        '''
        the get method returns a dataframe that contains the total cases, deaths, and recovories every day for a location of interest.
        If no location is passed in, then the function will return a data frame that sums the cases, deaths, and recoveries of every country
        '''
        if location != "Total":            
            if location not in self.__getDupCountries():
                cas_df , dea_df , rec_df = self.__sep_loc_dfs(location);
                dates = list(cas_df.columns[4:]);
                dic = {};
                for date in dates:
                    dic[str(date)] = [(list(cas_df[date])[0]) , (list(dea_df[date])[0]) , (list(rec_df[date])[0])];
                df = pd.DataFrame.from_dict(dic)
                df = self.__trimDfByStartDate(df);
                df = df.transpose();
                df.columns = ['Cases','Deaths','Recoveries'];

            else:
                # In case country has multiple provincies, just sum it up
                cas_df , dea_df , rec_df = self.__sep_loc_dfs(location);
                dates = list(self.cas_df.columns[4:]);
                dic = {};
                for date in dates:
                    dic[str(date)] = [(self.cas_df[date].sum()) ,(self.dea_df[date].sum()) , (self.rec_df[date].sum())]
                df = pd.DataFrame.from_dict(dic)
                df = self.__trimDfByStartDate(df);
                df = df.transpose();
                df.columns = ['Cases','Deaths','Recoveries'];
        else:
            dates = list(self.cases_glob_df.columns[4:]);
            dic = {};
            for date in dates:
                dic[str(date)] = [(self.cases_glob_df[date].sum()) ,(self.deaths_glob_df[date].sum()) , (self.recovs_glob_df[date].sum())]
            df = pd.DataFrame.from_dict(dic).transpose();
            df.columns = ['Cases','Deaths','Recoveries'];
        return df;

    def getSIRD_df(self, location = "Total", population = 7800000000):
        '''
        the getSIR_df method returns a dataframe that contains S I R values every day for a location of interest.
        If no location is passed in, then the function will return a data frame that sums the S I R values for every country
        '''
        loc_df = self.get(location);
        
        #loc_df['SIR_Suceptible'] = (population - (loc_df['Cases'] - loc_df['Deaths'] - loc_df['Recoveries']));
        #loc_df['SIR_Infected']   = (loc_df['Cases'] - loc_df['Deaths'] - loc_df['Recoveries']);
        #loc_df['SIR_Recovered']  = (loc_df['Recoveries']);

        loc_df['SIR_Suceptible'] = (population - (loc_df['Cases']))
        loc_df['SIR_Infected']   = (loc_df['Cases'] - loc_df['Deaths'] - loc_df['Recoveries']);
        loc_df['SIR_Recovered']  = (loc_df['Recoveries'])
        loc_df['SIR_Deaths']     = loc_df['Deaths'];
        
        #SIR_df = loc_df.loc[:, loc_df.columns.intersection(['SIR_Suceptible','SIR_Infected','SIR_Recovered' ])];
        SIR_df = loc_df.loc[:, loc_df.columns.intersection(['SIR_Suceptible','SIR_Infected','SIR_Recovered', 'SIR_Deaths' ])];

        return SIR_df;

    def get_closed_active_cases_df(self, location):
        loc_df = self.get(location);
        loc_df['Closed Cases'] = loc_df['Deaths']    + loc_df['Recovered'];
        loc_df['Active Cases'] = loc_df['Confirmed'] - loc_df['Closed Cases'];
        active_closed_df = loc_df.loc[:, loc_df.columns.intersection(['Active Cases','Closed Cases'])];
        return active_closed_df;


    
