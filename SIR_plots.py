import pandas as pd;
import numpy  as np;
import matplotlib.pyplot as plt;
import scipy.integrate; 

class SIRD_plots:
    def __init__(self, country, population, beta, gamma, mu, s0, i0, r0, d0, SIR_df = None):
        '''
        SIR_plots constructor. Takes in the name of the country, population, 
        beta value, gamma value, inital SIR conditions, and an SIR dataframe
        to display plots to analyze the SIR model
        '''
        self.country = country;
        self.pop = population;
        self.beta = beta;
        self.gamma = gamma;
        self.mu    = mu
        self.s0 = s0;
        self.i0 = i0;
        self.r0 = r0;
        self.d0 = d0;
        self.df = SIR_df;

    def __SIRD_model (self, y, t, beta, gamma, mu):
        '''
        __SIR_model is a private method that computes the
        SIR differentials
        '''
        S, I, R, D = y;
        dS_dt   = (-beta*S*I)/self.pop;
        dI_dt   = (((beta*S*I)/self.pop) - (gamma*I)) - (mu*I)
        dR_dt   = (gamma*I);
        dD_dt   = (mu*I)
        return    ([ dS_dt , dI_dt , dR_dt, dD_dt ]);

    def __init_plots(self):
        '''
        ___init_plots is a private method that is used to
        adjust the plots so that they are all dark mode and 
        the same size
        '''
        plt.style.use("dark_background");
        plt.figure(figsize = [7,5]);

    def __plot_details(self, title, x, y):
        '''
        __plot_details is a private method that makes it easy to 
        set the title, axis labels and other plot things for all
        of the figures
        '''
        plt.title (self.country + " " + title , size = 25);
        plt.xlabel(x , size = 16);
        plt.ylabel(y , size = 16);
        plt.legend(prop = {'size' : 20});
        plt.show();

    def plot_SIRD_model (self, time):
        '''
        plot_SIR_model calls SIR_model and forms a solution vector of integrated values. It will then plot the solution.
        A timeframe needs to be passed in.
        '''
        t = np.linspace(1,time,10000);
        solution = scipy.integrate.odeint(self.__SIRD_model, [self.s0, self.i0, self.r0, self.d0], t, args = (self.beta, self.gamma, self.mu))
        solution = np.array(solution);
    
        self.__init_plots();
    
        plt.plot(t , solution[:, 0]/self.pop , label = "S(t)" , color = 'orange');
        plt.plot(t , solution[:, 1]/self.pop , label = "I (t)" , color = 'indianred');
        plt.plot(t , solution[:, 2]/self.pop , label = "R (t)" , color = 'cyan');
        plt.plot(t , solution[:, 3]/self.pop , label = "D (t)" , color = 'Crimson');

        self.__plot_details("SIR Model " + str(time) + " Days" , "Time", "Proportion");

    def plot_SIRD_values (self):
        '''
        plot_SIR_values plots SIR values of the data provided in the constructor
        '''
        t = np.linspace(1,len(self.df.index),10000);
        self.__init_plots();
    
        #plt.plot(self.df['SIR_Suceptible'], label = "Suceptible Plot", color = 'yellow');
        plt.plot(self.df['SIR_Infected'],   label = "Infected Plot"  , color = 'crimson');
        plt.plot(self.df['SIR_Recovered'],  label = "Recovered Plot" , color = 'limegreen');
        plt.plot(self.df['SIR_Deaths'],     label = "Death Plot"     , color = 'indianred');
    
        self.__plot_details("SIR Values Plot", "Time", "Counts");

    def modelValueComparison (self):
        '''
        plot_SIR_plot_and_model compared an SIR_model with the SIR values provided in the constructor
        '''

        t = np.linspace(1,time,10000);
        solution = scipy.integrate.odeint(self.__SIRD_model, [self.s0, self.i0, self.r0, self.d0], t, args = (self.beta, self.gamma, self.mu))
        solution = np.array(solution);

        self.__init_plots();
    
        #plt.plot(t , solution[:, 0] , label = "S(t)" , color = 'orange');
        plt.plot(t , solution[:, 1] , label = "I(t)" , color = 'indianred');
        plt.plot(t , solution[:, 2] , label = "R(t)" , color = 'cyan');
        plt.plot(t , solution[:, 3] , label = "D(t)" , color = 'crimson');
    
        #plt.plot(self.df['SIR_Suceptible'], label = "Suceptible Plot", color = 'yellow');
        plt.plot(self.df['SIR_Infected'] ,  label = "Infected Plot"  , color = 'crimson');
        plt.plot(self.df['SIR_Recovered'] , label = "Recovered Plot" , color = 'limegreen');
        plt.plot(self.df['SIR_Deaths']    , label = "Deaths Plot"    , color = 'indianred');
    
        self.__plot_details("SIR Model and Values Plot", "Time", "Counts");

    def S_Comparison (self):
        '''
        S_Comparison plots the suceptible curve against the Suceptible values.
        '''
        time = self.df.shape[0]
        t = np.linspace(1,time,10000);
        solution = scipy.integrate.odeint(self.__SIRD_model, [self.s0, self.i0, self.r0, self.d0], t, args = (self.beta, self.gamma, self.mu))
        solution = np.array(solution);
        self.__init_plots();
    
        plt.plot(t , solution[:, 0] , label = "S(t)" , color = 'orange');
        plt.plot(self.df['SIR_Suceptible'], label = "Suceptible Plot", color = 'yellow');
        self.__plot_details("S Comparison", "Time", "Counts");

    def S_ComparisonTrain (self, trainX):
        '''
        S_Comparison plots the suceptible curve against the Suceptible values.
        '''
        time = self.df.shape[0]
        t = np.linspace(1,time,10000);
        solution = scipy.integrate.odeint(self.__SIRD_model, [self.s0, self.i0, self.r0, self.d0], t, args = (self.beta, self.gamma, self.mu))
        solution = np.array(solution);
        self.__init_plots();
    
        plt.plot(t , solution[:, 0] , label = "S(t)" , color = 'orange');
        plt.plot(self.df['SIR_Suceptible'], label = "Suceptible Plot", color = 'yellow');
        
        plt.plot(trainX['SIR_Suceptible'], label = "Training Set", color = 'white');

        self.__plot_details("S Comparison", "Time", "Counts");

    def I_Comparison (self):
        '''
        I_Comparison plots the infected curve against the infected values.
        '''
        time = self.df.shape[0]
        t = np.linspace(1,time,10000);
        solution = scipy.integrate.odeint(self.__SIRD_model, [self.s0, self.i0, self.r0, self.d0], t, args = (self.beta, self.gamma, self.mu))
        solution = np.array(solution);
        self.__init_plots();

        plt.plot(t , solution[:, 1] , label = "I(t)" , color = 'indianred');
        plt.plot(self.df['SIR_Infected'] , label = "Infected Plot"  , color = 'crimson');

        self.__plot_details("I Comparison", "Time", "Counts");

    def I_ComparisonTrain (self, trainX):
        '''
        I_Comparison plots the infected curve against the infected values.
        '''
        time = self.df.shape[0]
        t = np.linspace(1,time,10000);
        solution = scipy.integrate.odeint(self.__SIRD_model, [self.s0, self.i0, self.r0, self.d0], t, args = (self.beta, self.gamma, self.mu))
        solution = np.array(solution);
        self.__init_plots();

        plt.plot(t , solution[:, 1] , label = "I(t)" , color = 'indianred');
        plt.plot(self.df['SIR_Infected'] , label = "Infected Plot"  , color = 'crimson');
        plt.plot(trainX['SIR_Infected'], label = "Training Set", color = 'white');

        self.__plot_details("I Comparison", "Time", "Counts");

    def R_Comparison (self):
        '''
        R_Comparison plots the recovered curve against the recovered values.
        '''
        time = self.df.shape[0]
        t = np.linspace(1,time,10000);
        solution = scipy.integrate.odeint(self.__SIRD_model, [self.s0, self.i0, self.r0, self.d0], t, args = (self.beta, self.gamma, self.mu))
        solution = np.array(solution);
    
        plt.plot(t , solution[:, 2] , label = "R(t)" , color = 'cyan');
        plt.plot(self.df['SIR_Recovered'] , label = "Recovered Plot" , color = 'limegreen');
    
        self.__plot_details("R Comparison", "Time", "Counts");

    def D_Comparison (self):
        '''
        R_Comparison plots the recovered curve against the recovered values.
        '''
        time = self.df.shape[0]
        t = np.linspace(1,time,10000);
        solution = scipy.integrate.odeint(self.__SIRD_model, [self.s0, self.i0, self.r0, self.d0], t, args = (self.beta, self.gamma, self.mu))
        solution = np.array(solution);
    
        plt.plot(t , solution[:, 3] , label = "D(t)" , color = 'indianred');
        plt.plot(self.df['SIR_Deaths'] , label = "Deaths Plot" , color = 'crimson');
    
        self.__plot_details("D Comparison", "Time", "Counts");

    def R_ComparisonTrain (self, trainX):
        '''
        R_Comparison plots the recovered curve against the recovered values.
        '''
        time = self.df.shape[0]
        t = np.linspace(1,time,10000);
        solution = scipy.integrate.odeint(self.__SIRD_model, [self.s0, self.i0, self.r0, self.d0], t, args = (self.beta, self.gamma, self.mu))
        solution = np.array(solution);
    
        plt.plot(t , solution[:, 2] , label = "R(t)" , color = 'cyan');
        plt.plot(self.df['SIR_Recovered'] , label = "Recovered Plot" , color = 'limegreen');
        plt.plot(trainX['SIR_Recovered'], label = "Training Set", color = 'white');
    
        self.__plot_details("R Comparison", "Time", "Counts");

    def D_ComparisonTrain (self, trainX):
        '''
        R_Comparison plots the recovered curve against the recovered values.
        '''
        time = self.df.shape[0]
        t = np.linspace(1,time,10000);
        solution = scipy.integrate.odeint(self.__SIRD_model, [self.s0, self.i0, self.r0, self.d0], t, args = (self.beta, self.gamma, self.mu))
        solution = np.array(solution);
    
        plt.plot(t , solution[:, 3] , label = "D(t)" , color = 'indianred');
        plt.plot(self.df['SIR_Deaths'] , label = "Deaths Plot" , color = 'crimson');
        plt.plot(trainX['SIR_Deaths'], label = "Training Set", color = 'white');
    
        self.__plot_details("D Comparison", "Time", "Counts");