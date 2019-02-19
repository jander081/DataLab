

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin




#------------------------------------------------------------


class TypeSelector(BaseEstimator, TransformerMixin):
    
    '''np.object, np.number, np.bool_'''
    
    def __init__(self, dtype1, dtype2=None, dtype3=None):
        self.dtype1 = dtype1
        self.dtype2 = dtype2
        self.dtype3 = dtype3

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        assert isinstance(X, pd.DataFrame), "Gotta be Pandas"
        
        if self.dtype3 != None:
            
            output = (X.select_dtypes(include=[self.dtype1]),
                   X.select_dtypes(include=[self.dtype2]),
                   X.select_dtypes(include=[self.dtype3]))
            
        elif self.dtype2 != None:
            output = (X.select_dtypes(include=[self.dtype1]),
                   X.select_dtypes(include=[self.dtype2]))
            
        else:
            
            output = (X.select_dtypes(include=[self.dtype1]))
            
        return output
        
from sklearn.preprocessing import StandardScaler 

#------------------------------------------------------------
        

class StandardScalerDf(StandardScaler):
    
    """
    DataFrame Wrapper around StandardScaler; Recursive override
    """

    def __init__(self, copy=True, with_mean=True, with_std=True):
        super(StandardScalerDf, self).__init__(copy=copy,
                                               with_mean=with_mean,
                                               with_std=with_std)

    def transform(self, X, y=None):
        z = super(StandardScalerDf, self).transform(X.values)
        return pd.DataFrame(z, index=X.index, columns=X.columns)

#------------------------------------------------------------
       
    
        
from fancyimpute import SoftImpute

class SoftImputeDf(SoftImpute):
    
    """
    DataFrame Wrapper around SoftImpute
    """

    def __init__(self, shrinkage_value=None, convergence_threshold=0.001,
                 max_iters=100,max_rank=None,n_power_iterations=1,init_fill_method="zero",
                 min_value=None,max_value=None,normalizer=None,verbose=True):
        
        super(SoftImputeDf, self).__init__(shrinkage_value=shrinkage_value, 
                                           convergence_threshold=convergence_threshold,
                                           max_iters=max_iters,max_rank=max_rank,
                                           n_power_iterations=n_power_iterations,
                                           init_fill_method=init_fill_method,
                                           min_value=min_value,max_value=max_value,
                                           normalizer=normalizer,verbose=False)

    

    def fit_transform(self, X, y=None):
        
        assert isinstance(X, pd.DataFrame), "Must be pandas dframe"
        
        for col in X.columns:
            if X[col].isnull().sum() < 10:
                X[col].fillna(0.0, inplace=True)
        
        z = super(SoftImputeDf, self).fit_transform(X.values)
        return pd.DataFrame(z, index=X.index, columns=X.columns)


#------------------------------------------------------------

class CourseSeparater:
    """
    Separates the courses and converts to bools. The initial 'for' loop 
    that drops singular cardinalities does much of the work. After that, 
    it is simply a matter of determining the threshold for relevant classes.
    
    To use this, instantiated the class and run the course_df method. It will
    return the separated dframes.  
    
    Parameters
    -------------
    
    X: pandas dataframe
    
    core: boolean, default='True'
        returns a core course dframe with min limit threshold
        
    limit: int, default=300
        specifies at least 300 attendees from the dataset
    
    """
    
    def __init__(self, X, y=None):
        self.data = X
        self.label = y
        self.courses = []
        self.core_courses = []
    
        # Any feats with 1 value are immediately discarded 
        # this gets rid of a lot of headache
        for col in self.data.columns:
            if len(set(self.data[col])) == 1:
                self.data.drop([col], axis=1, inplace=True)
                
        # then define the cols        
        self.non_course = list(self.data.columns)
    
    def course_separater(self):
        
        for col in self.data.columns:
            if col[-5:] == 'Taken':
                self.non_course.remove(col) 
                self.courses.append(col)
         # 2 lists for cols       
        return self.non_course, self.courses
                       
     
    def core_course_cols(self):
        # the limit is set in the next func
        for course in self.courses:
            # the additional [:, None] is a reshaping shortcut
            if self.data[course].value_counts()[:, None][1] > self.limit:
                self.core_courses.append(course)
                
        return self.core_courses
    
    
    def course_df(self, core=True, limit=300):
        
        # only method that needs to be run
        self.limit = limit
        binary = lambda x: 1 if x == 'Yes' else 0
        non, courses = self.course_separater()
        df_courses = self.data[courses].applymap(binary).astype('bool')
        
        if core==True:
            core_c = self.core_course_cols()
            df_core = self.data[core_c].applymap(binary).astype('bool')
            
            return self.data[non], df_courses, df_core
            
        else:
            
            return self.data[non], df_courses
    


#---------------------------------------------------------------------------   

class FreqFeatures(BaseEstimator, TransformerMixin):
    
    """
    Intended for categorical high cardinality features. This should be used
    in a pipeline preceded by a TypeSelector(np.object)

    The class generates a dict for freq count and maps to the categorical 
    feat value. For the reduce func, this class uses np.vectorize to reduce
    computational time. This class was originally written for a much larger
    dataset
    
    Parameters
    -------------
        
    min_: int, default=50
        The minimum number of categorical values a feature must have in order 
        to be included. If the values do not exceed the max, the class will return
        the numeric freq count feat and the reduced object feat.
        
    max_: int, default=1000
        If feature value count exceeds the max_, only the freq vec will be allowed
        to remain. The original feat will be dropped and a reduction will not be
        performed
    
    Attributes
    -------------
    
    drops: list;
        View all the original features that were dropped
        
    """
       
    def __init__(self, min_=50, max_=1000):
        self.min_ = min_
        self.max_ = max_
        self.drops = []
        
    def make_dict(self, col):
        
        # generates the dict
        df = pd.DataFrame(self.data[col].value_counts())
        df.reset_index(level=0, inplace=True)
        df.rename(columns={'index': 'key', col: 'value'}, inplace=True)
        df_dict = defaultdict(list)
        for k, v in zip(df.key, df.value):
            df_dict[k] = (int(v))
        return df_dict
    
    @staticmethod
    def reduce(x,y):
        # easy to add/subtract discriminators
        if x <= 5:
            return 'rare'
        elif x <= 15:
            return 'less_common'
        else:
            return y

        
    def fit(self, X, y=None):
        return self

    
    def transform(self, X):
        self.data = X
        
        assert isinstance(self.data, pd.DataFrame), 'pls enter dframe'
        
        for col in self.data.columns:
            dict_ = self.make_dict(col) 
            # vec created
            freq_vec = self.data[col].map(dict_)
            
            # Will not try to reduce super high cardinality (eg zips)
            if len(set(self.data[col])) > self.max_:
                # attribute
                self.drops.append(col)
                self.data.drop([col], axis=1, inplace=True)
                self.data[col + '_freq'] = freq_vec
                
            elif len(set(self.data[col])) > self.min_:
                # Uses the numpy vectorize func
                y = self.data[col]
                vectfunc = np.vectorize(self.reduce,cache=False)
                vec = np.array(vectfunc(freq_vec,y))
                
                self.data[col + '_freq'] = freq_vec
                self.data[col + '_reduce'] = vec
                self.data.drop([col], axis=1, inplace=True)
                
        return self.data
  

#---------------------------------------------------------------------------   
              

class FeatureMaps:
    
    def __init__(self, X, y=None):
        self.data = X
        self.target = y
                   
      
    def create_bools(self):
       
        for col in self.data.columns:
            if self.data[col].iloc[0] in ['Yes', 'No', 'N', 'Y']:
                self.data[col] = self.data[col].apply(lambda x: 1 if x[0]=='Y' else 0).astype('bool')
         
        # more bools
        bools = ['RESIDENCE', 'STU_NATION', 'ADMISSION_TERM']
        self.data['Commuter'] = self.data[bools[0]].apply(lambda x: 1 if (x == 'Commuter' or x == 'Not Captured') else 0).astype('bool')
        self.data['US_res'] = self.data[bools[1]].apply(lambda x: 1 if x == 'USA' else 0).astype('bool')
        self.data['Summer_adm'] = self.data[bools[2]].apply(lambda x: 1 if str(x)[-2:] == '05' else 0).astype('bool')
        self.data.drop(bools, axis=1, inplace=True)
        
    
    def to_categorical(self, list_):
        assert isinstance(list_, list), 'pls enter list'
        for feat in list_:
            self.data[feat + '_cat'] = self.data[feat].apply(lambda x: 'cat_' + str(x))
            self.data.drop([feat], axis=1, inplace=True)
          
        
    def run_all(self):
        
        
        self.to_categorical(['ETHNICITY_CODE', 'STU_CLASSIFICATION'])
        self.create_bools()
        
        
# ------------------------------------------------------------
        
class SelectFeatures(BaseEstimator, TransformerMixin):
    
    """
    Used with Kbins to select features with sufficient cardinality. Could
    probably just join this with kbins
    """
    
    def __init__(self, val_count=50, categorical=False):
        self.val_count = val_count
        self.categorical = categorical
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        feat = pd.DataFrame()
        
        if self.categorical==False:           
            for col in X.columns:
                if len(X[col].value_counts()) > self.val_count:              
                    X[col + '_bin'] = X[col]
                    feat = pd.concat([feat, X[col + '_bin']], axis=1)
        else:
            for col in X.columns:
                if len(X[col].value_counts()) > self.val_count: 
                    feat = pd.concat([feat, X[col]], axis=1)                    
        return feat

    
#------------------------------------------------------------
from sklearn.preprocessing import KBinsDiscretizer

class KBins(KBinsDiscretizer):
    
    """DataFrame Wrapper around KBinsDiscretizer. Sometimes this will throw 
    the monotonically increase/decrease error. You can either reduce bins 
    or modify the selected features by value counts (increase)"""

    def __init__(self, n_bins=5, encode='onehot', strategy='quantile'):
        super(KBins, self).__init__(n_bins=n_bins,
                                    encode='ordinal',
                                    strategy=strategy)                               
        
       
    def transform(self, X, y=None):
        
        assert isinstance(X, pd.DataFrame), "Must be pandas dframe"
        
        
        z = super(KBins, self).transform(X)
        binned = pd.DataFrame(z, index=X.index, columns=X.columns)
        binned = binned.applymap(lambda x: 'category_' + str(x))
#         final = pd.concat([X, binned], axis=1)        
        return binned

       
        
        

