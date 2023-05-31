import pandas as pd
import numpy as np

from abc import ABC, abstractmethod

from datetime import timedelta

from scipy import stats
from scipy import fftpack
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter, freqz, filtfilt

#from methods.constants import *

def get_operation(features, method, strategy_resampling, samplingrate_resampling, samplingrate_format, samplingrate_fft,
                  strategy_filtering, samplingrate_filtering, cutoff, lowcutoff, highcutoff, order, strategy_transformation,
                  window_size_transformation, strategy_sliding_window):

    if type(features) == str:
        features = [features]

    if method == 'None':
        operation = NoTransformation(features)
    if method == 'Resampling':
        operation = Resampling(features, strategy_resampling, samplingrate_resampling, samplingrate_format)
    elif method == 'FFT':
        operation = FFT(features, samplingrate_fft)
    elif method == 'Filtering':
        if strategy_filtering == 'band pass':
            operation = Filtering(features, strategy_filtering, lowcutoff, samplingrate_filtering, order, highcutoff)
        else:
            operation = Filtering(features, strategy_filtering, cutoff, samplingrate_filtering, order)
    elif method == 'Transformation':
        if strategy_transformation == 'shift':
            operation = Shifting(features, window_size_transformation)
        elif strategy_transformation == 'sliding window':
            operation = SlidingWindow(features, strategy_sliding_window, window_size_transformation)
        elif strategy_transformation == 'differencing':
            operation = Differencing(features, window_size_transformation)
        elif strategy_transformation == 'standardization':
            operation = Standardization(features)
        elif strategy_transformation == 'normalization':
            operation = Normalization(features)

    return operation

class Transformation(ABC):

    def __init__(self, features):
        if type(features) == str:
            features = [features]
        self.features = features

    @abstractmethod
    def apply(self, df):
        pass

    @abstractmethod
    def get_description(self):
        pass

class NoTransformation(Transformation):

    def __init__(self, features):
        super().__init__(features)
        self.on_complete_df = False
        self.type = 'None'

    def apply(self, df):
        return df

    def get_description(self):
        return "No Transformation"

class Deletion(Transformation):

    def __init__(self, features):
        super().__init__(features)
        self.on_complete_df = False
        self.type = 'Deletion'

    def apply(self, df):
        df = df.drop(columns=self.features)
        return df

    def get_description(self):
        return f"Remove {self.features[0]}"

class Resampling(Transformation):

    def __init__(self, features, strategy, samplingrate, samplingrate_format):
        super().__init__(features)
        self.strategy = strategy
        self.samplingrate = samplingrate
        self.samplingrate_format = samplingrate_format
        self.sr = {SAMPLINGRATE_OPTIONS_TS[self.samplingrate_format]: int(self.samplingrate)}
        self.on_complete_df = True
        self.type = 'Resampling'

    def apply(self, df):
        """
        Resamples the dataframe to a new samplingrate.
        """
        df = df[self.features]

        if str(df.index.dtype) == 'int64':
            grad = self.samplingrate * TIMESTAMP_TO_INDEX[SAMPLINGRATE_OPTIONS_TS[self.samplingrate_format]]
            data_slice = pd.DataFrame(index = np.arange(df.index[0], df.index[-1], grad))
        else:
            grad = timedelta(**self.sr)
            data_slice = pd.DataFrame(index = np.arange(df.index[0], df.index[-1], grad))

        tmp = []
        cols = df.columns

        for i in data_slice.index:
            if self.strategy == 'mean':
                tmp.append(df.loc[i: i + grad].mean().values)
            elif self.strategy == 'max':
                tmp.append(df.loc[i: i + grad].max().values)
            elif self.strategy == 'min':
                tmp.append(df.loc[i: i + grad].min().values)
            elif self.strategy == 'std':
                tmp.append(df.loc[i: i + grad].std().values)
            else:
                print(f'Unrecognized agg function: {self.strategy}, use mean instead')
                tmp.append(df.loc[i: i + grad].mean().values)


        df_resampled = pd.DataFrame(tmp, index=data_slice.index)
        df_resampled.columns = cols

        return df_resampled

    def get_description(self):
        return f"{self.strategy.capitalize()} resampling with samplingrate={self.samplingrate} {self.samplingrate_format.capitalize()}"



class FFT(Transformation):

    def __init__(self, features, samplingrate):
        super().__init__(features)
        self.samplingrate = samplingrate
        self.on_complete_df = True
        self.type = 'FFT'

    def apply(self, df):
        """
        Applies the fast fourier transformation on the dataframe.
        """
        df = df[self.features]
        df_fft = pd.DataFrame()

        for i in self.features:
            col = df[i].values
            Fs = int(self.samplingrate)
            t = np.arange(0,1,1/Fs)

            # Perform Fourier transform
            y_fft = fft(col)

            # plot results
            n = np.size(t)
            fr = Fs/2 * np.linspace(0,1,n//2)
            y_m = 2/n * abs(y_fft[0:np.size(fr)])

            if i == 0:
                df_fft.index = fr
            df_fft[i] = y_m

        return df_fft

    def get_description(self):
        return f"FFT with samplingrate={self.samplingrate}"

class Filtering(Transformation):

    def __init__(self, features, strategy, cutoff, samplingrate, order, cutoff_2=None):
        super().__init__(features)
        self.strategy = strategy
        if self.strategy == 'low pass':
            self.btype = 'low'
            self.cutoff = cutoff
        elif self.strategy == 'high pass':
            self.btype = 'high'
            self.cutoff = cutoff
        else:
            self.btype = 'band'
            self.cutoff = (cutoff, cutoff_2)

        self.fs = samplingrate
        self.order = order
        self.on_complete_df = False
        self.type = 'Filtering'


    def apply(self, df):
        df_filtered = df.copy()
        for i in self.features:
            if self.btype == 'band':
                lowcutoff, highcutoff = self.cutoff
                nyq = 0.5 * self.fs
                b, a = butter(self.order, [lowcutoff,highcutoff], fs=self.fs, btype='band', analog=False)
                y = filtfilt(b, a, df[i])
            else:
                nyq = 0.5 * self.fs
                normal_cutoff = self.cutoff / nyq
                # Get the filter coefficients
                b, a = butter(self.order, normal_cutoff, btype=self.btype, analog=False)
                y = filtfilt(b, a, df[i])
            df_filtered[i] = y
        return df

    def get_description(self):
        if self.btype == 'band':
            return f"{self.strategy.capitalize()} filtering with samplingrate={self.fs}, order={self.order}, low cutoff={self.cutoff[0]} and high cutoff={self.cutoff[1]}"
        else:
            return f"{self.strategy.capitalize()} filtering with samplingrate={self.fs}, order={self.order} and cutoff={self.cutoff}"

class Shifting(Transformation):

    def __init__(self, features, periods):
        super().__init__(features)
        self.periods = periods
        self.on_complete_df = False
        self.type = 'Transformation'

    def apply(self, df):
        shifted_df = df.copy()
        for i in self.features:
            for j in range(1, self.periods+1):
                shifted_df[f'{i}_t_{j}'] = df[i].shift(j)

        shifted_df = shifted_df.fillna(method="bfill")

        return shifted_df

    def get_description(self):
        return f"Shifting with window size={self.periods}"

class SlidingWindow(Transformation):

    def __init__(self, features, strategy, periods):
        super().__init__(features)
        self.strategy = strategy
        self.periods = periods
        self.on_complete_df = False
        self.type = 'Transformation'

    def apply(self, df):
        slided_df = df.copy()

        for i in self.features:
            if self.strategy == 'sum':
                slided_df[i] = df[i].rolling(window=self.periods).sum()
            elif self.strategy == 'mean':
                slided_df[i] = df[i].rolling(window=self.periods).mean()
            elif self.strategy == 'min':
                slided_df[i] = df[i].rolling(window=self.periods).min()
            elif self.strategy == 'max':
                slided_df[i] = df[i].rolling(window=self.periods).max()
            elif self.strategy == 'std':
                slided_df[i] = df[i].rolling(window=self.periods).std()
            else:
                print(f'Unrecognized strategy function: {strategy}, use mean instead')
                slided_df[i] = df[i].rolling(window=self.periods).mean()
        return slided_df

    def get_description(self):
        return f"{self.strategy.capitalize()} sliding window with window size={self.periods}"

class Differencing(Transformation):

    def __init__(self, features, periods):
        super().__init__(features)
        self.periods = periods
        self.on_complete_df = False
        self.type = 'Transformation'

    def apply(self, df):
        diff_df = df.copy()

        for i in self.features:
            diff_df[i] = df[i].diff(periods=self.periods)

        diff_df = diff_df.fillna(method="bfill")

        return diff_df

    def get_description(self):
        return f"Differencing with window size={self.periods}"

class Standardization(Transformation):

    def __init__(self, features):
        super().__init__(features)
        self.on_complete_df = False
        self.type = 'Transformation'

    def apply(self, df):
        std_df = df.copy()
        for i in self.features:
            std_df[i] = (df[i] - df[i].mean()) / df[i].std()

        return std_df

    def get_description(self):
        return "Standardization"

class Normalization(Transformation):

    def __init__(self, features):
        super().__init__(features)
        self.on_complete_df = False
        self.type = 'Transformation'

    def apply(self, df):
        norm_df = df.copy()
        for i in self.features:
            norm_df[i] = (df[i] - df[i].min()) / (df[i].max() - df[i].min())

        return norm_df

    def get_description(self):
        return "Normalization"
