from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import os
import sys
import shutil
import time
import datetime


def main():
    path = "./data/discharge_data_100/"
    mode = sys.argv[1]
    method = sys.argv[2]
    alpha = float(sys.argv[3])
    run = bfi_analysis(path, elements=0, parse=mode, method=method, alpha=alpha, delete_all=True)
    run.run()

    return 0


class bfi_analysis():
    def __init__(self, folder_path, elements=0, filetype=".q", parse="Base", method="lh", alpha=0.925, delete_all=False, nan_ident=-9999, reflect=30, missing_data=19) -> None:
        self.mode_of_parsing: str = parse                                       # Whether we want to slice the raw data creating unparsed data
        valid = False
        if parse == "Clean" or parse == "Base":
            valid = True
        if not valid:
            exit("No valid program...")

        if method == "lh":
            self.method = self._baseflow_separation_LH
            self.folder_addition = method + "_" + str(alpha)
        elif method == "wf":
            self.method = self._baseflow_separation_wallingford
            self.folder_addition = method + "_" + str(alpha)

        pm = "/home/sigurd/BScProject/GEO3000/code/data/discharge_data_100/HbvFeltInfo_100.txt"  # Hardcoded path

        pd.options.mode.chained_assignment = None                               # Supress irrelevant warning

        self.df_output: pd.DataFrame                                            # Output dataframe initialized in memory
        self.folder_path: str = folder_path                                     # Relative path or absolute path to folder with data
        self.folder_path_final: str = "/".join(folder_path.split("/")[0:2]) + "/"  # Path used for final files with BFI
        self.folder_list_data: List[str]                                        # List of files in folder containing the data, based on file_type
        self.folder_list_supplementary: List[str]                               # All other files, used for meta data
        self.file_type: str = filetype                                          # File ending, denoting filetype
        self.elements: int = elements                                           # Number of items in folder to be read, chooses the first n elements, defaults to entire list
        self.delete_all = delete_all
        self.nan_ident = nan_ident                                              # NaN identifier for the dataset

        self.reflect = reflect                                                  # Length to reflect based on Ladson
        self.alpha = alpha
        self.missing_data = missing_data                                        # Number of days missing in a year for the year to be considered biased and unsuable

        self._create_metadata_df(pm)                                            # Metadata, but hardcoded
        self._mean_filename = "station_means.txt"

    def routine_1(self) -> None:
        """Used on the initial dataset and creates the clean folder and files. These files have only been sliced into hydrological years
        and contain information about the nan values in the files. No actual math is done here.
        """
        self.df_input: pd.DataFrame                             # Input dataframe initialized in memory
        start_time = time.time()

        c = 1
        for file in self.folder_list_data:
            print("Starting working on " + self.folder_path + file + " Current count is: " + str(c) + "\n")
            c += 1

            if not self.delete_all:
                os.remove(self.folder_path_final + file)
            self._read_file_raw(self.folder_path + file)        # Read the file and write it to df_input
            self._slice_dataframe()                             # We no longer care for the input frame, and work exclusively with the output frame
            nan_data = self._nanindex()
            self._write_to_file(file, aux_data=nan_data)

        print("Total time used: " + str(datetime.timedelta(seconds=time.time() - start_time)))

    def routine_2(self) -> None:
        """Works on the clean datafiles to calculate the BFI and create a dataframe and accompanying file which includes all metadata and more.
        """
        self.columns = ["File", "Name", "Period", "Area", "BFI", "Completness", "Datum", "Long", "Lat"]
        df = pd.DataFrame(columns=self.columns)                             # Empty dataframe

        start_time = time.time()

        c = 1
        for file in self.folder_list_data:                                  # Loops through the list of files
            nan_list = self._open_cleaned_file(file)                        # Open the file - dataframe and list of nan locatios
            print("Starting working on " + self.folder_path + file + " Current count is: " + str(c) + "\n")

            if not self.delete_all:
                os.remove(self.folder_path_final + file)

            self._add_hydyear2dataframe()                                            # Adds a column with the hydrological year at index 0

            self.df_output["Qb"] = pd.Series(dtype=np.float64)              # Adds a empty column for the baseflow

            nan_list = self._merge_nan_list(nan_list)                  # Congregate the nan list to exclude series shorter than self.reflect

            self._segment_df_baseflow_calc(nan_list)                     # Segments the dataframe and does the baseflow seperation
            self.df_output_copy = self.df_output.copy()                     # Create a copy to prevent overwriting

            self._write_to_file_separated(file)

            self.df_output = self.df_output.groupby("WY").sum()             # Group by year
            self.df_output["BFI"] = self.df_output["Qb"] / self.df_output["discharge"]  # Calculate yearly BFI

            aux_data = 0
            # Handle if it includes nan vlues.

            if len(nan_list) != 0:
                aux_data = self._missing_data_info(nan_list)
                self._weighted_segments()
            else:
                aux_data = (0, 0, 0)
            c += 1

            self.df_output[self.df_output <= 0] = np.nan
            self._write_to_file(file, aux_data)                                 # Write the new dataframe to a file [index[year, month, date], WY, total flow, baseflow]

            self.df_output.drop(labels=["Qb", "discharge", "WY"], axis=1, inplace=True, errors="ignore")

            self.df_output.dropna(inplace=True)

            period = (self.df_output.index[0], self.df_output.index[-1])
            bfi = self.df_output.mean().values[0]
            df_aux = self._df_aux(bfi, period, aux_data, file)
            df = pd.concat([df, df_aux], ignore_index=False)

        print("Total time used: " + str(datetime.timedelta(seconds=time.time() - start_time)))
        df.set_index("File", inplace=True)
        self._write_to_file_mean(df)                                       # Write a mean value dataframe with the accompanying data seen in self.columns

    def run(self) -> None:
        if self.mode_of_parsing == "Clean":
            self.folder_path_final = self.folder_path[:-1] + "_cleaned/"

        if self.mode_of_parsing == "Base":
            self.folder_path_sep = self.folder_path_final + self.folder_addition + "_sep/"
            self.folder_path = self.folder_path[:-1] + "_cleaned/"
            self.folder_path_final = self.folder_path_final + self.folder_addition + "_final/"

        if self.delete_all:
            self._folder_management(self.folder_path_sep)
            self._folder_management(self.folder_path_final)
        self._get_list_of_files_from_folder(self.folder_path)               # Get the list of files in the wanted folder

        if self.elements != 0:
            self.folder_list_data = self.folder_list_data[:self.elements]  # Only do for x elements, used in testing

        if self.mode_of_parsing == "Clean":
            self.routine_1()
        if self.mode_of_parsing == "Base":
            self.routine_2()

    def _folder_management(self, folder_path) -> None:
        """Creates a new folder if it does not exist else it deletes the folder and all subfolders and recreates it.

        Args:
            folder_path (str): path to wanted folder
        """
        if self.delete_all:
            if os.path.exists(folder_path):  # Does the folder exist
                shutil.rmtree(folder_path)  # If yes, delete
                os.mkdir(folder_path)       # And create new
            else:
                os.mkdir(folder_path)       # Else, create new

    def _get_list_of_files_from_folder(self, folder_path) -> None:
        """Gets a list of files from a folder

        Args:
            folder_path (str): from which folder to get files
        """
        self.folder_list_data: List[str] = []
        self.folder_list_supplementary: List[str] = []

        folder: os.DirEntry = os.scandir(folder_path)               # Reads the directory
        for file in folder:                                         # Loop over files
            if file.name[-len(self.file_type):] == self.file_type:  # Checks for files of filetype
                self.folder_list_data.append(file.name)             # Adds filenames with correct filetype to list
            else:
                self.folder_list_supplementary.append(file.name)    # Add all other files to a supplementary list
        folder.close()                                              # Close the folder to prevent mishaps

    def _read_file_raw(self, path_to_file) -> None:
        """Reads raw datafiles on a csv format with delim set to whitespace.

        Args:
            path_to_file (str): str to the file to read
        """
        self.df_input = pd.read_csv(path_to_file, delim_whitespace=True, parse_dates=[0], names=["date", "discharge"])  # Creates a dataframe with space as delim

    def _open_cleaned_file(self, file):
        """Reads the cleaned file. Adds it to the output dataframe we are working with form here on .
        Returns the list of the leading nan values in the list. Nan_list may be empty

        Args:
            file (str): file name

        Returns:
            List: List over nan values
        """
        nan_list = []
        file_object = open(self.folder_path + file, "r")
        file_object.readline()
        for line in file_object:
            if line[:2] == "#*":
                self.df_output = pd.read_csv(file_object, delim_whitespace=True, header=0, index_col=[0, 1, 2])
                break
            nan_list.append(eval(line[2:]))

        file_object.close()
        return nan_list

    def _create_metadata_df(self, path):
        """Called by one function, and creates a metadata dataframe based on a specific .txt file
        """
        self.df_metadata = pd.read_csv(path, delim_whitespace=True, header=0, index_col=[0, 1])

    def _slice_dataframe(self) -> None:
        """Slices the head and tails of a dataframe such that the start and end are within the hydrological year.
        If first entry is 01.01.xxxx and last entry is given as 31.12.xxxx + n, so that the entire dataset spans more than one calender year
        it returns a dataframe in which the range goes from 01.09.xxxx to 31.08.xxxx + n where both the start and end values of the dataset is non nan values
        """
        self.df_input["day"] = self.df_input["date"].dt.day
        self.df_input["month"] = self.df_input["date"].dt.month
        self.df_input["year"] = self.df_input["date"].dt.year
        indexes = list(zip(self.df_input["year"], self.df_input["month"], self.df_input["day"]))
        index = pd.MultiIndex.from_tuples(indexes, names=["year", "month", "day"])
        self.df_output: pd.DataFrame = pd.DataFrame(self.df_input["discharge"], copy=True)
        self.df_output.set_index(index, inplace=True)

        hyearstart = (self.df_output.iloc[0].name[0], 9, 1)                     # Start of the hydrological year in Norway
        hyearend = (self.df_output.iloc[-1].name[0], 8, 31)                     # End of the hydrological year in Norway
        self.df_output = self.df_output.loc[hyearstart:hyearend]                # Slice the df to fit within the hydrological years

        while True:                                                             # This will from testing always break at one point
            start_year = self.df_output.iloc[0].name[0]                         # Find the start year of the new data
            end_year = self.df_output.iloc[-1].name[0]                          # End year
            slice_head = self.df_output.loc[(start_year, 9, 1)].values == -9999  # Check if it needs to be cut
            slice_tail = self.df_output.loc[(end_year, 8, 31)].values == -9999  # Check if it needs to be cut

            start_changed = True
            end_changed = True
            if slice_head:
                hyearstart = (start_year + 1, 9, 1)                             # If head is wrong, try next year
                start_changed = False
            if slice_tail:
                hyearend = (end_year - 1, 8, 31)                                # If tail is wrong try previous year
                end_changed = False
            self.df_output = self.df_output.loc[hyearstart:hyearend]            # Slice
            end_loop = start_changed and end_changed
            if end_loop:
                break                                                           # End the While loop if valid

    def _segment_df_baseflow_calc(self, nan_list):
        """Seperates the dataframe into segments for which to separate baseflow from total flow.

        Args:
            nan_list (List): List of tuples of nan placements, output of the self._merge_nan_list(x)
        """
        # Start of loop
        start = 0
        for tup in nan_list:

            input_array = self.df_output["discharge"].iloc[start:tup[0]].to_numpy()        # creates the input array of valid length
            baseflow = self.method(input_array, alpha=self.alpha)                              # finds the baseflow seperation from said input array
            self.df_output["Qb"].iloc[start:tup[0]] = baseflow                             # Adds baseflow to the dataframe

            start = tup[1]
        # End of loop
        # Handles the tail of the dataframe, this should always be valid, but this makes sure it is valid.
        # If not valid set end to nan values.
        tail = len(self.df_output["discharge"].iloc[start:]) > self.reflect
        if tail:                                                                           # Fixes the tail if it exists.
            input_array = self.df_output["discharge"].iloc[start:].to_numpy()              # Creates the final array and handles that
            baseflow = self.method(input_array, alpha=self.alpha)                              # finds the baseflow seperation from said input array
            self.df_output["Qb"].iloc[start:] = baseflow                                   # Adds baseflow to the dataframe
        else:
            self.df_output["Qb"].iloc[start:] = np.nan(len(self.df_output["Qb"].iloc[start:]), np.nan)

    def _baseflow_separation_LH(self, q, alpha=0.925):
        """Calculates the baseflow for a set dataframe using the Lyne-Hollick method.
        This function is based on the implementation recommended by Tony Ladson (2013) and his accompanying R-implementation avaiable at
        - DOI:       10.7158/W12-028.2013.17.1.
        - Code:      https://github.com/TonyLadson/BaseflowSeparation_LyneHollick
        - Blogpost:  https://tonyladson.wordpress.com/2013/10/01/a-standard-approach-to-baseflow-separation-using-the-lyne-and-hollick-filter/

        Date for implementation: 2022/02/23
        Crossreferenced last:    2022/02/23

        Freedoms have been taken in how data is handled and worked with as a different langauge is used.
        The essence of the recommended implementation is followed, while not all functions may be working.
        Some unnecessary implementation is done to stay true to the original R-implementation used as base.

        The input array is the array for which the baseflow needs to be calculated, sequencing and slicing happens prior to this.

        Args:
            q (np.ndarray): Input streamflow data in the form of a numpy array
            alpha (float, optional): Digital filter parameter based on the local and/or literature values for this parameter. Defaults to 0.925.
            reflect (int, optional): Amount of values to reflect. Defaults to recommended value for daily streamflow data. Defaults to 30.

        Returns:
            Tuple(np.ndarray): baseflow values for the streamflow.
        """
        # TODO: Look for simplifications if there is no need to save values they can be removed.

        # Expecting q to be a vector from numpy
        if not isinstance(q, np.ndarray):
            print("q must be a np.ndarray, it is ", type(q))
            exit()

        if self.reflect >= len(q):
            print("Data set must be longer than the reflect period. Exiting...")
            exit()

        if alpha < 0 or alpha >= 1:
            print("alpha must be between 0 and 1. Exiting...")
            exit()

        def first_pass(q, a) -> pd.DataFrame:
            """Firsts forward pass of the dataframe, needs to be handled slightly differently than the later forward pass, due to being implemented as a dataframe.
            Choice of this was to avoid indexing issues when implementing.

            Args:
                q (np.ndarray): streamflow values
                a (float): digital filter parameter

            Returns:
                pd.DataFrame: Dataframe with columns ["qf", "qb"], which are the seperated quickflow and baseflow for use in the following passes
            """
            b = 0.5 * (1 + a)
            qf = np.zeros(len(q))  # Empty quickflow
            qf[0] = q[0]
            for i in range(1, len(qf)):
                qf[i] = a * qf[i - 1] + b * (q[i] - q[i - 1])

            qb1 = np.where(qf > 0, q - qf, q)

            return pd.DataFrame({"qf": qf, "qb": qb1})

        def backwards_pass(q, a) -> pd.DataFrame:
            """Backwards pass in the Lyne-Hollick method

            Args:
                q (pd.DataFrame): Dataframe with columns ["qf", "qb"], which are the seperated quickflow and baseflow for use in the following passes
                a (float): digital filter parameter

            Returns:
                pd.DataFrame: Dataframe with columns ["qf", "qb"], which are the seperated quickflow and baseflow for use in the following passes
            """
            n = len(q["qb"])
            qb = q["qb"]
            b = 0.5 * (1 + a)

            qf = np.zeros(n)  # Empty array
            qf[-1] = qb.iloc[-1]

            for i in range(n - 2, 0, -1):
                qf[i] = a * qf[i + 1] + b * (qb.iloc[i] - qb.iloc[i + 1])

            qb2 = np.where(qf > 0, qb - qf, qb)

            return pd.DataFrame({"qf": qf, "qb": qb2})

        def forward_pass(q, a) -> pd.DataFrame:
            """Forward pass, similar to the first pass, but uses dataframes instead

            Args:
                q (pd.DataFrame): Dataframe with columns ["qf", "qb"], which are the seperated quickflow and baseflow for use in the following passes
                a (float): digital filter parameter

            Returns:
                pd.DataFrame: Dataframe with columns ["qf", "qb"], which are the seperated quickflow and baseflow for use in the following passes
            """
            n = len(q["qb"])
            qb = q["qb"]
            b = 0.5 * (1 + a)

            qf = np.zeros(n)  # Empty array
            qf[0] = qb.iloc[0]

            for i in range(1, n):
                qf[i] = a * qf[i - 1] + b * (qb.iloc[i] - qb.iloc[i - 1])

            qb2 = np.where(qf > 0, qb - qf, qb)

            return pd.DataFrame({"qf": qf, "qb": qb2})

        q_in = np.pad(q, (self.reflect, self.reflect), mode="reflect")  # Pad the dataset

        # First pass always needed
        df_tmp = first_pass(q_in, alpha)

        df_tmp = backwards_pass(df_tmp, alpha)

        df_tmp = forward_pass(df_tmp, alpha)

        qb = df_tmp["qb"][self.reflect:-self.reflect].to_numpy()

        qb[qb < 0] = 0                  # Set values less than zero to zero

        return qb

    def _baseflow_separation_wallingford(self, q, alpha=0.9):
        """Implementation of the Wallingford method, based on the Institute of Hydrology paper 1980 low flow studies
        Date of implementation: 20.03.2022

        Args:
            q (np.ndarray): input streamflow data
            alpha (float, optional): consistency with the LH method. Defaults to 0.9.

        Returns:
            np.ndarray: separated baseflow
        """

        # ************************************************
        # This sets the number of segments to be a multiple of 5
        n = len(q)
        if n % 5 != 0:
            nm = n - n % 5
        else:
            nm = n
        # ************************************************
        # Finds all minimas in the 5 day non-overlapping chunks
        qb = np.zeros(n)

        minimas = []
        c = 0
        while c < nm:
            x = q[c:c + 5]
            idxm = x.argmin()
            minimas.append((idxm + c, x[idxm]))
            c += 5
        # ************************************************
        # 0.9 times the central values
        # [[x,y],[x,y],[x,y]]
        c = 1
        t = []
        while c < len(minimas) - 1:
            if minimas[c][1] * alpha <= np.min((minimas[c - 1][1], minimas[c + 1][1])):
                t.append(minimas[c])

            c += 1
        # ************************************************
        # Draws the lines between two consecutive turning points
        ip = 0
        jp = t[0][1]  # q[0]
        for i, j in t:

            l = i - ip
            x = np.linspace(0, l - 1, l)
            a = (j - jp) / (i - ip)
            def f(x): return a * x + jp

            qb[ip:i] = f(x)
            ip = i
            jp = j

        qb[ip:] = jp
        # ************************************************
        # Puts the limit that baseflow can not be higher than streamflow
        qb = np.where(qb > q, q, qb)
        return qb

    def _nanindex(self) -> List[Tuple]:
        """Works on the current dataframe
        Retunrs as follows: # Start_Nan End_Nan Length_Nan Length_NoNan
        Start_Nan is the index value of the first nan values encountered
        End_Nan is the first non nan value after
        Length_Nan is the length of the sequence of Nan values
        Length_NoNan is the prior length of no nan values

        Returns:
            List[Tuple]: Tuple of the above property
        """
        data = self.df_output["discharge"].to_numpy()
        l = []
        i = 0
        vq = 0
        while i < len(data):
            val = data[i]
            if val == self.nan_ident:
                v1 = i
                for j, val in enumerate(data[i:]):
                    if val != self.nan_ident:
                        v2 = i + j
                        break
                l.append((v1, v2, v2 - v1, v1 - vq))
                vq = v2

                i += j
            i += 1
        return l

    def _merge_nan_list(self, nan_list):
        """Congregates the nan_lists.
        In most cases this functions does nothing.
        If Length_NoNan is shorter than reflect then it will congregate the sequence and essentially treat the data even though present as missing.
        So if Length_NoNan < reflect then we ignore the entire non nan segment and following calculations.

        Potential issue with this is if there is a single nan followed by n<reflect non-nans with a following nan. Where an entire segment is thrown when not needed.
        To counter his an interpolation may be used on the single nan or few nans to have a better dataset.

        Args:
            nan_list (List): List of nan values generated earlier by the self._nanindex function

        Returns:
            List: New list of nan tuples that are congregated.
        """
        # Start_Nan End_Nan Length_Nan Length_NoNan

        if len(nan_list) == 0:
            return nan_list

        start_index = []
        prev_nonan = []
        new_nan_list = []

        sequence = False

        for nan_indicies in nan_list:
            no_nan = nan_indicies[3]
            if sequence and no_nan > self.reflect:
                new_tuple = (start_index[0], start_index[-1], start_index[-1] - start_index[0], prev_nonan[0])
                new_nan_list.append(new_tuple)

                start_index = []
                prev_nonan = []
                sequence = False
            start_index.append(nan_indicies[0])
            start_index.append(nan_indicies[1])

            prev_nonan.append(no_nan)
            if no_nan < self.reflect:
                continue
            else:
                sequence = True

        if nan_indicies == nan_list[-1]:
            new_tuple = (start_index[0], start_index[-1], start_index[-1] - start_index[0], prev_nonan[0])
            new_nan_list.append(new_tuple)
        return new_nan_list

    def _range_of_nan(self, x):
        """
        Takes the array of the baseflow and returns the indexes of starting and ending nan values
        """
        x = np.hstack([[False], x, [False]])  # padding
        d = np.diff(x.astype(int))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        return starts, ends

    def _list_nan_data(self, inp) -> str:
        """Converts nan data to a list, to prevent errors later in the program flow

        Args:
            inp (List): input data to convert to list

        Returns:
            List: List with a header or only header if empty list
        """
        if isinstance(inp, Tuple):
            inp = [inp]
            inp.insert(0, "Completness MissingValues TotalValues")  # Header for human readability
            return inp

        inp.insert(0, "Start_Nan End_Nan Length_Nan Length_NoNan")  # Header for human readability
        if not isinstance(inp, list):                         # Check if it is a list, if not make it
            inp = [inp]
        return inp

    def _missing_data_info(self, nan_list) -> Tuple[float, float, float]:
        """Gives some vlues for completness of the data, such as missing values and %.

        Args:
            nan_list (List): list of nan values in the dataframe

        Returns:
            Tuple[float, float, float]: completness data, containing the %, missing and total values
        """
        s = 0
        for tup in nan_list:
            # Start_Nan End_Nan Length_Nan Length_NoNan
            s += tup[2]
        r2 = len(self.df_output_copy.index)
        r1 = s / r2
        return (r1, s, r2)

    def _weighted_avg(self, x, y, starts, ends):
        """
        Computes the weighted avg of the baseflow based on the starts and ends of the nan vlaues

        Two issues, if the first nan value is 0.
        And the end of the list is non-zero.
        Above is fixed.
        """
        n1 = 0
        w = []
        calc = []
        for i, j in zip(starts, ends):
            if i == 0:
                w.append(i - n1)
                calc.append(0)
                n1 = j
            else:
                w.append(i - n1)
                calc.append(sum(x[n1:i]) / sum(y[n1:i]))
                n1 = j

        if n1 != len(x):
            w.append(len(x[n1:]))
            calc.append(sum(x[n1:]) / sum(y[n1:]))

        qb = 0
        for i in range(len(w)):
            qb += calc[i] * w[i]
        bfi = qb / sum(w)
        return bfi

    def _weighted_segments(self) -> None:
        """
        Calculates the weighted segments based on the years which are missing values. This also sets values taht should be missing to np.nan.

        """
        df_tmp: pd.DataFrame = self.df_output_copy.copy()                    # Need to calculate the BFI values for the years in which nan values are present.

        df_tmp["nn"] = df_tmp["Qb"].isnull()                            # Find nan locations in the entire series of the baseflow values

        df_grouped = df_tmp.groupby("WY")["nn"].sum()                   # Group the baseflow nans so it is per years

        reqd_index_drop = df_grouped[df_grouped >= self.missing_data].index.tolist()   # Find all years where more than 5% of the baseflow values are missing. Change 19 if more values are acceptable

        reqd_index = df_grouped[df_grouped > 0].index.tolist()          # Find all years where there is more than 1 missing value

        for element in reqd_index_drop:                                 # Removes overlap between the two sets of years
            if element in reqd_index:
                reqd_index.remove(element)

        for year in reqd_index:                                         # Loop over all years that are of interest
            tmp = df_tmp[df_tmp["WY"] == year]                          # Extract that specific year
            tmp_n = tmp["nn"].to_numpy()                                # Convert to numpy array
            starts, ends = self._range_of_nan(tmp_n)                        # Computes the starts and ends of the nan values
            wa = self._weighted_avg(tmp["Qb"], tmp["discharge"], starts, ends)  # Computes the weighted avg
            self.df_output["BFI"].loc[year] = wa

        for year in reqd_index_drop:
            self.df_output["BFI"].loc[year] = np.nan

    def _add_hydyear2dataframe(self):
        """Adds the hydrological year to the dataframe as a new column and sets it as column number 1, index 0
        """
        def assign_wy(row):
            if row.name[1] > 8:
                return(int(row.name[0] + 1))
            else:
                return(int(row.name[0]))
        self.df_output['WY'] = self.df_output.apply(lambda x: assign_wy(x), axis=1)     # Adds column
        self.df_output = self.df_output[["WY", "discharge"]]                             # Sorts order

    def _df_aux(self, bfi, period, completness, file):
        """Creates the auxilliary dataframe for the means values

        Args:
            bfi (float): bfi
            period (Tuple[float,float]): the period of the dataframe
            completness (Tuple[float, float, float]): the completness of missingdata
            file (str): filename

        Returns:
            pd.DataFrame: dataframe of intereset
        """
        name, area, datum, long, lat = self._metadata(file)
        d = {
            "File": file,
            "Name": name,
            "Period": str(period),
            "Area": area,
            "BFI": bfi,
            "Completness": str(completness),
            "Datum": datum,
            "Long": long,
            "Lat": lat
        }
        return pd.DataFrame(d, index=[0])

    def _metadata(self, file):
        """Creates a metadata string, which works on the metadata dataframe. The

        Args:
            file (str): file to which it makes the metadata

        Returns:
            str: single string of meta data
        """
        reg, main = file[:-2].split(".")  # ["1", "2"]              # Get reg and main numbers from the file name
        meta_data = self.df_metadata.loc[(int(reg), int(main))]     # Grab the relevant data
        meta_data = meta_data.apply(str)                            # Convert it to string for later use

        # This code may be useful later,so keep
        """
        # index = meta_data.index                                     # Grab the index's
        # values = meta_data.values                                   # Grab the values
        # temp_list = ["regno " + reg, "mainno " + main]              # Create the temporary list with reg and main in them
        # for x, y in zip(index, values):
        #     temp_list.append(" ".join([x, y]))                      # Add all the meta data

        # supp_data = []  # Dummy variable

        # if not isinstance(supp_data, list):                         # Check if it is a list, if not make it
        #     supp_data = [supp_data]

        # for ele in supp_data:
        #     temp_list.append(str(ele))                              # Append the wanted metadata

        # return " ".join(temp_list)                                  # Return as a string
        """

        return meta_data[["Name", "Area", "UTM", "East", "North"]]

    def _write_to_file(self, file, aux_data=0):
        """Simple write to file function, which includes some extra data that needs to be written to file. 

        Args:
            file (str): path to the file save location
            aux_data (List, optional): List of values to add to the file at the top. Defaults to 0
            print("Starting working on " + self.folder_path + file + " Current count is: " + str(c) + "\n")
        """
        if aux_data != 0:
            list_supp_data = self._list_nan_data(aux_data)

        path_clean = self.folder_path_final + file             # Create the final path for the files

        print("Clean path " + path_clean)
        with open(path_clean, "w") as o:                              # Write metadata and seperator to the file
            if aux_data != 0:
                for line in list_supp_data:
                    o.write("# " + str(line) + "\n")
            o.write("#" + "*" * 100 + "\n")
        self.df_output.to_csv(path_clean, sep=" ", mode="a")          # Append the data

    def _write_to_file_separated(self, file):
        """Writes to inital seperated datafiles, which gives full baseflow and discharge.

        Args:
            file (str): filename
        """
        self.df_output.to_csv(self.folder_path_sep + file, sep=" ", mode="w")

    def _write_to_file_mean(self, df) -> None:
        """writes the means of all dataseries to a file, this includes the long term values.
         uses a pre specified filename

        Args:
            df (pd.DataFrame): dataframe which to write to file
        """
        df.columns = ["Name", "Period", "Area", "BFI", "Completness", "Datum", "East", "North"]
        df.to_csv(self.folder_path_final + self._mean_filename, sep=" ")


if __name__ == "__main__":
    main()
