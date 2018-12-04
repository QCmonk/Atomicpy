# -*- coding: utf-8 -*-
# @Author: Spark
# @Date:   2018-02-02 13:32:25
# @Last Modified by:   Helios
# @Last Modified time: 2018-04-29 16:44:05

import os
import re
import sys
import h5py
import numpy as np
import pandas as pd 
from time import gmtime
import matplotlib.pyplot as plt

# path to experiment data file on monash or home machine
if os.path.isdir('C:\\Users\\joshm'):
    archivepath = 'C:\\Users\\joshm\\Documents\\Projects\\Research\\Archive\\FYP_archive.hdf5'
else:
    archivepath = 'C:\\Users\\Joshua\\Documents\\Projects\\Research\\Archive\\FYP_archive.hdf5'



#--------------------------------------------------------------------------
# COMPLETE
#--------------------------------------------------------------------------


# master function for storing FYP related data to a time coded h5 file
def data_flush(data, name, group="today", attrs=None, gtags=None, archivepath=archivepath):
	with h5py.File(archivepath, 'a') as archive: 

		# if no group specified add it to todays generic group
		if group == "today": 
			# generate todays group name
			date = gmtime()
			group = "{}_{}_{}".format(date.tm_mday, date.tm_mon, date.tm_year)
			try:
				# shortcut access to specified group
				group_short = archive.create_group(group)
				group_short.attrs["tags"] = ""
			except ValueError:
				# catch name already exists error 
				group_short = archive[group]
			
		# create group for todays work if not already done and add defaults
		elif group not in archive:
			group_short = archive.create_group(group)
			group_short.attrs["tags"] = ""

		# add tag attribute to todays group if desired
		if gtags is not None:
			# convert to list if not already one 
			if type(gtags) != list: gtags=[gtags]
			# retrieve current tags from group attributes
			current_tags = group_short.attrs["tags"]

			#check for repeated tags with case insensitivity
			tags = [new_tag for new_tag in gtags if re.match(r"{}".format(str(new_tag).lower()+","), current_tags.lower()) is None]

			if len(tags) > 0:
				updated_tags = current_tags + ",".join(map(str, tags)) +','
				group_short.attrs["tags"] = updated_tags
				
		dataset = group_short.create_dataset(name, data=np.asarray(data))
		# add any specified attributes to dataset
		if attrs != None:
			for attr,val in attrs:
				dataset.attrs[attr] = val


#--------------------------------------------------------------------------
# TO BE IMPLEMENTED
#--------------------------------------------------------------------------



#--------------------------------------------------------------------------
# CURRENTLY WORKING ON
#--------------------------------------------------------------------------



#--------------------------------------------------------------------------

# test data run
if __name__ == "__main__":
	data_flush(data=np.ones((25,25), dtype=np.float), group="testing", name="test_group/testing_attributes", attrs=[("test1", 1), ("test2", True)])
