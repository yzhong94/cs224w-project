This folder hosts all of our scripts to preprocessing data

Due to the size of the data files, we cannot host them on github.  Download the data from the links below and place them in an adjacent folder called "cs224w-data" for our processing scripts to run.

Final dataset used:
- bill data
	- https://www.govtrack.us/developers/data: has legislator name, state, congress term, date signed, district for matching to partywe, senate/house


Data considered for bill data:
- bill data
	- these two dataset do not show the names of the co-sponsors
		- 80th through 92 congress: http://www.congressionalbills.org/billfiles/bills80-92.zip
		- 93rd through 114th: http://www.congressionalbills.org/billfiles/bills93-114.zip
	- from james fowler: http://jhfowler.ucsd.edu/cosponsorship.htm
		- processed by james fowler: http://fowler.ucsd.edu/data/cosponsor.zip
			- only goes to 2004
		- raw from gov track: https://www.govtrack.us/developers/data
			-> use this data set, alreaady in link format (name under the same bill number are co-sponsors), has thomas_id
			- need party for the senators? maybe from the other data?
	- if we need more information on the legistlator: https://github.com/unitedstates/congress-legislators
- campaign data