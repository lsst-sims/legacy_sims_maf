__author__ = 'schandra'

from lsst.sims.operations.maf.db.Table import Table

class Database:
    def __init__(self, dbAddress, sessionID, tables=[['Config','configID'],
						     ['Session', 'sessionID'],
						     ['Proposal', 'propID'], 
						     ['Config_File', 'config_fileID'],
						     ['Log', 'logID'],
                                                     ['TimeHistory', 'timeHistID'], 
						     ['Cloud', 'cloudID'],
						     ['Seeing', 'seeingID'],
						     ['Proposal_Field', 'proposal_field_id'],
                                                     ['Field', 'fieldID'], 
						     ['SeqHistory', 'sequenceID'],
						     ['MissedHistory', 'missedHistID'],
						     ['ObsHistory', 'obsHistID'],
						     ['SeqHistory_MissedHistory', 'seqhistory_missedHistID'],
						     ['SeqHistory_ObsHistory', 'seqhistory_obsHistID'],
                                                     ['ObsHistory_Proposal', 'obsHistory_propID'],
						     ['SlewActivities', 'slewActivityID'],
						     ['SlewHistory', 'slewID'],
                                                     ['SlewMaxSpeeds', 'slewMaxSpeedID'],
						     ['SlewState', 'slewIniStatID']]):
        self.dbAddress = dbAddress
        self.sessionID = sessionID
        self.tableRegistry = {}
        for table in tables:
            print "Creating registry entry for [%s %s]" % (table[0], table[1])
            self.tableRegistry[table[0]] = Table(table[0], table[1], dbAddress)

    # ObsHistory Table functions
    def getObservations(self):
	constraint = 'Session_sessionID = %d' % self.sessionID
	return self.tableRegistry['ObsHistory'].query_columns(constraint = constraint)

    # ObsHistory_Proposal Table functions
    def getObservationsForPropID(self, propID):
	pass

    # SlewHistory Table functions
    def getSlews(self):
	constraint = 'ObsHistory_Session_sessionID = %d' % self.sessionID
	return self.tableRegistry['SlewHistory'].query_columns(constraint = constraint)

    def getSlewForObsHistID(self, obsHistID):
	constraint = 'ObsHistory_Session_sessionID = %d and ObsHistory_obsHistID = %d' % (self.sessionID, obsHistID)
	return self.tableRegistry['SlewHistory'].query_columns(constraint = constraint)

    # SlewState Table functions
    def getSlewStates(self):
        pass

    def getSlewStatesForSlewID(self, slewID):
        pass

    # SlewMaxSpeeds Table functions
    def getSlewMaxSpeeds(self):
        pass

    def getSlewMaxSpeedsForSlewID(self, slewID):
        pass

    # SlewActivities Table functions
    def getSlewActivities(self):
        pass

    def getSlewActivitiesForSlewID(self, slewID):
        pass

    # Proposal Table functions
    def getProposals(self):
        constraint = 'Session_sessionID = %d' % self.sessionID
	return self.tableRegistry['Proposal'].query_columns(constraint = constraint)

    # Session Table functions
    def getSession(self):
	constraint = 'sessionID = %d' % self.sessionID
        return self.tableRegistry['Session'].query_columns(constraint = constraint)

    # Config Table functions
    def getConfig(self):
        constraint = 'Session_sessionID = %d' % self.sessionID
	return self.tableRegistry['Config'].query_columns(constraint = constraint)

    # Config_File Table functions
    def getConfigFile(self):
        pass

    # Log Table functions
    def getLog(self):
        pass

    # TimeHistory Table functions
    def getTimeHistory(self):
	constraint = 'Session_sessionID = %d' % self.sessionID
	return self.tableRegistry['TimeHistory'].query_columns(constraint = constraint)

    # Proposal_Field Table functions
    def getFieldsForPropID(self, propID):
        pass

    # Field Table functions
    def getAllFields(self):
	return self.tableRegistry['Field'].query_columns()

    # MissedHistory functions
    def getMissedObservations(self):
        constraint = 'Session_sessionID = %d' % self.sessionID
	return self.tableRegistry['MissedHistory'].query_columns(constraint = constraint)

    # SeqHistory functions
    def getSequences(self):
        constraint = 'Session_sessionID = %d' % self.sessionID
        return self.tableRegistry['SeqHistory'].query_columns(constraint = constraint)

    def getObservationsForSeqID(self, seqID):
        pass

    def getMissedObservationsForSeqID(self, seqID):
        pass

    # Cloud
    def getCloudData(self):
        return self.tableRegistry['Cloud'].query_columns()

    # Seeing
    def getSeeingData(self):
	return self.tableRegistry['Seeing'].query_columns()


    def execute(self, tableName, colnames=None, chunk_size=None, constraint=None, numLimit=None):
	return self.tableRegistry[tableName].query_columns(colnames, chunk_size, constraint, numLimit)
