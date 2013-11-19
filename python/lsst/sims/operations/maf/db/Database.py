__author__ = 'schandra'

import lsst.sims.catalogs.generation.db import DBOject, ChunkIterator
import numpy
from sqlalchemy import func
from sqlalchemy.sql import expression

class Database:

    def __init__(self, connString, username, password, sessionID):
        pass

    # ObsHistory Table functions
    def getObservations(self):
        pass

    # ObsHistory_Proposal Table functions
    def getObservationsForPropID(self, propID):
        pass

    # SlewHistory Table functions
    def getSlews(self):
        pass

    def getSlewForObsHistID(self, obsHistID):
        pass

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
        pass

    # Session Table functions
    def getSession(self):
        pass

    # Config Table functions
    def getConfig(self):
        pass

    # Config_File Table functions
    def getConfigFile(self):
        pass

    # Log Table functions
    def getLog(self):
        pass

    # TimeHistory Table functions
    def getTimeHistory(self):
        pass

    # Proposal_Field Table functions
    def getFieldsForPropID(self, propID):
        pass

    # Field Table functions
    def getAllFields(self):
        pass

    # MissedHistory functions
    def getMissedObservations(self):
        pass

    # SeqHistory functions
    def getSequences(self):
        pass

    def getObservationsForSeqID(self, seqID):
        pass

    def getMissedObservationsForSeqID(self, seqID):
        pass

    # Cloud
    def getCloudData(self):
        pass

    # Seeing
    def getSeeingData(self):
        pass

    # Execute arbitrary SQL
    def executeSQL(self, sql):
        pass

    # Disconnect from DB
    def disconnect(self):
        pass