from lsst.sims.catalogs.generation.db.utils import loadData, make_engine
import sys, os, argparse
def getDbAddress():
    home_path = os.getenv("HOME")
    f=open("%s/dbLogin"%(home_path),"r")
    authDictionary = {}
    for l in f:
        els = l.rstrip().split()
        authDictionary[els[0]] = els[1]
    return authDictionary

if __name__ == "__main__":
    #Parse the command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("fileName", type=str, help="Path to file to load")
    parser.add_argument("tableName", type=str, help="Name of table in database")
    parser.add_argument("--delimiter", type=str, default=None, help="Delimiter to use in parsing the file -- "\
                                                                  "Default is white space")
    parser.add_argument("--primaryKey", type=str, default="obsHistId", help="Column to use as primary key --"\
                                                                            "Default is obsHistID")
    parser.add_argument("--numGuess", type=int, default=1000, help="Number of rows to use to guess dtype --"\
                                                                   "Default is 1000")
    parser.add_argument("--connectionName", type=str, default='MSSQL_MAF_WRITER', 
                       help="Key for the connection string to use in your dbLogin file -- "\
                            "Default is MSSQL_MAF_WRITER")
    args = parser.parse_args()
    authDictionary = getDbAddress()
    dbAddress = authDictionary['MSSQL_MAF_WRITER']
    engine, metaData = make_engine(dbAddress)
    loadData(args.fileName, None, args.delimiter, args.tableName, args.primaryKey, engine, metaData, args.numGuess)
