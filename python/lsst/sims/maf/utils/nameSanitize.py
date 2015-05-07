
__all__ = ['nameSanitize']

def nameSanitize(inString):
    """
    Convert a string to a more file name friendly format
    """
    # Replace <, > and = signs.
    outString = inString.replace('>', 'gt').replace('<', 'lt').replace('=', 'eq')

    # Remove single-spaces, strip '.'s and ','s
    outString = outString.replace(' ', '_').replace('.', '_').replace(',', '')
    # and remove / and \
    outString = outString.replace('/', '_').replace('\\', '_')
    # and remove parentheses
    outString = outString.replace('(', '').replace(')', '')
    # Remove ':' and ';"
    outString = outString.replace(':','_').replace(';','_')
    # Remove '__'
    while '__' in outString:
        outString = outString.replace('__','_')

    return outString
