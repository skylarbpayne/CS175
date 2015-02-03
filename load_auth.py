def load_auth(filename: str) -> dict:
    '''
        This loads authorization data from a file in the format of name:value
        If the data in your file is not in the correct format, all other scripts WILL fail!
    '''
    with open(filename, 'r') as auth_file:
        return {l[0].strip(): l[1].strip() for l in [line.split(':') for line in auth_file.readlines()]}
