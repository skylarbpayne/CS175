def tokenize(x: str) -> list:
    '''
    Splits a string into a list of tokens.

    :param x: the string to tokenize
    
    :return : the list of tokens
    '''

    return x.strip().split()

if __name__ == '__main__':
    assert(tokenize('hey john') == ['hey', 'john'])
