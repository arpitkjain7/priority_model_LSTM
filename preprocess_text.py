import pandas as pd


def remove_punctuations(df):
    punctuations = '''|+!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for idx, values in enumerate(df):
        no_punct = ""
        for char in values:
            if char not in punctuations:
                no_punct = no_punct + char

        # display the unpunctuated string
        df[idx] = no_punct.strip()
    return df
