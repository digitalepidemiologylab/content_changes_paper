POSITIVE_FILTER=r'\b(?<!\")(?:I|We)(?: have |\'ve |ve | just | )tested positive for (?:covid|corona|sars-cov-2)|\b(?<!\")(?:my|our) (?:covid|corona|sars-cov-2) test is positive|\b(?<!\")(?:found out|turns out|confirms|confirm) I(?:ve|\'ve| have| got| contracted) (?:covid|corona|sars-cov-2)'
VACCINE_FILTER=r'\b(?:I am getting|I\'m getting) (?:(?:vaccinated|vaxxed)|(?:my|the|a)) (?:[^.\(\;\[\{]{0,30}?(?:vaccine|vaccination)|(?:pfizer|moderna|astrazeneca|johnson&johnson|johnson & johnson) (?:shot|vaccine))|\bI(?: have |\'ve | )(?:got|received|took) (?:[^.\(\;\[\{]{0,30}?vaccin|(?:(?:a |the )pfizer|moderna|astrazeneca|johnson&johnson|johnson & johnson) (?:shot|vaccine))'
PERSONAL_FILTER=r'\bI(?: am|\'m| have|\'ve) [^.\(\;\[\{]{0,30}?'
SYM_FILTER={
    'fever': PERSONAL_FILTER+r'fever ',
    'cough': PERSONAL_FILTER+r'cough',
    'body_pain': r'(?:'+PERSONAL_FILTER+r'body pain |\bmy body [^.\(\;\[\{]{0,30}?pain)'
}
