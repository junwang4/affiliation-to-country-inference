MODEL_DIR = '/tmp/models'

DEBUG = False
#DEBUG = True

DEFAULT_FOLD_FOR_CALIBRATED_CLASSIFIER = 3

# 50 countries
COUNTRIES_OF_INTEREST = [
'US', 'DE', 'JP', 'IT', 'TR', 'SE', 'AU', 'IN', 'ES', 'FR',
'NL', 'GB', 'CA', 'CN', 'KR', 'BR', 'CH', 'TW', 'BE', 'PL',
'DK', 'IL', 'IR', 'GR', 'AT', 'FI', 'NO', 'RU', 'MX', 'PT',
'HK', 'AR', 'IE', 'ZA', 'NZ', 'CZ', 'EG', 'HU', 'TH', 'SG',
'SA', 'MY', 'NG', 'CL', 'PK', 'RO', 'HR', 'CO', 'RS', 'VN',
]

"""
# 150 countries
COUNTRIES_OF_INTEREST = [
'US', 'DE', 'JP', 'IT', 'TR', 'SE', 'AU', 'IN', 'ES', 'FR',
'NL', 'GB', 'CA', 'CN', 'KR', 'BR', 'CH', 'TW', 'BE', 'PL',
'DK', 'IL', 'IR', 'GR', 'AT', 'FI', 'NO', 'RU', 'MX', 'PT',
'HK', 'AR', 'IE', 'ZA', 'NZ', 'CZ', 'EG', 'HU', 'TH', 'SG',
'SA', 'MY', 'NG', 'CL', 'PK', 'RO', 'HR', 'CO', 'RS', 'SI',
'SK', 'ET', 'TN', 'BG', 'ID', 'UA', 'LT', 'JO', 'MA', 'LB',
'BD', 'NP', 'KE', 'VE', 'CU', 'KW', 'EE', 'VN', 'LK', 'PE',
'AE', 'UY', 'GH', 'TZ', 'OM', 'UG', 'IS', 'PH', 'IQ', 'QA',
'CM', 'BA', 'SN', 'DZ', 'PR', 'CY', 'LU', 'GE', 'SD', 'ZW',
'LV', 'CR', 'BY', 'ME', 'JM', 'CI', 'EC', 'BF', 'CG', 'MW',
'SY', 'MT', 'BH', 'PS', 'AM', 'KZ', 'ZM', 'PA', 'TT', 'MG',
'BJ', 'LY', 'PG', 'TG', 'ML', 'GA', 'YE', 'BW', 'GM', 'GT',
'AL', 'UZ', 'MZ', 'XK', 'KH', 'RW', 'BN', 'BB', 'MC', 'NE',
'BO', 'GF', 'PY', 'MD', 'MM', 'GP', 'AZ', 'NI', 'MN', 'NC',
'NA', 'MU', 'FJ', 'MQ', 'AF', 'HN', 'HT', 'DO', 'PF', 'SL']

# 249 countries // this will generate a huge LinearSVC model (could be 10G - 20G if not compressed)
COUNTRIES_OF_INTEREST = ['BR', 'US', 'IN', 'ES', 'GB', 'CN', 'CO', 'RU', 'MX',
'PT', 'FR', 'DE', 'IT', 'PE', 'AU', 'TR', 'UA', 'JP', 'PL', 'ID', 'IR', 'CA',
'ZA', 'KR', 'NG', 'AR', 'PK', 'CH', 'BE', 'EG', 'NL', 'CL', 'EC', 'MY', 'SE',
'PH', 'CU', 'IQ', 'BD', 'DK', 'AT', 'GR', 'CZ', 'RO', 'VE', 'TW', 'IE', 'FI',
'SA', 'VN', 'KE', 'RS', 'NO', 'ET', 'DZ', 'NP', 'TH', 'HU', 'HR', 'KZ', 'NZ',
'MA', 'SG', 'GH', 'LK', 'IL', 'BG', 'HK', 'TN', 'AE', 'SK', 'JO', 'UZ', 'UG',
'CR', 'CM', 'UY', 'TZ', 'BY', 'LB', 'SI', 'PY', 'SD', 'BO', 'PA', 'LT', 'MZ',
'ZW', 'DO', 'BA', 'CY', 'PR', 'AZ', 'GE', 'LV', 'AL', 'RW', 'QA', 'LU', 'AO',
'MK', 'PS', 'GT', 'LY', 'AM', 'EE', 'SY', 'OM', 'ZM', 'MW', 'MD', 'CD', 'MN',
'NI', 'MM', 'YE', 'IS', 'AF', 'HN', 'KW', 'SV', 'KG', 'SN', 'BJ', 'MT', 'BF',
'CI', 'BW', 'FJ', 'KH', 'MG', 'BH', 'BT', 'SO', 'NA', 'ML', 'CV', 'TT', 'SZ',
'JM', 'TJ', 'MO', 'SL', 'ME', 'LR', 'HT', 'MU', 'TG', 'PG', 'ER', 'GM', 'NE',
'BN', 'TL', 'GN', 'CG', 'BI', 'LS', 'SS', 'GA', 'LA', 'RE', 'BB', 'TD', 'MV',
'BS', 'NC', 'GD', 'GW', 'VA', 'UM', 'AD', 'LI', 'VI', 'GY', 'PF', 'MC', 'BZ',
'GF', 'GP', 'AS', 'MR', 'KP', 'SC', 'ST', 'VU', 'SR', 'MQ', 'CF', 'CW', 'FO',
'TM', 'KN', 'XK', 'KM', 'WS', 'GQ', 'BQ', 'DM', 'BM', 'GL', 'DJ', 'JE', 'SB',
'KY', 'GI', 'IM', 'AQ', 'IO', 'SM', 'AG', 'AW', 'LC', 'GU', 'VC', 'TO', 'VG',
'AX', 'KI', 'CX', 'MP', 'BV', 'SX', 'FM', 'TC', 'AI', 'TV', 'FK', 'PW', 'MS',
'PN', 'CK', 'NR', 'YT', 'GS', 'SJ', 'GG', 'NF', 'MF', 'MH', 'NU', 'BL', 'EH',
'TF', 'WF', 'CC', 'TK', 'SH', 'HM', 'PM']

"""
