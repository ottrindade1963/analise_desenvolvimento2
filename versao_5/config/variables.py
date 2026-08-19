"""config/variables.py — Dataset variables: countries, indicators, target."""

TARGET = "valor_agregado_industrial_percent_pib"

COUNTRIES = {
    "DZA": "Argélia",   "EGY": "Egito",    "LBY": "Líbia",
    "MAR": "Marrocos",  "TUN": "Tunísia",
    "BHR": "Bahrein",   "IRN": "Irão",     "IRQ": "Iraque",
    "JOR": "Jordânia",  "KWT": "Kuwait",   "LBN": "Líbano",
    "OMN": "Omã",       "QAT": "Qatar",    "SAU": "Arábia Saudita",
    "ARE": "Emirados Árabes Unidos",        "YEM": "Iémen",
    "TUR": "Turquia",   "ZAF": "África do Sul",
    "NGA": "Nigéria",   "KEN": "Quénia",   "GHA": "Gana",
    "ETH": "Etiópia",   "TZA": "Tanzânia", "CIV": "Costa do Marfim",
    "SEN": "Senegal",   "CMR": "Camarões", "AGO": "Angola",
    "MOZ": "Moçambique","UGA": "Uganda",   "RWA": "Ruanda",
    "COD": "RD Congo",  "ZMB": "Zâmbia",   "BWA": "Botsuana",
    "MUS": "Maurícia",  "NAM": "Namíbia",  "GAB": "Gabão",
    "MDG": "Madagáscar",
}
COUNTRY_CODES = list(COUNTRIES.keys())

WDI_INDICATORS = {
    "NV.IND.TOTL.ZS":    TARGET,
    "NY.GDP.PCAP.PP.KD": "pib_per_capita_ppc",
    "NY.GDP.MKTP.KD.ZG": "crescimento_pib_anual",
    "NE.GDI.FTOT.ZS":    "formacao_bruta_capital_fixo_percent_pib",
    "BX.KLT.DINV.WD.GD.ZS": "ied_percent_pib",
    "NE.TRD.GNFS.ZS":    "comercio_percent_pib",
    "NE.EXP.GNFS.ZS":    "exportacoes_percent_pib",
    "NE.IMP.GNFS.ZS":    "importacoes_percent_pib",
    "EG.USE.ELEC.KH.PC": "consumo_eletricidade_per_capita",
    "IT.NET.USER.ZS":    "utilizadores_internet_percent",
    "GB.XPD.RSDV.GD.ZS": "despesa_id_percent_pib",
    "SE.XPD.TOTL.GD.ZS": "despesa_educacao_percent_pib",
    "SL.TLF.TOTL.IN":    "forca_trabalho_total",
    "SE.TER.ENRR":       "taxa_matricula_terciario",
    "SP.URB.TOTL.IN.ZS": "populacao_urbana_percent",
    "SP.POP.GROW":       "crescimento_populacional",
    "FS.AST.PRVT.GD.ZS": "credito_privado_percent_pib",
    "NV.IND.MANF.ZS":    "valor_agregado_manufatura_percent_pib",
    "NY.GDP.TOTL.RT.ZS": "rendas_recursos_naturais_percent_pib",
}

WGI_INDICATORS = {
    "CC.EST": "wgi_controle_corrupcao",
    "GE.EST": "wgi_eficacia_governo",
    "PV.EST": "wgi_estabilidade_politica",
    "RQ.EST": "wgi_qualidade_regulatoria",
    "RL.EST": "wgi_estado_direito",
    "VA.EST": "wgi_voz_responsabilizacao",
}

WGI_COLS = list(WGI_INDICATORS.values())
WDI_COLS = [v for v in WDI_INDICATORS.values() if v != TARGET]

YEAR_START = 1996
YEAR_END   = 2023

REGIONS = {
    "Norte de África":  ["DZA","EGY","LBY","MAR","TUN"],
    "Médio Oriente":    ["BHR","IRN","IRQ","JOR","KWT","LBN","OMN","QAT","SAU","ARE","YEM","TUR"],
    "África Ocidental": ["GHA","NGA","SEN","CIV","CMR"],
    "África Oriental":  ["ETH","KEN","TZA","UGA","RWA","MOZ"],
    "África Austral":   ["ZAF","BWA","NAM","ZMB","GAB","MUS","MDG","AGO","COD"],
}
