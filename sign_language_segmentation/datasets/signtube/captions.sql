SELECT *
FROM captions
WHERE "start" != 0
  AND (
    "videoId" LIKE 'swn%' OR -- Sign Word Net https://www.sign-lang.uni-hamburg.de/easier/sign-wordnet/
    "videoId" LIKE 'fasl%' OR -- Fleurs ASL https://www.kaggle.com/datasets/googleai/fleurs-asl
    "videoId" LIKE 's2m%' OR -- Sign2MINT https://sign2mint.de/
    "videoId" LIKE 'mfasl%' OR -- 2M-Flores-ASL https://huggingface.co/datasets/facebook/2M-Flores-ASL
    "videoId" LIKE 'dgstypes%' OR -- DGS Types https://www.sign-lang.uni-hamburg.de/meinedgs/ling/types_en.html
    "videoId" LIKE 'dictio%' OR -- Dictio https://www.dictio.info/
    "videoId" LIKE 'isl%' OR -- ISL Dictionary https://isl.danfishgold.com/
    "videoId" LIKE 'ss%' OR -- SignSuisse https://signsuisse.sgb-fss.ch/
    "videoId" LIKE 'sts%' OR -- Spread The Sign https://spreadthesign.com/
    "videoId" LIKE 'fs-sts%' OR -- Spread The Sign Fingerspelling https://spreadthesign.com/
    "videoId" LIKE 'whatsthatsign%' -- What's That Sign https://whatsthatsign.com/
    )
