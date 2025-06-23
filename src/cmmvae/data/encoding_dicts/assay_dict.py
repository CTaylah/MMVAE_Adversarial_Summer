import random

assay = { 
"DroNc-seq": 0, 
"microwell-seq": 1, 
"ScaleBio single cell RNA sequencing": 2, 
"Seq-Well": 3, 
"BD Rhapsody Whole Transcriptome Analysis": 4, 
"TruDrop": 5, 
"10x 3' v2": 6, 
"Smart-seq v4": 7, 
"10x 3' v3": 8, 
"10x 5' v2": 9, 
"10x 5' transcription profiling": 10, 
"Smart-seq": 11, 
"10x 3' v1": 12, 
"sci-RNA-seq": 13, 
"BD Rhapsody Targeted mRNA": 14, 
"STRT-seq": 15, 
"10x 5' v1": 16, 
"10x 3' transcription profiling": 17, 
"SPLiT-seq": 18, 
"CEL-seq2": 19, 
"GEXSCOPE technology": 20, 
"Smart-seq2": 21, 
"inDrop": 22, 
"Drop-seq": 23, 
"MARS-seq": 24, 
"Seq-Well S3": 25, 
} 

def get_random_assay() -> str:
    """
    Get a random assay name from the assay dictionary.

    Returns:
        str: A random assay name.
    """
    return random.choice(list(assay.keys()))