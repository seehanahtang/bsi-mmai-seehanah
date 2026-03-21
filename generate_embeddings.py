import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:512'

from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import gc
import warnings
warnings.filterwarnings('ignore')

import sys
import time
import pandas as pd
import gc
import numpy as np
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from typing import List
from functools import reduce
from transformers import AutoTokenizer, AutoModel, logging
from tqdm import tqdm
import pyodbc

#from embeddings_model_language import *

def generate_embeddings(spec = "TruncatedNotes", suffix = "most_recent_3", load = False):
    """
    Generates embeddings for text.
    """

    if load:
        # Load from huggingface and save
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        tokenizer.save_pretrained("/datafs_a/haim5_helper/CBERT-pretrained")
        model.save_pretrained("/datafs_a/haim5_helper/CBERT-pretrained")

    notes = pd.read_csv(f'/datafs_a/carolgao/haim14_data/blood_infection/notes_matched_{suffix}.csv')

    # Memory stuff
    with torch.no_grad():
        torch.cuda.empty_cache()

    gc.collect()

    # Generate embeddings

    tokenizer = AutoTokenizer.from_pretrained("/datafs_a/carolgao/CBERT-pretrained")
    model = AutoModel.from_pretrained("/datafs_a/carolgao/CBERT-pretrained")
    model = model.to('cuda')

    def tokenize_and_reshape(text, max_length=512):
        # Tokenize texts without splitting into chunks first
        encoded = tokenizer(text, return_tensors="pt")

        # Determine padding necessary to make input_ids length a multiple of max_length
        current_length = encoded.input_ids.size(1)
        padding_needed = (max_length - current_length % max_length) % max_length

        # Apply padding using torch.nn.functional.pad
        pad_config = (0, padding_needed)
        padded_input_ids = F.pad(encoded.input_ids, pad_config, value=tokenizer.pad_token_id)
        padded_token_type_ids = F.pad(encoded.token_type_ids, pad_config, value=0)
        padded_attention_mask = F.pad(encoded.attention_mask, pad_config, value=0)

        # Reshape the tensors to have sequences of max_length and send to cuda
        chunks = {
            'input_ids': padded_input_ids.reshape(-1, max_length).to('cuda'),
            'token_type_ids': padded_token_type_ids.reshape(-1, max_length).to('cuda'),
            'attention_mask': padded_attention_mask.reshape(-1, max_length).to('cuda')
        }

        return chunks


    def get_embedding(text):
        model.eval()
        with torch.no_grad():
            tokens = tokenize_and_reshape(text)

            # Obtain embeddings from model
            output = model(**tokens).pooler_output

            note_embedding = output.mean(dim=0).detach().to('cpu').numpy()

        torch.cuda.empty_cache()

        return note_embedding

    def obtain_embeddings(spec):
        # Replace missing values with empty string, important for BioBert error handling
        notes[spec] = notes[spec].replace(to_replace=np.nan, value='')

        all_embeddings = np.array([get_embedding(x) for x in tqdm(notes[spec], desc="Embedding notes")])
        embeddings_df = pd.DataFrame(
            all_embeddings,
            columns=[f'cn_{i}' for i in range(all_embeddings.shape[1])]
        )

        processed = pd.concat(
                [notes[['EncounterKey',]], embeddings_df],
                axis=1
            )
        processed.to_csv(f'/datafs_a/carolgao/haim14_data/blood_infection/blood_processed_notes_{spec}_{suffix}.csv', index=False)

        del embeddings_df

    obtain_embeddings(spec)