import sys
import json
from collections import defaultdict, Counter
import itertools

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.models.ensemble import Ensemble

from tqdm import tqdm
import numpy as np

sys.path.append('../num_fh/resolution/framework/models/')
sys.path.append('../num_fh/resolution/framework/dataset_readers/')
sys.path.append('../num_fh/resolution/framework/predictors/')
from src.framework.models.deep_moji_model import DeepMojiModel
from src.framework.dataset_readers.deep_moji_reader import DeepMojiReader
from src.framework.predictors
from model_base import NfhDetector
from nfh_reader import NFHReader
from model_base_predictor import NfhDetectorPredictor