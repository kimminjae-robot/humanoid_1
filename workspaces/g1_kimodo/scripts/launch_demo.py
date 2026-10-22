"""Official Kimodo G1 UI; GPU motion model and pinned CPU Llama encoder."""
import os,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEMO_DEVICE=os.environ.get('G1_DEMO_DEVICE','cuda')
if DEMO_DEVICE not in ('cpu','cuda'):raise ValueError('G1_DEMO_DEVICE must be cpu or cuda')
if DEMO_DEVICE=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HOME']=str(ROOT/'cache/huggingface')
os.environ['CHECKPOINT_DIR']=str(ROOT/'checkpoints')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['SERVER_NAME']='127.0.0.1'
os.environ['SERVER_PORT']='7860'
os.environ['kimodo_EMBED_CACHE_DIR']=str(ROOT/'cache/demo_embeddings')
sys.path.insert(0,str(ROOT/'scripts'))
import torch
torch.set_num_threads(4)
if DEMO_DEVICE=='cuda' and not torch.cuda.is_available():
    raise RuntimeError('GPU mode requested but CUDA is unavailable')
from official_cached_encoder import CachedOfficialEncoder

class InteractiveOfficialEncoder:
    llm_dim=4096
    def __init__(self):
        self.cached=CachedOfficialEncoder();self.encoder=None;self.lock=threading.Lock()
    def __call__(self,texts):
        with self.lock:
            try:return self.cached(texts)
            except KeyError:pass
            if self.encoder is None:
                print('Loading pinned official Llama encoder on CPU for new prompts...',flush=True)
                from official_encoder import PinnedLLM2VecEncoder
                self.encoder=PinnedLLM2VecEncoder()
            return self.encoder(texts)
    def get_device(self):return torch.device('cpu')
    def to(self,*args,**kwargs):return self
    def eval(self):return self

from kimodo.demo import app
original_load_model=app.load_model
encoder=InteractiveOfficialEncoder()
def load_interactive(*args,**kwargs):
    if kwargs.get('text_encoder') is None:kwargs['text_encoder']=encoder
    return original_load_model(*args,**kwargs)
app.load_model=load_interactive
class InteractiveDemo(app.Demo):
    def prewarm_embedding_cache(self,*args,**kwargs):
        # Defer the large text encoder until the user submits a new prompt.
        pass

if __name__=='__main__':
    print(f'Official G1 authoring UI: http://127.0.0.1:7860 ({DEMO_DEVICE} motion model, pinned CPU encoder)',flush=True)
    InteractiveDemo(default_model_name='Kimodo-G1-RP-v1').run()
